#!/usr/bin/env python3
"""Prepare a tiny real-world CDIL stream from iWildCam 2022."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import struct
import sys
import time
import zipfile
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ZIP_URL = "https://storage.googleapis.com/public-datasets-lila/iwildcam-2022/iwildcam-2022.zip"
ZIP_PREFIX = "iwildcam-2022/"
TRAIN_ANNOTATIONS = "iwildcam-2022/metadata/iwildcam2022_train_annotations.json"
GPS_LOCATIONS = "iwildcam-2022/metadata/gps_locations.json"


@dataclass
class ZipEntry:
    name: str
    compress_type: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int


def request_range(url: str, start: int, end: int, timeout: int = 120, retries: int = 4) -> bytes:
    headers = {"Range": f"bytes={start}-{end}", "User-Agent": "EMCift-iWildCamTiny/1.0"}
    last_error = None
    for attempt in range(retries):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout) as resp:
                expected = end - start + 1
                if resp.status != 206 or resp.headers.get("Content-Range", "").split("/")[0] != f"bytes {start}-{end}":
                    raise RuntimeError("Server did not honor the requested byte range; refusing full download")
                data = resp.read(expected + 1)
            expected = end - start + 1
            if len(data) != expected:
                raise RuntimeError(f"range {start}-{end} returned {len(data)} bytes, expected {expected}")
            return data
        except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to fetch byte range {start}-{end}: {last_error}")


def get_content_length(url: str) -> int:
    req = Request(url, method="HEAD", headers={"User-Agent": "EMCift-iWildCamTiny/1.0"})
    with urlopen(req, timeout=60) as resp:
        return int(resp.headers["Content-Length"])


def parse_zip64_directory_info(url: str) -> Tuple[int, int, int]:
    size = get_content_length(url)
    tail_start = max(0, size - 65536)
    tail = request_range(url, tail_start, size - 1)
    locator_sig = b"PK\x06\x07"
    locator_pos = tail.rfind(locator_sig)
    if locator_pos < 0:
        raise RuntimeError("Zip64 locator not found")
    _, _, zip64_eocd_offset, _ = struct.unpack_from("<4sLQL", tail, locator_pos)
    eocd = request_range(url, zip64_eocd_offset, zip64_eocd_offset + 97)
    if eocd[:4] != b"PK\x06\x06":
        raise RuntimeError("Zip64 EOCD signature not found")
    fields = struct.unpack_from("<4sQ2H2L4Q", eocd, 0)
    entries_total = fields[7]
    cd_size = fields[8]
    cd_offset = fields[9]
    return cd_offset, cd_size, entries_total


def parse_central_directory(cd: bytes) -> Dict[str, ZipEntry]:
    entries: Dict[str, ZipEntry] = {}
    offset = 0
    sig = b"PK\x01\x02"
    while offset < len(cd):
        if cd[offset : offset + 4] != sig:
            raise RuntimeError(f"central directory parse failed at offset {offset}")
        fields = struct.unpack_from("<4s6H3L5H2L", cd, offset)
        (
            _,
            _ver_made,
            _ver_needed,
            _flags,
            compress_type,
            _mtime,
            _mdate,
            _crc,
            compressed_size,
            uncompressed_size,
            name_len,
            extra_len,
            comment_len,
            _disk,
            _int_attr,
            _ext_attr,
            local_header_offset,
        ) = fields
        name_start = offset + 46
        name = cd[name_start : name_start + name_len].decode("utf-8")
        extra = cd[name_start + name_len : name_start + name_len + extra_len]

        if (
            compressed_size == 0xFFFFFFFF
            or uncompressed_size == 0xFFFFFFFF
            or local_header_offset == 0xFFFFFFFF
        ):
            values = []
            extra_pos = 0
            while extra_pos + 4 <= len(extra):
                header_id, data_size = struct.unpack_from("<HH", extra, extra_pos)
                data = extra[extra_pos + 4 : extra_pos + 4 + data_size]
                if header_id == 0x0001:
                    for i in range(0, len(data), 8):
                        if i + 8 <= len(data):
                            values.append(struct.unpack_from("<Q", data, i)[0])
                    break
                extra_pos += 4 + data_size
            vi = 0
            if uncompressed_size == 0xFFFFFFFF:
                uncompressed_size = values[vi]
                vi += 1
            if compressed_size == 0xFFFFFFFF:
                compressed_size = values[vi]
                vi += 1
            if local_header_offset == 0xFFFFFFFF:
                local_header_offset = values[vi]

        entries[name] = ZipEntry(
            name=name,
            compress_type=compress_type,
            compressed_size=compressed_size,
            uncompressed_size=uncompressed_size,
            local_header_offset=local_header_offset,
        )
        offset = name_start + name_len + extra_len + comment_len
    return entries


def download_index(url: str, cache_path: Path) -> Dict[str, ZipEntry]:
    if cache_path.exists():
        cd = cache_path.read_bytes()
    else:
        cd_offset, cd_size, entries_total = parse_zip64_directory_info(url)
        print(f"[index] central_dir_offset={cd_offset} size={cd_size} entries={entries_total}", flush=True)
        cd = request_range(url, cd_offset, cd_offset + cd_size - 1, timeout=240)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(cd)
    return parse_central_directory(cd)


def extract_member(url: str, entries: Dict[str, ZipEntry], name: str) -> bytes:
    entry = entries[name]
    header = request_range(url, entry.local_header_offset, entry.local_header_offset + 30 - 1)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"bad local header for {name}")
    fields = struct.unpack_from("<4s5H3L2H", header, 0)
    name_len, extra_len = fields[-2], fields[-1]
    data_start = entry.local_header_offset + 30 + name_len + extra_len
    compressed = request_range(url, data_start, data_start + entry.compressed_size - 1, timeout=240)
    if entry.compress_type == zipfile.ZIP_STORED:
        return compressed
    if entry.compress_type == zipfile.ZIP_DEFLATED:
        return zlib.decompress(compressed, -15)
    raise RuntimeError(f"unsupported compression {entry.compress_type} for {name}")
