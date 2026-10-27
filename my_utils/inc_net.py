"""Main-table ViT branches; unused experimental network variants are omitted."""
import copy
import logging
import torch
from torch import nn
from torch.nn import functional as F
from convs.linears import CosineLinear
import timm


def get_convnet(args, pretrained=False):
    name = args['convnet_type'].lower()
    if name in ('pretrained_vit_b16_224_in21k', 'vit_base_patch16_224_in21k'):
        model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    if name != 'pretrained_vit_b16_224_in21k_adapter':
        raise NotImplementedError('Only the main-table ImageNet-21K ViT-B/16 is included.')
    from convs import vision_transformer_adapter
    from easydict import EasyDict
    tuning_config = EasyDict(
        ffn_adapt=True,
        ffn_option='parallel',
        ffn_adapter_layernorm_option='none',
        ffn_adapter_init_option='lora',
        ffn_adapter_scalar='0.1',
        ffn_num=args['ffn_num'],
        d_model=768,
        vpt_on=False,
        vpt_num=0,
    )
    model = vision_transformer_adapter.vit_base_patch16_224_in21k_adapter(
        num_classes=0, global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
    model.out_dim = 768
    return model.eval()


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()


        print('This is for the BaseNet initialization.')
        self.convnet = get_convnet(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None
        self.device = args["device"][0]

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class SimpleCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc


class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self.device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.convnet(x)

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x)
        out.update({'features': x})
        return out


class MultiBranchCosineIncrementalNet_v5(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)


        print(
            'Clear the convnet in MultiBranchCosineIncrementalNet, since we are using self.convnets with dual branches')
        self.convnet = torch.nn.Identity()
        for param in self.convnet.parameters():
            param.requires_grad = False

        self.convnets = nn.ModuleList()
        self.args = args
        self.prototype_manager = None

        if 'resnet' in args['convnet_type']:
            self.modeltype = 'cnn'
        else:
            self.modeltype = 'vit'

    def set_prototype_manager(self, prototype_manager):
        self.prototype_manager = prototype_manager

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self._feature_dim, nb_classes).to(self.device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self._feature_dim).to(self.device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        if self.modeltype == 'cnn':
            features = [convnet(x)["features"] for convnet in self.convnets]
            features = torch.cat(features, 1)
        else:
            features = [convnet(x) for convnet in self.convnets]
            features = torch.cat(features, 1)
        return features

    def forward(self, x):
        if self.modeltype == 'cnn':
            features = [convnet(x)["features"] for convnet in self.convnets]
            features = torch.cat(features, 1)
        else:
            features = [convnet(x) for convnet in self.convnets]
            features = torch.cat(features, 1)

        out = self._forward_with_multi_prototypes(features)
        out.update({"features": features})
        return out

    def forward_with_fc(self, x):
        """Compute logits with the classification head."""
        if self.modeltype == 'cnn':
            features = [convnet(x)["features"] for convnet in self.convnets]
            features = torch.cat(features, 1)
        else:
            features = [convnet(x) for convnet in self.convnets]
            features = torch.cat(features, 1)

        out = self.fc(features)
        out.update({"features": features})
        return out

    def _forward_with_multi_prototypes(self, features):
        """Score each class by its maximum prototype cosine similarity."""
        if self.prototype_manager is None:
            return self.fc(features)

        batch_size = features.size(0)
        num_classes = self.fc.out_features

        features_norm = F.normalize(features, p=2, dim=1)
        logits = torch.zeros(batch_size, num_classes).to(features_norm.device)
        all_prototypes = self.prototype_manager.get_all_prototypes()

        for class_idx, prototypes in all_prototypes.items():
            if class_idx >= num_classes:
                continue

            class_similarities = []
            for proto in prototypes:
                proto = proto.to(features_norm.device)
                proto_norm = F.normalize(proto.unsqueeze(0), p=2, dim=1)
                similarity = torch.matmul(features_norm, proto_norm.transpose(0, 1)).squeeze(1)
                class_similarities.append(similarity)

            if class_similarities:
                class_similarities = torch.stack(class_similarities, dim=1)
                max_similarity, _ = torch.max(class_similarities, dim=1)
                logits[:, class_idx] = max_similarity * self.fc.sigma

        for class_idx in range(num_classes):
            if class_idx not in all_prototypes:
                weight = self.fc.weight[class_idx].unsqueeze(0)
                weight_norm = F.normalize(weight, p=2, dim=1)
                similarity = torch.matmul(features_norm, weight_norm.transpose(0, 1)).squeeze(1)
                logits[:, class_idx] = similarity * self.fc.sigma

        return {'logits': logits}

    def freeze_conv(self):
        for param in self.convnets[0].parameters():
            param.requires_grad = False
        self.convnets[0].eval()

    def construct_dual_branch_network(self, tuned_model):
        if 'ssf' in self.args['convnet_type']:
            newargs = copy.deepcopy(self.args)
            newargs['convnet_type'] = newargs['convnet_type'].replace('_ssf', '')
            print(newargs['convnet_type'])
            self.convnets.append(get_convnet(newargs))
        elif 'vpt' in self.args['convnet_type']:
            newargs = copy.deepcopy(self.args)
            newargs['convnet_type'] = newargs['convnet_type'].replace('_vpt', '')
            print(newargs['convnet_type'])
            self.convnets.append(get_convnet(newargs))
        elif 'adapter' in self.args['convnet_type']:
            newargs = copy.deepcopy(self.args)
            newargs['convnet_type'] = newargs['convnet_type'].replace('_adapter', '')
            print(newargs['convnet_type'])
            self.convnets.append(get_convnet(newargs))
        else:
            self.convnets.append(get_convnet(self.args))

        self.convnets.append(tuned_model.convnet)

        self._feature_dim = self.convnets[0].out_dim * len(self.convnets)
        self.fc = self.generate_fc(self._feature_dim, self.args['init_cls'])
        self.freeze_conv()
