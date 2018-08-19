import torch
import torch.nn as nn
import torchvision

default_mean = [0.485, 0.456, 0.406]
default_std = [0.229, 0.224, 0.225]

class Normalization(torch.nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(Normalization, self).__init__()

        mean = torch.FloatTensor(mean).view(-1, 1, 1)
        std = torch.FloatTensor(std).view(-1, 1, 1)

        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        return (x - self.mean) / self.std

    def recover(self, x):
        return (x * self.std + self.mean).clamp(0, 1)


class FeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

    def add_layer(self, name, layer):
        self.add_module(name, layer)

    def forward(self, x, layers):
        feats = []
        end = max(layers)
        for i, module in enumerate(self._modules):
            x = self._modules[module](x)
            if i in layers:
                feats.append(x)
            if i == end:
                break
        return feats
        
def merge_list(lst):
    res = []
    for l in lst:
        res.extend(l)
    return res

class DeepFeature(nn.Module):
    def __init__(self, base_model='vgg19'):
        super(DeepFeature, self).__init__()

        # build model
        vgg19_model = getattr(torchvision.models, base_model)(pretrained=True)
        self.cnn_temp = vgg19_model.features
        self.model = FeatureExtractor()  # the new Feature extractor module network
        conv_counter = 1
        relu_counter = 1
        batn_counter = 1

        block_counter = 1
        self.stage2layer = {}
        self.layer2num = {}

        for i, layer in enumerate(list(self.cnn_temp)):

            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(block_counter) + "_" + str(conv_counter) + "__" + str(i)
                conv_counter += 1
                self.model.add_layer(name, layer)                

            if isinstance(layer, nn.ReLU):
                name = "relu_" + str(block_counter) + "_" + str(relu_counter) + "__" + str(i)
                if relu_counter == 1:
                    self.stage2layer[block_counter] = i
                relu_counter += 1
                self.model.add_layer(name, nn.ReLU(inplace=False))

            if isinstance(layer, nn.MaxPool2d):
                name = "pool_" + str(block_counter) + "__" + str(i)
                batn_counter = relu_counter = conv_counter = 1
                block_counter += 1
                self.model.add_layer(name, nn.MaxPool2d((2, 2), ceil_mode=True))  # ***

            if isinstance(layer, nn.BatchNorm2d):
                name = "batn_" + str(block_counter) + "_" + str(batn_counter) + "__" + str(i)
                batn_counter += 1
                self.model.add_layer(name, layer)  # ***

            self.layer2num[name.split('__')[0]] = i

        self.model.eval()

        # normalization
        self.norm = Normalization()


    def get_feat_with_layer(self, x, layers=None):
        x = self.norm(x).contiguous()
        if layers is None:
            layers = list(range(len(self.model)))
        return self.model(x, layers)

    def forward(self, x, layers, detach=False):
        elem = layers[0]
        if isinstance(elem, str):
            target_layers = [self.layer2num[s] for s in layers]
        elif isinstance(elem, str):
            target_layers = layers
        elif isinstance(elem, list):
            target_layers = list(set(merge_list(layers)))
            target_layers.sort()
            distribute_layers = []
            for e in layers:
                distribute_layers.append([target_layers.index(i) for i in e])
            if isinstance(target_layers[0], str):
                target_layers = [self.layer2num[s] for s in target_layers]

        features = self.get_feat_with_layer(x, target_layers)
        if detach:
            features = [f.detach() for f in features]

        if isinstance(elem, list):
            feats = []
            for e in distribute_layers:
                feats.append([features[i] for i in e])
            features = feats

        return features


if __name__ == '__main__':
    from PIL import Image
    img = torchvision.transforms.ToTensor()(Image.open('image.jpg')).unsqueeze(0)

    extractor = DeepFeature('vgg19')
    feats = extractor(img)

    for f in feats:
        print(f.shape)