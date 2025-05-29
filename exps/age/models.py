import torch
import torchvision
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

# from masksembles.torch import Masksembles1D
import copy

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.elu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.elu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.elu(self.bn1(self.conv1(x)))
        out = F.elu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.elu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.linear = nn.Linear(6400, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.elu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 5)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


class Net(torch.nn.Module):

  def __init__(self, classes_num=10, p=0.0, init_ch=3):

    super().__init__()
    
    self.activation = torch.nn.LeakyReLU()

    resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    init_conv = torch.nn.Conv2d(init_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    modules = [init_conv] + list(resnet.children())[1:-1]
    
#     modules = [
#         modules[0],
#         modules[1],
#         torch.nn.LeakyReLU(),
#         torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
#     ] + modules[4:]

#     self.extractor = torch.nn.Sequential(*modules, torch.nn.AvgPool2d(2), 
#                                          torch.nn.Conv2d(512, 512//9, 1), torch.nn.LeakyReLU())

    self.extractor = torch.nn.Sequential(*modules)
    
    CH = 128
    self.fc1 = torch.nn.Linear(512, CH)
    self.fc2 = torch.nn.Linear(CH, CH)
    self.fc3 = torch.nn.Linear(CH, CH)
    self.fc4 = torch.nn.Linear(CH, CH)
    self.fc5 = torch.nn.Linear(CH, classes_num)

    self.dp = torch.nn.functional.dropout
    self.p = p

    self.classes_num = classes_num

  def forward(self, x, features=False):
    feat = self.activation(self.extractor(x)).squeeze(2).squeeze(2) #view(-1, 504) 
    x = self.activation(self.fc1(feat))
    x = self.dp(self.activation(self.fc2(x)) + x, p=self.p, training=True)
    x = self.dp(self.activation(self.fc3(x)) + x, p=self.p, training=True)
    x = self.dp(self.activation(self.fc4(x)) + x, p=self.p, training=True)
    if features: 
        return self.fc5(x), feat
    return self.fc5(x)



###### Varprop ######

class VarPropModule(torch.nn.Module):
    
    def __init__(self, CH=64, out_channel=1):
        
        super(VarPropModule, self).__init__()
        
        self.activation = torch.nn.ReLU()
        
        self.CH = CH

        self.fc1 = torch.nn.Linear(512, CH)
        self.fc2 = torch.nn.Linear(CH, CH)
        self.fc3 = torch.nn.Linear(CH, CH)
        self.fc4 = torch.nn.Linear(CH, CH)
        self.fc5 = torch.nn.Linear(CH, out_channel)
        
    def forward(self, x, train=True):
        
        def jac(x):
            v = torch.tensor([0.5]).to(x.device)
            J = torch.stack([torch.diag(vec) for vec in torch.heaviside(x, v)])
            return J.to(x.device)
        
        if train:

            z = torch.randn(x.shape).to(x.device)

            # inject noise
            x = x + 0.1 * z

            x = self.activation(self.fc1(x))     
            x = self.activation(self.fc2(x))
            x = self.activation(self.fc3(x))
            x = self.activation(self.fc4(x))
            return self.fc5(x)

        else:
            sigma = 0.1 * torch.eye(self.CH).tile([x.shape[0], 1, 1]).to(x.device)
#             print(sigma.shape, x.shape, self.fc1.weight.shape)
#             x = self.fc1(x); sigma = torch.matmul(self.fc1.weight.T, torch.matmul(sigma, self.fc1.weight))
#             x = self.activation(x); sigma = torch.matmul(jac(x).T, torch.matmul(sigma, jac(x)))
            x = self.activation(self.fc1(x))
            x = self.fc2(x); sigma = torch.matmul(self.fc2.weight.T, torch.matmul(sigma, self.fc2.weight))
            print(jac(x).T.shape, sigma.shape, jac(x).shape)
            x = self.activation(x); sigma = torch.matmul(jac(x).T, torch.matmul(sigma, jac(x)))
            x = self.fc3(x); sigma = torch.matmul(self.fc3.weight.T, torch.matmul(sigma, self.fc3.weight))
            x = self.activation(x); sigma = torch.matmul(jac(x).T, torch.matmul(sigma, jac(x)))
            x = self.fc4(x); sigma = torch.matmul(self.fc4.weight.T, torch.matmul(sigma, self.fc4.weight))
            x = self.activation(x); sigma = torch.matmul(jac(x).T, torch.matmul(sigma, jac(x)))
            x = self.fc5(x); sigma = torch.matmul(self.fc5.weight.T, torch.matmul(sigma, self.fc5.weight))
            return x, sigma



class VarpropNet(torch.nn.Module):

  def __init__(self, classes_num=10, p=0.0, init_ch=3):

    super().__init__()
    
    self.activation = torch.nn.LeakyReLU()

    resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    init_conv = torch.nn.Conv2d(init_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    modules = [init_conv] + list(resnet.children())[1:-1]

    self.extractor = torch.nn.Sequential(*modules)
    
    CH = 128
#     self.fc1 = torch.nn.Linear(512, CH)
#     self.fc2 = torch.nn.Linear(CH, CH)
#     self.fc3 = torch.nn.Linear(CH, CH)
#     self.fc4 = torch.nn.Linear(CH, CH)
#     self.fc5 = torch.nn.Linear(CH, classes_num)
    self.varprop = VarPropModule(CH=CH, out_channel=classes_num)

    self.dp = torch.nn.functional.dropout
    self.p = p

    self.classes_num = classes_num

  def forward(self, x, train=True):
    feat = self.activation(self.extractor(x)).squeeze(2).squeeze(2) #view(-1, 504) 
    return self.varprop(feat, train=train)
#     x = self.activation(self.fc2(x)) + x
#     x = self.activation(self.fc3(x)) + x
#     x = self.activation(self.fc4(x)) + x
#     return self.fc5(x, train=train)



##### SNGP ######

import torch.nn as nn
import math
from torch.nn.utils.parametrizations import spectral_norm

def RandomFeatureLinear(i_dim, o_dim, bias=True, require_grad=False):
    m = nn.Linear(i_dim, o_dim, bias)
    # https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/models/bert_sngp.py
    nn.init.normal_(m.weight, mean=0.0, std=0.05)
    # freeze weight
    m.weight.requires_grad = require_grad
    if bias:
        nn.init.uniform_(m.bias, a=0.0, b=2. * math.pi)
        # freeze bias
        m.bias.requires_grad = require_grad
    return m

class SNGP(nn.Module):
    def __init__(self,
                 hidden_size=768,
                 gp_kernel_scale=1.0,
                 num_inducing=1024,
                 gp_output_bias=0.,
                 layer_norm_eps=1e-12,
                 n_power_iterations=1,
                 spec_norm_bound=0.95,
                 scale_random_features=False,
                 normalize_input=False,    
                 gp_cov_momentum=0.999,
                 gp_cov_ridge_penalty=1e-3,
                 epochs=40,
                 num_classes=3,
                 device='cuda'):
        super(SNGP, self).__init__()
#         self.backbone = backbone
        self.final_epochs = epochs - 1
        self.gp_cov_ridge_penalty = gp_cov_ridge_penalty
        self.gp_cov_momentum = gp_cov_momentum

        self.pooled_output_dim = hidden_size
#         self.last_pooled_layer = spectral_norm(BertLinear(hidden_size, self.pooled_output_dim),
#                                                n_power_iterations=n_power_iterations, norm_bound=spec_norm_bound)

        self.gp_input_scale = 1. / math.sqrt(gp_kernel_scale)
        self.gp_feature_scale = math.sqrt(2. / float(num_inducing))
        self.gp_output_bias = gp_output_bias
        self.scale_random_features = scale_random_features
        self.normalize_input = normalize_input

        self._gp_input_normalize_layer = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps).to(device)
        self._gp_output_layer = nn.Linear(num_inducing, num_classes, bias=False).to(device)
        # bert gp_output_bias_trainable is false
        # https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L69
        self._gp_output_bias = torch.tensor([self.gp_output_bias] * num_classes).to(device)
        self._random_feature = RandomFeatureLinear(self.pooled_output_dim, num_inducing).to(device)

        # Laplace Random Feature Covariance
        # Posterior precision matrix for the GP's random feature coefficients.
        self.initial_precision_matrix = (self.gp_cov_ridge_penalty * torch.eye(num_inducing).to(device))
        self.precision_matrix = torch.nn.Parameter(copy.deepcopy(self.initial_precision_matrix), requires_grad=False)

    def extract_bert_features(self, latent_feature):
        # https://github.com/google/uncertainty-baselines/blob/b3686f75a10b1990c09b8eb589657090b8837d2c/uncertainty_baselines/models/bert_sngp.py#L336
        # Extract BERT encoder output (i.e., the CLS token).
        first_token_tensors = latent_feature[:, 0, :]
        cls_output = self.last_pooled_layer(first_token_tensors)
        return cls_output

    def gp_layer(self, gp_inputs, update_cov=True):
        # Supports lengthscale for custom random feature layer by directly
        # rescaling the input.
        if self.normalize_input:
            gp_inputs = self._gp_input_normalize_layer(gp_inputs)

        gp_feature = self._random_feature(gp_inputs)
        # cosine
        gp_feature = torch.cos(gp_feature)

        if self.scale_random_features:
            gp_feature = gp_feature * self.gp_input_scale

        # Computes posterior center (i.e., MAP estimate) and variance.
        gp_output = self._gp_output_layer(gp_feature) + self._gp_output_bias

        if update_cov:
            # update precision matrix
            self.update_cov(gp_feature)
        return gp_feature, gp_output

    def reset_cov(self):
        self.precision_matrix = torch.nn.Parameter(copy.deepcopy(self.initial_precision_matrix), requires_grad=False)

    def update_cov(self, gp_feature):
        # https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L346
        batch_size = gp_feature.size()[0]
        precision_matrix_minibatch = torch.matmul(gp_feature.t(), gp_feature)
        # Updates the population-wise precision matrix.
        if self.gp_cov_momentum > 0:
            # Use moving-average updates to accumulate batch-specific precision
            # matrices.
            precision_matrix_minibatch = precision_matrix_minibatch / batch_size
            precision_matrix_new = (
                    self.gp_cov_momentum * self.precision_matrix +
                    (1. - self.gp_cov_momentum) * precision_matrix_minibatch)
        else:
            # Compute exact population-wise covariance without momentum.
            # If use this option, make sure to pass through data only once.
            precision_matrix_new = self.precision_matrix + precision_matrix_minibatch
        #self.precision_matrix.weight = precision_matrix_new
        self.precision_matrix = torch.nn.Parameter(precision_matrix_new, requires_grad=False)

    def compute_predictive_covariance(self, gp_feature):
        # https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L403
        # Computes the covariance matrix of the feature coefficient.
        feature_cov_matrix = torch.linalg.inv(self.precision_matrix)

        # Computes the covariance matrix of the gp prediction.
        cov_feature_product = torch.matmul(feature_cov_matrix, gp_feature.t()) * self.gp_cov_ridge_penalty
        gp_cov_matrix = torch.matmul(gp_feature, cov_feature_product)
        return gp_cov_matrix

    def forward(self, x, return_gp_cov: bool = False, update_cov: bool = True):
        gp_feature, gp_output = self.gp_layer(x, update_cov=update_cov)
        if return_gp_cov:
            gp_cov_matrix = self.compute_predictive_covariance(gp_feature)
            return gp_output, gp_cov_matrix
        return gp_output
    
class SNGPNet(torch.nn.Module):

  def __init__(self, classes_num=10, p=0.0, init_ch=3):

    super().__init__()
    
    self.activation = torch.nn.LeakyReLU()

    resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    init_conv = torch.nn.Conv2d(init_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    modules = [init_conv] + list(resnet.children())[1:-1]

    self.extractor = torch.nn.Sequential(*modules)
    
    CH = 128
    self.fc1 = spectral_norm(torch.nn.Linear(512, CH))
    self.fc2 = spectral_norm(torch.nn.Linear(CH, CH))
    self.fc3 = spectral_norm(torch.nn.Linear(CH, CH))
    self.fc4 = spectral_norm(torch.nn.Linear(CH, CH))
    self.fc5 = SNGP(hidden_size=CH, num_classes=classes_num, num_inducing=CH, device="cuda")

    self.dp = torch.nn.functional.dropout
    self.p = p

    self.classes_num = classes_num

  def forward(self, x, return_gp_cov=False):
    feat = self.activation(self.extractor(x)).squeeze(2).squeeze(2) #view(-1, 504) 
    x = self.activation(self.fc1(feat))
    x = self.activation(self.fc2(x)) + x
    x = self.activation(self.fc3(x)) + x
    x = self.activation(self.fc4(x)) + x
    return self.fc5(x, return_gp_cov=return_gp_cov)



class BatchEnsemble1D(torch.nn.Module):

    def __init__(self, channels: int, n: int):

        super().__init__()

        self.channels = channels
        self.n = n

        weights = torch.ones([n, channels])
        self.masks = torch.nn.Parameter(weights, requires_grad=True)

    def forward(self, inputs):
        batch = inputs.shape[0]
        x = torch.split(inputs.unsqueeze(1), batch // self.n, dim=0)
        x = torch.cat(x, dim=1).permute([1, 0, 2])
        x = x * self.masks.unsqueeze(1).to(x.device)
        x = torch.cat(torch.split(x, 1, dim=0), dim=1)
        return x.squeeze(0)

class BatchNet(torch.nn.Module):

  def __init__(self, classes_num=10, p=0.0, init_ch=3):

    super().__init__()
    
    self.activation = torch.nn.LeakyReLU()

    resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    init_conv = torch.nn.Conv2d(init_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    modules = [init_conv] + list(resnet.children())[1:-1]

    self.extractor = torch.nn.Sequential(*modules)
    
    CH = 128
    self.fc1 = torch.nn.Linear(512, CH)
    self.fc2 = torch.nn.Linear(CH, CH)
    self.fc3 = torch.nn.Linear(CH, CH)
    self.fc4 = torch.nn.Linear(CH, CH)
    self.fc5 = torch.nn.Linear(CH, classes_num)
    
    self.masks1 = BatchEnsemble1D(CH, 5)
    self.masks2 = BatchEnsemble1D(CH, 5)
    self.masks3 = BatchEnsemble1D(CH, 5)
    self.masks4 = BatchEnsemble1D(CH, 5)

    self.dp = torch.nn.functional.dropout
    self.p = p

    self.classes_num = classes_num

  def forward(self, x):
    x = self.activation(self.extractor(x)).squeeze(2).squeeze(2) #view(-1, 504) 
    x = self.masks1(self.activation(self.fc1(x)))
    x = self.masks2(self.activation(self.fc2(x)) + x)
    x = self.masks3(self.activation(self.fc3(x)) + x)
    x = self.masks4(self.activation(self.fc4(x)) + x)
    return self.fc5(x)


# class MaskNet(torch.nn.Module):

#   def __init__(self, classes_num=10, p=0.0, init_ch=3):

#     super().__init__()
    
#     self.activation = torch.nn.LeakyReLU()

#     resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#     init_conv = torch.nn.Conv2d(init_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#     modules = [init_conv] + list(resnet.children())[1:-1]

#     self.extractor = torch.nn.Sequential(*modules)
    
#     CH = 128
#     self.fc1 = torch.nn.Linear(512, CH)
#     self.fc2 = torch.nn.Linear(CH, CH)
#     self.fc3 = torch.nn.Linear(CH, CH)
#     self.fc4 = torch.nn.Linear(CH, CH)
#     self.fc5 = torch.nn.Linear(CH, classes_num)
    
#     self.masks1 = Masksembles1D(CH, 5, 1.5)
#     self.masks2 = Masksembles1D(CH, 5, 1.5)
#     self.masks3 = Masksembles1D(CH, 5, 1.5)
#     self.masks4 = Masksembles1D(CH, 5, 1.5)

#     self.dp = torch.nn.functional.dropout
#     self.p = p

#     self.classes_num = classes_num

#   def forward(self, x):
#     x = self.activation(self.extractor(x)).squeeze(2).squeeze(2) #view(-1, 504) 
#     x = self.masks1(self.activation(self.fc1(x))).float()
#     x = self.masks2(self.activation(self.fc2(x)) + x).float()
#     x = self.masks3(self.activation(self.fc3(x)) + x).float()
#     x = self.masks4(self.activation(self.fc4(x)) + x).float()
#     return self.fc5(x)


class NoiseNet(torch.nn.Module):

  def __init__(self, classes_num=10, p=0.0, init_ch=4):

    super().__init__()
    
    self.activation = torch.nn.LeakyReLU()

    resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    init_conv = torch.nn.Conv2d(init_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    modules = [init_conv] + list(resnet.children())[1:-1]
  
    self.extractor = torch.nn.Sequential(*modules)
    
    CH = 128
    self.fc1 = torch.nn.Linear(512, CH)
    self.fc2 = torch.nn.Linear(CH, CH)
    self.fc3 = torch.nn.Linear(CH, CH)
    self.fc4 = torch.nn.Linear(CH, CH)
    self.fc5 = torch.nn.Linear(CH, classes_num)

    self.dp = torch.nn.functional.dropout
    self.p = p

    self.classes_num = classes_num

  def forward(self, inputs):
    x = inputs
    x = self.activation(self.extractor(x)).squeeze(2).squeeze(2) #view(-1, 504) 
    x = self.activation(self.fc1(x))
    x = self.dp(self.activation(self.fc2(x)) + x, p=self.p, training=True)
    x = self.dp(self.activation(self.fc3(x)) + x, p=self.p, training=True)
    x = self.dp(self.activation(self.fc4(x)) + x, p=self.p, training=True)
    return self.fc5(x)

class MCNet(torch.nn.Module):

  def __init__(self, classes_num=10):

    super().__init__()

    resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    modules = list(resnet.children())[:-1]
    self.extractor = torch.nn.Sequential(*modules)

    self.fc1 = torch.nn.Linear(512, 128)
    self.fc2 = torch.nn.Linear(128, 64)
    self.fc3 = torch.nn.Linear(64, 32)
    self.fc4 = torch.nn.Linear(32, classes_num)

    self.activation = torch.nn.LeakyReLU()
    
    self.dropout = torch.nn.Dropout(0.1)

    self.classes_num = classes_num

  def forward(self, x):
    x = self.extractor(x).squeeze(2).squeeze(2)
    x = self.activation(self.fc1(x))
    x = torch.nn.functional.dropout(self.activation(self.fc2(x)), p=0.1, training=True)
    x = torch.nn.functional.dropout(self.activation(self.fc3(x)), p=0.1, training=True)
    return self.fc4(x)