import torch
import torchvision
import torch.nn as nn

class MobileNet_v2(nn.Module):
    def __init__(self, num_classes, in_channel):
        super().__init__()
        self.model = torchvision.models.mobilenet_v2(weights=None, num_classes=num_classes)
        self.model.features[0][0] = nn.Conv2d(in_channel, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x):
        #print(x.shape)
        x = self.model(x)
        return x

class DenseNet(nn.Module):
    def __init__(self, num_classes, in_channel):
        super().__init__()
        self.model = torchvision.models.densenet161(weights=None, num_classes=num_classes)
        self.model.features[0] = nn.Conv2d(in_channel, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        x = self.model(x)
        return x
    
class EfficientNet_v2(nn.Module):
    def __init__(self, num_classes, in_channel):
        super().__init__()
        self.model = torchvision.models.efficientnet_v2_m(weights=None, num_classes=num_classes)
        self.model.features[0][0] = nn.Conv2d(in_channel, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x):
        x = self.model(x)
        return x
    
class ShuffleNet_v2(nn.Module):
    def __init__(self, num_classes, in_channel):
        super().__init__()
        self.model = torchvision.models.shufflenet_v2_x1_0(weights=None, num_classes=num_classes)
        self.model.conv1[0] = nn.Conv2d(in_channel, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x):
        x = self.model(x)
        return x
    
class Permute(nn.Module):
    def __init__(self, dim) -> None:
        self.dim = dim
        super().__init__()

    def forward(self, x):
        #print(x.shape)
        x = torch.flatten(x,start_dim=3,end_dim=-1)
        x = x.reshape([-1,self.dim,112,112])
        x = x.permute(0,3,2,1)
        #print(x.shape)
        return x
    
class Swin_transfomer(nn.Module):
    def __init__(self, num_classes, in_channel):
        super().__init__()
        self.model = torchvision.models.swin_b()
        self.model.features[0][0] = nn.Conv3d(in_channel,128,kernel_size=(2,4,4),stride=(2,4,4))
        #self.model.features[0][0] = nn.Conv3d(in_channel,128,kernel_size=(2,4,4),stride=(2,4,4))
        self.model.features[0][1] = Permute(num_classes)
        self.model.head = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        #print(x.shape)
        x = self.model(x)
        return x
