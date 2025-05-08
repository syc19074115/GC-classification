import torch
from torch import nn
import config
from thop import profile

def generate_model(args, device):
    
    print('Create {} model'.format(args.model_type))
    #assert args.model_depth in [18, 50, 101]
    if args.model_type == 'resnet18':
        import model.resnet as resnet
        model = resnet.resnet18(
            n_input_channels = args.in_channel,
            num_classes=args.feature_dim,
            shortcut_type=args.shortcut_type,
        )

    elif args.model_type == 'resnet50':
        import model.resnet as resnet
        model = resnet.resnet50(
            num_classes=args.feature_dim,
            shortcut_type=args.shortcut_type,
            n_input_channels = args.in_channel
        )

    elif args.model_type == 'resnet101':
        import model.resnet as resnet
        model = resnet.resnet101(
            num_classes=args.feature_dim,
            shortcut_type=args.shortcut_type,
            n_input_channels = args.in_channel
        )

    elif args.model_type == 'resnext101':
        import model.resnext as resnext
        model = resnext.resnext101(
            num_classes=args.feature_dim,
            shortcut_type=args.shortcut_type,
            cardinality=32,
            sample_size=args.sample_size,
            sample_duration=args.sample_duration,
            channel = args.in_channel
        )

    elif args.model_type == 'ViT':
        import model.ViT as ViT
        model = ViT.VisionTransformer(
            num_classes=args.feature_dim,
            depth=args.vit_depth,
            num_heads=args.vit_heads,
            mlp_ratio=args.vit_ratio,
            in_chans=args.in_channel
        )

    elif args.model_type == 'T2T-ViT':
        import model.T2TViT as T2TViT
        model = T2TViT.T2T_ViT(
            num_classes=args.feature_dim,
            depth=args.vit_depth,
            num_heads=args.vit_heads,
            mlp_ratio=args.vit_ratio,
            in_chans=args.in_channel
        )

    elif args.model_type == 'MobileNet':
        import model.tvmodel as tv
        model = tv.MobileNet_v2(
            num_classes=args.feature_dim,
            in_channel=args.in_channel
        )

    elif args.model_type == 'DenseNet':
        import model.tvmodel as tv
        model = tv.DenseNet(
            num_classes=args.feature_dim,
            in_channel=args.in_channel
        )
    elif args.model_type == 'EfficientNet':
        import model.tvmodel as tv
        model = tv.EfficientNet_v2(
            num_classes=args.feature_dim,
            in_channel=args.in_channel
        )
        
    elif args.model_type == 'ShuffleNet':
        import model.tvmodel as tv
        model = tv.ShuffleNet_v2(
            num_classes=args.feature_dim,
            in_channel=args.in_channel
        )
    
    elif args.model_type == '3d_ShuffleNet':
        import model.ShuffleNetv2 as sf2
        model = sf2.ShuffleNetV2(
            num_classes=args.feature_dim,
            in_channel=args.in_channel
        )
    
    elif args.model_type == '3d_mobilenetv2':
        import model.mobilenetv2 as mb2
        model = mb2.MobileNetV2(
            num_classes=args.feature_dim,
            in_channel=args.in_channel #这句为了后面注释掉  为了回归任务
            # in_channel=1
        )
    
    elif args.model_type == '3d_mobilenet':
        import model.mobilenet as mb
        model = mb.MobileNet(
            num_classes=args.feature_dim,
            in_channel=args.in_channel
        )
    elif args.model_type == '3d_ViT':
        import model.ViT_3D as VIT3D
        model = VIT3D.VisionTransformer(
            num_classes=args.feature_dim,
            depth=args.vit_depth,
            num_heads=args.vit_heads,
            mlp_ratio=args.vit_ratio,
            in_chans=args.in_channel
        )
    elif args.model_type == 'Swin':
        import model.tvmodel
        model = model.tvmodel.Swin_transfomer(
            num_classes=args.feature_dim,
            in_channel=args.in_channel
        )
    

    return model

class ClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes,):
        super(ClassifierHead, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)
    def forward(self, x):
        x = self.fc(x)
        return x

class Model(nn.Module):
    def __init__(self, args, device):
        super(Model, self).__init__()
        self.feature_model = generate_model(args, device)
        self.num_classes = args.num_classes
        self.device = device
        self.fc = nn.Linear(args.feature_dim+1, self.num_classes) #加入年龄
        #self.fc = nn.Linear(args.feature_dim, self.num_classes) 
        #self.softmax = nn.Softmax(dim=1)
        #self.fc = ClassifierHead(in_features, num_classes)
        self._initialize_weights()
        

    def _initialize_weights(self):
        print("initialize weights for network!")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        age = x[1].reshape(x[0].shape[0],1)
        x = x[0]
        x = self.feature_model(x)
        _, infeature = x.shape
        x = torch.cat((x,age),dim=1) 
        x = self.fc(x)
        return x


if __name__ == '__main__':
    args = config.parse_args()
    args.in_channels = 2
    device = 'cpu'
    net = Model(args=args, device=device)
    net = net.to(device)
    input = torch.randn([1,2,16,224,224])
    #input = torch.randn([1,3,224,224])
    age = 0.1
    train_predict = net([input,age])
    print(train_predict)
    


