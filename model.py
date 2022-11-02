import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from Cream.TinyViT.models.tiny_vit import *

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class Efficientb0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        self.backbone = models.efficientnet_b0(pretrained=True)
#         self.backbone = tiny_vit_21m_224(pretrained=True) # backbone 모델을 tiny_vit으로 교체
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.backbone(x)
        x = self.classifier(x)
        return x

class TinyVit_224(nn.Module):
    def __init__(self, num_classes):
        super().__init__()     
        self.backbone = tiny_vit_21m_224(pretrained=True)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    
class Efficientb3(nn.Module):
    def __init__(self, num_classes=50):
        super(Efficientb3, self).__init__()
        self.backbone = models.efficientnet.efficientnet_b3(pretrained=True)
        self.classifier = nn.Linear(1000, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    

class TinyVit_384(nn.Module):
    def __init__(self, num_classes):
        super().__init__()     
        self.backbone = tiny_vit_21m_384(pretrained=True)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x