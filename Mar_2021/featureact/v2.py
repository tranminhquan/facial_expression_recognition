
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

import math

from pytorch_metric_learning import losses, miners, distances, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

class FeatureAct(nn.Module):
    def __init__(self, src_depth, target_depth, reduce_factor=1, return_cor_feature=False):
        super(FeatureAct, self).__init__()
        
        self.return_cor_feature = return_cor_feature
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        
        self.src_conv = nn.Conv2d(src_depth, src_depth // reduce_factor, kernel_size=1, padding=False)
        self.target_conv = nn.Conv2d(target_depth, target_depth // reduce_factor, kernel_size=1, padding=False)
        
    def forward(self, src_x, target_x):
        
        src_features = self.src_conv(src_x)
        src_features = self.pool(src_features)
        
        target_features = self.target_conv(target_x)
        
        
        cor = nn.Flatten(-2)(src_features[:, None, :, :]).permute(0,2,1,3) * nn.Flatten(-2)(target_features[:, None, :, :])
        cor = cor / torch.norm(cor)
#         print(cor.size())
        
        cor_feature = torch.cat([(src_features * cor[:, :, k].view(-1, src_features.size(-3), src_features.size(-2), src_features.size(-1))).sum(1, keepdim=True) for k in range(cor.size(2))], dim=1)
        
        target_x = target_x * (1 + cor_feature)
        
        if self.return_cor_feature:
            return target_x, cor_feature
        
        return target_x
    

class BaselineFeatureAct(nn.Module):
    def __init__(self, return_embeddings=False):
        
        super(BaselineFeatureAct, self).__init__()
        
        self.return_embeddings = return_embeddings
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        
        self.conv0 = nn.Conv2d(1, 32, kernel_size=(3,3), padding=1, stride=1)
        self.conv0b = nn.Conv2d(32, 32, kernel_size=(3,3), padding=1, stride=1)
        self.bn0 = nn.BatchNorm2d(32)
        self.bn0b = nn.BatchNorm2d(32)
        
        self.conv1 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=1, stride=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=(3,3), padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn1b = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3,3), padding=1, stride=1)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=(3,3), padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn2b = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3,3), padding=1, stride=1)
        self.conv3b = nn.Conv2d(256, 256, kernel_size=(3,3), padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn3b = nn.BatchNorm2d(256)
  
        self.fact0_1 = FeatureAct(32, 64)
        self.fact1_2 = FeatureAct(64, 128)
        self.fact2_3 = FeatureAct(128, 256)
        
       
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(2304, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                        nn.Linear(256, 128), nn.BatchNorm1d(128))

    def forward(self, x):
        
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.conv0b(x)
        x = self.bn0b(x)
        x = self.relu(x)
        
        x0 = self.maxpool(x)
        
        x = self.conv1(x0)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1b(x)
        x = self.bn1b(x)
        x = self.relu(x)
        
        x1 = self.maxpool(x)

        xfact0_1 = self.fact0_1(x0, x1)
        
        x = self.conv2(xfact0_1)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2b(x)
        x = self.bn2b(x)
        x = self.relu(x)
        
        x2 = self.maxpool(x)

        xfact1_2 = self.fact1_2(x1, x2)
        
        x = self.conv3(xfact1_2)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3b(x)
        x = self.bn3b(x)
        x = self.relu(x)
        
        x3 = self.maxpool(x)

        xfact2_3 = self.fact2_3(x2, x3)
        
        x_emb = nn.Flatten()(xfact2_3)
        x = self.classifier(x_emb)
        
        if self.return_embeddings:
            return x, F.normalize(x_emb, p=2, dim=1)
        
        return x
    
    
class BaselineFeatureActExtend(nn.Module):
    '''
    consider cor_feature
    '''
    def __init__(self, return_embeddings=False):
        
        super(BaselineFeatureActExtend, self).__init__()
        
        self.return_embeddings = return_embeddings
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        
        self.conv0 = nn.Conv2d(1, 32, kernel_size=(3,3), padding=1, stride=1)
        self.conv0b = nn.Conv2d(32, 32, kernel_size=(3,3), padding=1, stride=1)
        self.bn0 = nn.BatchNorm2d(32)
        self.bn0b = nn.BatchNorm2d(32)
        
        self.conv1 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=1, stride=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=(3,3), padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn1b = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3,3), padding=1, stride=1)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=(3,3), padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn2b = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3,3), padding=1, stride=1)
        self.conv3b = nn.Conv2d(256, 256, kernel_size=(3,3), padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn3b = nn.BatchNorm2d(256)
  
        self.fact0_1 = FeatureAct(32, 64)
        self.fact1_2 = FeatureAct(64, 128)
        self.fact2_3 = FeatureAct(128, 256)
        
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(2304, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                        nn.Linear(256, 128), nn.BatchNorm1d(128))

        self.cf_classifier0 = nn.Sequential(nn.Flatten(), 
                                            nn.Linear(9216, 256), 
                                            nn.BatchNorm1d(256), 
                                            nn.ReLU(inplace=True),
                                            nn.Linear(256, 7), 
                                            nn.BatchNorm1d(7), 
                                            nn.Softmax())
        
        self.cf_classifier1 = nn.Sequential(nn.Flatten(), 
                                            nn.Linear(4608, 256), 
                                            nn.BatchNorm1d(256), 
                                            nn.ReLU(inplace=True),
                                            nn.Linear(256, 7), 
                                            nn.BatchNorm1d(7), 
                                            nn.Softmax())

        self.cf_classifier2 = nn.Sequential(nn.Flatten(), 
                                            nn.Linear(2304, 256), 
                                            nn.BatchNorm1d(256), 
                                            nn.ReLU(inplace=True),
                                            nn.Linear(256, 7), 
                                            nn.BatchNorm1d(7), 
                                            nn.Softmax())

        self.weighted_classifier = nn.Sequential(nn.Linear(7*4, 7), nn.BatchNorm1d(7))

    def forward(self, x):
        
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.conv0b(x)
        x = self.bn0b(x)
        x = self.relu(x)
        
        x0 = self.maxpool(x)
        
        x = self.conv1(x0)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1b(x)
        x = self.bn1b(x)
        x = self.relu(x)
        
        x1 = self.maxpool(x)

        xfact0_1, cfeature_0 = self.fact0_1(x0, x1)
        
        x = self.conv2(xfact0_1)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2b(x)
        x = self.bn2b(x)
        x = self.relu(x)
        
        x2 = self.maxpool(x)

        xfact1_2, cfeature_1 = self.fact1_2(x1, x2)
        
        x = self.conv3(xfact1_2)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3b(x)
        x = self.bn3b(x)
        x = self.relu(x)
        
        x3 = self.maxpool(x)

        xfact2_3, cfeature_2 = self.fact2_3(x2, x3)
        
        x = nn.Flatten()(xfact2_3)
        x = self.classifier(x)

        # print(cfeature_0.size(),cfeature_1.size(), cfeature_2.size() )

        cf_class0 = self.cf_classifier0(cfeature_0)
        cf_class1 = self.cf_classifier1(cfeature_1)
        cf_class2 = self.cf_classifier2(cfeature_2)

        x_emb = torch.cat([x, cf_class0, cf_class1, cf_class2], dim=1)
        x_out = self.weighted_classifier(x_emb)
        
        if self.return_embeddings:
            return x_out, F.normalize(x_emb, p=2, dim=1)
        
        return x_out
    
 