import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.init as init
from torchvision.models import resnet152

##############################
#         Encoder
##############################


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        resnet = resnet152(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.final = nn.Sequential(
            nn.Linear(resnet.fc.in_features, latent_dim), nn.BatchNorm1d(latent_dim, momentum=0.01)
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.final(x)


##############################
#           LSTM
##############################


class LSTM(nn.Module):
    def __init__(self, latent_dim, num_layers, hidden_dim, bidirectional):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.hidden_state = None

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self, x):
        x, self.hidden_state = self.lstm(x, self.hidden_state)
        return x


##############################
#      Attention Module
##############################


class Attention(nn.Module):
    def __init__(self, latent_dim, hidden_dim, attention_dim):
        super(Attention, self).__init__()
        self.latent_attention = nn.Linear(latent_dim, attention_dim)
        self.hidden_attention = nn.Linear(hidden_dim, attention_dim)
        self.joint_attention = nn.Linear(attention_dim, 1)

    def forward(self, latent_repr, hidden_repr):
        if hidden_repr is None:
            hidden_repr = [
                Variable(
                    torch.zeros(latent_repr.size(0), 1, self.hidden_attention.in_features), requires_grad=False
                ).float()
            ]
        h_t = hidden_repr[0]
        latent_att = self.latent_attention(latent_att)
        hidden_att = self.hidden_attention(h_t)
        joint_att = self.joint_attention(F.relu(latent_att + hidden_att)).squeeze(-1)
        attention_w = F.softmax(joint_att, dim=-1)
        return attention_w


##############################
#         ConvLSTM
##############################


class ConvLSTM(nn.Module):
    def __init__(
        self, num_classes, latent_dim=512, lstm_layers=1, hidden_dim=1024, bidirectional=True, attention=True
    ):
        super(ConvLSTM, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.lstm = LSTM(latent_dim, lstm_layers, hidden_dim, bidirectional)
        self.output_layers = nn.Sequential(
            nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=-1),
        )
        self.attention = attention
        self.attention_layer = nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, 1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.encoder(x)
        x = x.view(batch_size, seq_length, -1)
        x = self.lstm(x)
        if self.attention:
            attention_w = F.softmax(self.attention_layer(x).squeeze(-1), dim=-1)
            x = torch.sum(attention_w.unsqueeze(-1) * x, dim=1)
        else:
            x = x[:, -1]
        return self.output_layers(x)


##############################
#     Conv2D Classifier
#        (Baseline)
##############################


class ConvClassifier(nn.Module):
    def __init__(self, num_classes, latent_dim):
        super(ConvClassifier, self).__init__()
        resnet = resnet152(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.final = nn.Sequential(
            nn.Linear(resnet.fc.in_features, latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=0.01),
            nn.Linear(latent_dim, num_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.feature_extractor(x)
        x = x.view(batch_size * seq_length, -1)
        x = self.final(x)
        x = x.view(batch_size, seq_length, -1)
        return x


##############################
#         P3D
##############################
class P3D_Block(nn.Module):

    def __init__(self, blockType, inplanes, planes, stride=1):
        super(P3D_Block, self).__init__()
        self.expansion = 4
        self.blockType=blockType
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        if self.blockType=='A':
            self.conv2D = nn.Conv3d(planes, planes, kernel_size=(1,3,3), stride=(1,stride,stride),
                                   padding=(0,1,1), bias=False)
            self.conv1D = nn.Conv3d(planes, planes, kernel_size=(3,1,1), stride=(stride,1,1),
                                    padding=(1,0,0), bias=False)
        elif self.blockType == 'B':
            self.conv2D = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=stride,
                                    padding=(0, 1, 1), bias=False)
            self.conv1D = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1), stride=stride,
                                    padding=(1, 0, 0), bias=False)
        else:
            self.conv2D = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=stride,
                                    padding=(0, 1, 1), bias=False)
            self.conv1D = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1), stride=1,
                                    padding=(1, 0, 0), bias=False)
        self.bn2D = nn.BatchNorm3d(planes)
        self.bn1D = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU()
        self.stride = stride

        if self.stride != 1 or inplanes!= planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * self.expansion),
            )
        else:
            self.downsample=None


    def forward(self, x):
        x_branch = self.conv1(x)
        x_branch = self.bn1(x_branch)
        x_branch = self.relu(x_branch)

        if self.blockType=='A':
            x_branch = self.conv2D(x_branch)
            x_branch = self.bn2D(x_branch)
            x_branch = self.relu(x_branch)
            x_branch = self.conv1D(x_branch)
            x_branch = self.bn1D(x_branch)
            x_branch = self.relu(x_branch)
        elif self.blockType=='B':
            x_branch2D = self.conv2D(x_branch)
            x_branch2D = self.bn2D(x_branch2D)
            x_branch2D = self.relu(x_branch2D)
            x_branch1D = self.conv1D(x_branch)
            x_branch1D = self.bn1D(x_branch1D)
            x_branch=x_branch1D+x_branch2D
            x_branch=self.relu(x_branch)
        else:
            x_branch = self.conv2D(x_branch)
            x_branch = self.bn2D(x_branch)
            x_branch = self.relu(x_branch)
            x_branch1D = self.conv1D(x_branch)
            x_branch1D = self.bn1D(x_branch1D)
            x_branch=x_branch+x_branch1D
            x_branch=self.relu(x_branch)

        x_branch = self.conv3(x_branch)
        x_branch = self.bn3(x_branch)

        if self.downsample is not None:
            x = self.downsample(x)

        x =x+ x_branch
        x = self.relu(x)
        return x

class P3D (nn.Module):
    # input size: 16 x 160 x 160
    def __init__(self, num_class):
        super(P3D, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.conv2 = nn.Sequential(P3D_Block('A',64,64,2),
                                    P3D_Block('B', 64 * self.expansion, 64),
                                    P3D_Block('C', 64 * self.expansion, 64))
        self.conv3 = nn.Sequential(P3D_Block('A', 64 * self.expansion, 128, 2),
                                   P3D_Block('B', 128 * self.expansion, 128),
                                   P3D_Block('C', 128 * self.expansion, 128),
                                   P3D_Block('A', 128 * self.expansion, 128))
        self.conv4 = nn.Sequential(P3D_Block('B', 128 * self.expansion, 256, 2),
                                   P3D_Block('C', 256 * self.expansion, 256),
                                   P3D_Block('A', 256 * self.expansion, 256),
                                   P3D_Block('B', 256 * self.expansion, 256),
                                   P3D_Block('C', 256 * self.expansion, 256),
                                   P3D_Block('A', 256 * self.expansion, 256))
        self.conv5 = nn.Sequential(P3D_Block('B', 256 * self.expansion, 512, 2),
                                   P3D_Block('C', 512 * self.expansion, 512),
                                   P3D_Block('A', 512 * self.expansion, 512))
        self.average_pool=nn.AvgPool3d((1,3,3))
        self.fc=nn.Linear(512 * self.expansion,num_class)

    def forward(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.average_pool(x)
        x=x.view(x.size(0),-1)
        x = self.fc(x)
        return x


##############################
#         ARTNet
##############################
class SMART_block(nn.Module):

    def __init__(self, in_channel,out_channel,kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)):
        super(SMART_block, self).__init__()

        self.appearance_conv=nn.Conv3d(in_channel, out_channel, kernel_size=(1,kernel_size[1],kernel_size[2]),stride= stride,padding=(0, padding[1], padding[2]),bias=False)
        self.appearance_bn=nn.BatchNorm3d(out_channel)

        self.relation_conv=nn.Conv3d(in_channel, out_channel,kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.relation_bn1=nn.BatchNorm3d(out_channel)
        self.relation_pooling=nn.Conv3d(out_channel,out_channel//2,kernel_size=1,stride=1,groups=out_channel//2,bias=False)
        nn.init.constant_(self.relation_pooling.weight,0.5)
        self.relation_pooling.weight.requires_grad=False
        self.relation_bn2 = nn.BatchNorm3d(out_channel//2)

        self.reduce=nn.Conv3d(out_channel+out_channel//2,out_channel,kernel_size=1,bias=False)
        self.reduce_bn=nn.BatchNorm3d(out_channel)

        self.relu = nn.ReLU()
        if in_channel != out_channel or stride[0] != 1 or stride[1] != 1:
            self.down_sample = nn.Sequential(nn.Conv3d(in_channel, out_channel, kernel_size=1,
                                                       stride=stride,
                                                       bias=False),
                                             nn.BatchNorm3d(out_channel))
        else:
            self.down_sample = None

    def forward(self, x):
        appearance=x
        relation=x
        appearance=self.appearance_conv(appearance)
        appearance=self.appearance_bn(appearance)
        relation=self.relation_conv(relation)
        relation=self.relation_bn1(relation)
        relation=torch.pow(relation,2)
        relation=self.relation_pooling(relation)
        relation=self.relation_bn2(relation)
        stream=self.relu(torch.cat([appearance,relation],1))
        stream=self.reduce(stream)
        stream=self.reduce_bn(stream)
        if self.down_sample is not None:
            x=self.down_sample(x)

        return self.relu(stream+x)


class ARTNet(nn.Module):
    # Input size: 16x112x112
    def __init__(self, num_class):
        super(ARTNet, self).__init__()

        self.conv1=SMART_block(3,64,kernel_size=(3,7,7),stride=(2,2,2),padding=(1,3,3))
        self.conv2=nn.Sequential(SMART_block(64,64),
                                 SMART_block(64, 64))
        self.conv3=nn.Sequential(SMART_block(64,128,stride=(2,2,2)),
                                 SMART_block(128, 128))
        self.conv4 = nn.Sequential(SMART_block(128, 256, stride=(2,2,2)),
                                   SMART_block(256, 256))
        self.conv5 = nn.Sequential(SMART_block(256, 512, stride=(2,2,2)),
                                   SMART_block(512, 512))
        self.avg_pool=nn.AvgPool3d(kernel_size=(1,7,7))
        self.linear=nn.Linear(512,num_class)

    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.avg_pool(x)
        return self.linear(x.view(x.size(0),-1))


##############################
#         Res3D
##############################
class ResBlock(nn.Module):
    def __init__(self, in_channel,out_channel, spatial_stride=1,temporal_stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channel, out_channel,kernel_size=(3,3,3),stride=(temporal_stride,spatial_stride,spatial_stride),padding=(1,1,1))
        self.conv2 = nn.Conv3d(out_channel, out_channel,kernel_size=(3, 3, 3),stride=(1, 1, 1),padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.relu = nn.ReLU()
        if in_channel != out_channel or spatial_stride != 1 or temporal_stride != 1:
            self.down_sample=nn.Sequential(nn.Conv3d(in_channel, out_channel,kernel_size=1,stride=(temporal_stride,spatial_stride,spatial_stride),bias=False),
                                           nn.BatchNorm3d(out_channel))
        else:
            self.down_sample=None

    def forward(self, x):
        x_branch = self.conv1(x)
        x_branch = self.bn1(x_branch)
        x_branch = self.relu(x_branch)
        x_branch = self.conv2(x_branch)
        x_branch = self.bn2(x_branch)
        if self.down_sample is not None:
            x=self.down_sample(x)
        return self.relu(x_branch+x)

class Res3D(nn.Module):
    # Input size: 8x224x224
    def __init__(self, num_class):
        super(Res3D, self).__init__()

        self.conv1=nn.Conv3d(3,64,kernel_size=(3,7,7),stride=(1,2,2),padding=(1,3,3))
        self.conv2=nn.Sequential(ResBlock(64,64,spatial_stride=2),
                                 ResBlock(64, 64))
        self.conv3=nn.Sequential(ResBlock(64,128,spatial_stride=2,temporal_stride=2),
                                 ResBlock(128, 128))
        self.conv4 = nn.Sequential(ResBlock(128, 256, spatial_stride=2,temporal_stride=2),
                                   ResBlock(256, 256))
        self.conv5 = nn.Sequential(ResBlock(256, 512, spatial_stride=2,temporal_stride=2),
                                   ResBlock(512, 512))
        self.avg_pool=nn.AvgPool3d(kernel_size=(1,7,7))
        self.linear=nn.Linear(512,num_class)

    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.avg_pool(x)
        return self.linear(x.view(x.size(0),-1))


##############################
#         R21D
##############################
class Res21D_Block(nn.Module):
    def __init__(self, in_channel,out_channel, spatial_stride=1,temporal_stride=1):
        super(Res21D_Block, self).__init__()
        self.MidChannel1=int((27*in_channel*out_channel)/(9*in_channel+3*out_channel))
        self.MidChannel2 = int((27 * out_channel * out_channel) / ( 12 * out_channel))
        self.conv1_2D = nn.Conv3d(in_channel,self.MidChannel1 , kernel_size=(1, 3, 3), stride=(1, spatial_stride, spatial_stride),
                                padding=(0, 1, 1))
        self.bn1_2D = nn.BatchNorm3d(self.MidChannel1)
        self.conv1_1D=nn.Conv3d(self.MidChannel1, out_channel, kernel_size=(3, 1, 1), stride=(temporal_stride, 1, 1),
                                padding=(1, 0, 0))
        self.bn1_1D = nn.BatchNorm3d(out_channel)

        self.conv2_2D = nn.Conv3d(out_channel, self.MidChannel2, kernel_size=(1, 3, 3), stride=1,
                                  padding=(0, 1, 1))
        self.bn2_2D = nn.BatchNorm3d(self.MidChannel2)
        self.conv2_1D = nn.Conv3d(self.MidChannel2, out_channel, kernel_size=(3, 1, 1), stride=1,
                                  padding=(1, 0, 0))
        self.bn2_1D = nn.BatchNorm3d(out_channel)

        self.relu = nn.ReLU()
        if in_channel != out_channel or spatial_stride != 1 or temporal_stride != 1:
            self.down_sample=nn.Sequential(nn.Conv3d(in_channel, out_channel,kernel_size=1,stride=(temporal_stride, spatial_stride, spatial_stride),bias=False),
                                           nn.BatchNorm3d(out_channel))
        else:
            self.down_sample=None

    def forward(self, x):

        x_branch = self.conv1_2D(x)
        x_branch=self.bn1_2D(x_branch)
        x_branch = self.relu(x_branch)
        x_branch=self.conv1_1D(x_branch)
        x_branch=self.bn1_1D(x_branch)
        x_branch = self.relu(x_branch)

        x_branch = self.conv2_2D(x_branch)
        x_branch = self.bn2_2D(x_branch)
        x_branch = self.relu(x_branch)
        x_branch = self.conv2_1D(x_branch)
        x_branch = self.bn2_1D(x_branch)

        if self.down_sample is not None:
            x=self.down_sample(x)
        return self.relu(x_branch+x)

class Res21D(nn.Module):
    # Input size: 8 x 112 x 112
    def __init__(self, num_class):
        super(Res21D, self).__init__()

        self.conv1=nn.Conv3d(3,64,kernel_size=(3,7,7),stride=(1,2,2),padding=(1,3,3))
        self.conv2=nn.Sequential(Res21D_Block(64, 64, spatial_stride=2),
                                 Res21D_Block(64, 64),
                                 Res21D_Block(64, 64))
        self.conv3=nn.Sequential(Res21D_Block(64,128,spatial_stride=2,temporal_stride=2),
                                 Res21D_Block(128, 128),
                                 Res21D_Block(128, 128),
                                 Res21D_Block(128, 128),)
        self.conv4 = nn.Sequential(Res21D_Block(128, 256, spatial_stride=2,temporal_stride=2),
                                   Res21D_Block(256, 256),
                                   Res21D_Block(256, 256),
                                   Res21D_Block(256, 256),
                                   Res21D_Block(256, 256),
                                   Res21D_Block(256, 256))
        self.conv5 = nn.Sequential(Res21D_Block(256, 512, spatial_stride=2,temporal_stride=2),
                                   Res21D_Block(512, 512),
                                   Res21D_Block(512, 512))
        self.avg_pool=nn.AvgPool3d(kernel_size=(1,4,4))
        self.linear=nn.Linear(512,num_class)

    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.avg_pool(x)
        return self.linear(x.view(x.size(0),-1))




