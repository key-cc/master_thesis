import torch
import torch.nn as nn
import torch.utils.data as dataf
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.nn import init
import h5py
import glob
import socket
from datetime import datetime
from torchsummary import summary
from tensorboardX import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from log import *
import torchvision.transforms as transforms
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.autograd import Variable
from torchvision.models import resnet18,resnet152,resnet101,resnet50




device_ids = [0,1,2,3]
logger = get_logger('ConvLSTM_hockeyFight.log')

save_dir_root = os.path.join(os.path.dirname(os.path.abspath('__file__')))
runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        resnet = resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.final = nn.Sequential(
            nn.Linear(resnet.fc.in_features, latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=0.01),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.final(x)


class LSTM(nn.Module):
    def __init__(self, latent_dim, num_layers, hidden_dim, bidirectional):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            latent_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.hidden_state = None

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self, x):
        x, self.hidden_state = self.lstm(x, self.hidden_state)
        return x


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
                    torch.zeros(
                        latent_repr.size(0), 1, self.hidden_attention.in_features
                    ),
                    requires_grad=False,
                ).float()
            ]
        h_t = hidden_repr[0]
        latent_att = self.latent_attention(latent_att)
        hidden_att = self.hidden_attention(h_t)
        joint_att = self.joint_attention(F.relu(latent_att + hidden_att)).squeeze(-1)
        attention_w = F.softmax(joint_att, dim=-1)
        return attention_w


class ConvLSTM(nn.Module):
    def __init__(
        self,
        num_classes,
        latent_dim=512,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
        attention=True,
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
        self.attention_layer = nn.Linear(
            2 * hidden_dim if bidirectional else hidden_dim, 1
        )

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

# ---------------------------------------------------------------------------------------------

# train
def train():
    
    model.train()
    
    logger.info('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    train_loss_data = 0.
    train_acc = 0.

    starttime = datetime.now()
    logger.info('start training!')

    
    for batch_idx, train_data in enumerate(train_loader):
        
        # data loading
        train_video, train_label = train_data
        train_video, train_label = train_video.cuda(device_ids[0]), train_label.cuda(device_ids[0])

        # Reset LSTM hidden state
        model.lstm.reset_hidden_state()

        train_out = model(train_video)

        # calculate the loss
        train_loss = criterion(train_out, train_label)
        # train_loss = L2loss(train_out,train_label,batch_size)
        train_loss_data += train_loss.item()

        # calculate the accuracy
        train_pred = torch.max(train_out, 1)[1]
        train_correct = (train_pred == train_label).sum()
        train_acc += train_correct.item()

        # update the grad
        optimizer.zero_grad()
        train_loss.backward()
        #optimizer.module.step()
        optimizer.step()

        # Print log
        logger.info('Train Epoch: {}  [{}/{} ({:.0f}%)]     Batch_Loss: {:.6f}       Batch_Acc: {:.3f}%'.format(
            epoch + 1, 
            batch_idx * len(train_video), 
            len(train_dataset),
            100. * batch_idx / len(train_loader),
            train_loss.item() / batch_size,
            100. * train_correct.item() / batch_size
            )
        )

        logger.info('-----------------------------------------------------------------------------------------------------------')
    endtime = datetime.now()
    time = (endtime-starttime).seconds
    logger.info('###############################################################################################################\n')
    logger.info(('Train Epoch: [{}\{}]\ttime: {}s').format(epoch+1,num_epoches,time))
    
    #for param_lr in optimizer.module.param_groups: 
    for param_lr in optimizer.param_groups:
        logger.info('lr_rate: ' + str(param_lr['lr']) + '\n')

    logger.info('Train_Loss: {:.6f}      Train_Acc: {:.3f}%\n'.format(train_loss_data / (len(train_dataset)),
                                                            100. * train_acc / (len(train_dataset))
                                                            )
          )
    logger.info('-----------------------------------------------------------------------------------------------------------')
    writer.add_scalar('data/train_loss_epoch', train_loss_data / (len(train_dataset)), epoch + 1)
    writer.add_scalar('data/train_acc_epoch', 100. * train_acc / (len(train_dataset)), epoch + 1)

# test
def test():
    
    model.eval()

    test_loss_data = 0.
    test_acc = 0.

    for test_data in test_loader:
        
        # data loading
        test_video, test_label = test_data
        test_video, test_label = test_video.cuda(device_ids[0]), test_label.cuda(device_ids[0])
        with torch.no_grad():
                # Reset LSTM hidden state
                model.lstm.reset_hidden_state()
                # Get sequence predictions
                test_out = model(test_video)

        #calculate the loss
        test_loss = criterion(test_out, test_label)
        test_loss_data += test_loss.item()

        # calculate the accuracy
        test_pred = torch.max(test_out, 1)[1]
        test_correct = (test_pred == test_label).sum()
        test_acc += test_correct.item()

    # Log test performance
    logger.info('Test_Loss: {:.6f}      Test_Acc: {:.3f}%\n'.format(test_loss_data / (len(test_dataset)),
                                                            100. * test_acc / (len(test_dataset))
                                                            )
          )
    logger.info('--------------------------------------------------------')
    writer.add_scalar('data/val_loss_epoch', test_loss_data / (len(test_dataset)), epoch+1)
    writer.add_scalar('data/val_acc_epoch', 100. * test_acc / (len(test_dataset)), epoch+1)

# ---------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------
# training dataset
f = h5py.File('hockey_train_lstm.h5','r')
train_video = f['data'][()]
# train_video, train_label = create_train(1000,60,90)

#train_video = train_video.transpose((0,2,1,3,4))     
train_label = f['label'][()]

train_video = torch.from_numpy(train_video)
train_label = torch.from_numpy(train_label)

# test dataset
f1 = h5py.File('hockey_test_lstm.h5','r')                           
test_video = f1['data'][()]    
#test_video = test_video.transpose((0,2,1,3,4))     
test_label = f1['label'][()] 

test_video = torch.from_numpy(test_video)
test_label = torch.from_numpy(test_label)


# ---------------------------------------------------------------------------------------------

# parameters 
batch_size = 50
learning_rate = 1e-4
num_epoches = 200

# ---------------------------------------------------------------------------------------------

# training dataset 
train_dataset = dataf.TensorDataset(train_video, train_label)
train_loader = dataf.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# test dataset
test_dataset = dataf.TensorDataset(test_video, test_label)
test_loader = dataf.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# initialize the model
model = ConvLSTM(num_classes=2)
model = model.cuda(device_ids[0])
#model = nn.DataParallel(model, device_ids=device_ids)
#summary(model,(3,16,112,112))

# loss and optimization 
criterion = nn.CrossEntropyLoss()

# def L2loss(x,y,batchsize):
#     loss_batch = 0
#     for i in range(0, batchsize):
#         temp_label = y[i].item()
#         temp_data = x[i].item()
#         loss_batch += (temp_data[temp_label] - temp_label) ** 2
#     loss_batch = torch.div(loss_batch, batchsize)
#     return loss_batch

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.001)
#optimizer = nn.DataParallel(optimizer, device_ids=device_ids)

# ----------------------------------------------------------------------------------------------
log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
writer = SummaryWriter(log_dir=log_dir)

best_acc = 0
for epoch in range(num_epoches):
    train()
    test_acc = test()
    # Save model checkpoint
    if epoch % 5 == 0:
        os.makedirs("model_checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"model_checkpoints/{model.__class__.__name__}_{epoch}.pth")

writer.close()

