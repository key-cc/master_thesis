import os.path

import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
from c3d_models import C3D,get_1x_lr_params,get_10x_lr_params,CNN3D,get_res3d_model
from c3d_dataset import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
import time
import datetime
import torch.nn as nn
from network.i3d import i3d_resnet18
from tensorboardX import SummaryWriter
import glob
#from datetime import datetime
import socket
from tqdm import tqdm
import timeit


def seed_everything(seed = 3447):
    '''
    set seed
    :param seed:
    :param device:
    :return:
    '''
    import os
    import random
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/hy-tmp/pytorch_C3DandBiLSTM/data/UCF-101-frames", help="Path to UCF-101 dataset")
    parser.add_argument("--split_path", type=str, default="/hy-tmp/pytorch_C3DandBiLSTM/data/ucfTrainTestlist", help="Path to train/test split")
    parser.add_argument("--split_number", type=int, default=1, help="train/test split number. One of {1, 2, 3}")
    parser.add_argument("--num_epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Size of each training batch")
    parser.add_argument("--sequence_length", type=int, default=16, help="Number of frames in each sequence")
    parser.add_argument("--img_dim", type=int, default=112, help="Height / width dimension")
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
    parser.add_argument("--latent_dim", type=int, default=512, help="Dimensionality of the latent representation")
    parser.add_argument("--checkpoint_model", type=str, default="", help="Optional path to checkpoint model")
    parser.add_argument(
        "--checkpoint_interval", type=int, default=5, help="Interval between saving model checkpoints"
    )
    opt = parser.parse_args()
    print(opt)
    save_dir = "saved_models_c3d"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    

    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

    save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
    
    
    

    image_shape = (opt.channels, opt.img_dim, opt.img_dim)

    # Define training set
    train_dataset = Dataset(
        dataset_path=opt.dataset_path,
        split_path=opt.split_path,
        split_number=opt.split_number,
        input_shape=image_shape,
        sequence_length=opt.sequence_length,
        training=True,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True,shuffle=True, num_workers=8)

    # Define test set
    test_dataset = Dataset(
        dataset_path=opt.dataset_path,
        split_path=opt.split_path,
        split_number=opt.split_number,
        input_shape=image_shape,
        sequence_length=opt.sequence_length,
        training=False,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8)

    # Classification criterion
    cls_criterion = nn.CrossEntropyLoss().to(device)
    lr = 1e-4
    # Define network
    model = C3D(num_classes=101, pretrained=True)
    train_params = [{'params': get_1x_lr_params(model), 'lr': lr},
                    {'params': get_10x_lr_params(model), 'lr': lr * 10}]
    #model = i3d_resnet18(num_classes=101)
    # model = get_res3d_model()
    # model = C3D(pretrained=False,num_classes=101)
    # model = CNN3D(t_dim=opt.sequence_length,img_x=image_shape[-1],img_y=image_shape[-2],num_classes=101)
    model = model.to(device)
    # train_params = [{'params': get_1x_lr_params(model), 'lr': lr},
    #                 {'params': get_10x_lr_params(model), 'lr': lr * 10}]
    # Add weights from checkpoint model if specified
    if opt.checkpoint_model:
        model.load_state_dict(torch.load(opt.checkpoint_model))
    training_parameter = model.parameters()
    # crnn_params = list(model.encoder.final.parameters()) + list(model.lstm.parameters())
    optimizer = torch.optim.SGD(training_parameter, lr=lr,momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs
    #optimizer = torch.optim.Adam(training_parameter, lr=lr,weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.num_epochs)
    
    log_dir = os.path.join(save_dir, 'models', datetime.datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    train_loss_values = np.zeros([opt.num_epochs])
    train_acc_values = np.zeros([opt.num_epochs])
    test_loss_values = np.zeros([opt.num_epochs])
    test_acc_values = np.zeros([opt.num_epochs])
    def ts_model(epoch):
        """ Evaluate the model on the test set """
        # print("")
        model.eval()
        test_metrics = {"loss": [], "acc": []}
        for batch_i, (X, y) in enumerate(test_dataloader):
            image_sequences = Variable(X.to(device), requires_grad=False)
            labels = Variable(y, requires_grad=False).to(device)
            with torch.no_grad():
                # Reset LSTM hidden state
                predictions = model(image_sequences)
                # probs = nn.Softmax(dim=1)(predictions)
                # preds = torch.max(probs, 1)[1]
            # Compute metrics
            acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()
            loss = cls_criterion(predictions, labels).item()
            # Keep track of loss and accuracy
            test_metrics["loss"].append(loss)
            test_metrics["acc"].append(acc)
            # Log test performance
            sys.stdout.write(
                "\rTesting -- [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%)]"
                % (
                    batch_i,
                    len(test_dataloader),
                    loss,
                    np.mean(test_metrics["loss"]),
                    acc,
                    np.mean(test_metrics["acc"]),
                )
            )
        model.train()
        # print("")
        return np.mean(test_metrics["loss"]),np.mean(test_metrics["acc"])

    best_acc = 0

    for epoch in range(opt.num_epochs):
        epoch_metrics = {"loss": [], "acc": []}
        prev_time = time.time()
        print(f"--- Epoch {epoch} ---")

        for batch_i, (X, y) in enumerate(train_dataloader):
            model.train()
            if X.size(0) == 1:
                continue

            image_sequences = Variable(X.to(device), requires_grad=True)
            labels = Variable(y.to(device), requires_grad=False)

            optimizer.zero_grad()

            # Get sequence predictions
            predictions = model(image_sequences)
            # probs = nn.Softmax(dim=1)(predictions)
            # preds = torch.max(probs, 1)[1]
            # Compute metrics
            loss = cls_criterion(predictions, labels)
            acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()

            loss.backward()
            optimizer.step()

            # Keep track of epoch metrics
            epoch_metrics["loss"].append(loss.item())
            epoch_metrics["acc"].append(acc)

            # Determine approximate time left
            batches_done = epoch * len(train_dataloader) + batch_i
            batches_left = opt.num_epochs * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%)] ETA: %s"
                % (
                    epoch,
                    opt.num_epochs,
                    batch_i,
                    len(train_dataloader),
                    loss.item(),
                    np.mean(epoch_metrics["loss"]),
                    acc,
                    np.mean(epoch_metrics["acc"]),
                    time_left,
                )
            )
            # epoch_metrics = {"loss": [], "acc": []}


            # Empty cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        scheduler.step()
        train_loss_values[epoch] = np.mean(epoch_metrics["loss"])
        train_acc_values[epoch] = np.mean(epoch_metrics["acc"])
        writer.add_scalar('data/train_loss_epoch', train_loss_values[epoch], epoch)
        writer.add_scalar('data/train_acc_epoch', train_acc_values[epoch], epoch)
        # Evaluate the model on the test set
        test_loss ,test_acc = ts_model(epoch)
        test_loss_values[epoch] = test_loss
        test_acc_values[epoch] = test_acc
        writer.add_scalar('data/test_loss_epoch', test_loss_values[epoch], epoch)
        writer.add_scalar('data/test_acc_epoch', test_acc_values[epoch], epoch)
        # Save model checkpoint
        if epoch % opt.checkpoint_interval == 0:
            # os.makedirs("model_checkpoints", exist_ok=True)
            torch.save(model.state_dict(), save_dir+os.sep+f"{model.__class__.__name__}_{epoch}.pth")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_dir+os.sep+"best.pth")

        print(f"\n train_acc {train_acc_values[epoch]},test_acc {test_acc_values[epoch]}")
        with open(os.path.join(save_dir, "epoch_val_result.txt"), 'a') as f:
            f.write(f"epoch: {epoch},train_acc_score:{train_acc_values[epoch]}, val_acc_score: {test_acc_values[epoch]}\n")
            
    writer.close()

