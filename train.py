import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from vgg16 import vgg16
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lrs
from data import get_training_set
import socket
import time
import random
import torch.nn as nn

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-6, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--decay', type=int, default='100', help='learning rat0e decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data_dir', type=str, default='../dataset/train')
parser.add_argument('--eval_dir', type=str, default='../dataset/eval')
parser.add_argument('--im0_train_dataset', type=str, default='original')
parser.add_argument('--im1_train_dataset', type=str, default='high-quality')
parser.add_argument('--im2_train_dataset', type=str, default='low-quality')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--model_type', type=str, default='IQA')
parser.add_argument('--pretrained_model', type=str, default='.pth', help='pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='/', help='Location to save checkpoint models')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

opt = parser.parse_args()
device = torch.device(opt.device)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

seed = opt.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

best_acc = 0
best_epoch_acc = 0


def train(epoch):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        im1, im2, score = batch[0], batch[1], batch[2]
        if cuda:
            im1 = im1.to(device, dtype=torch.float32)
            im2 = im2.to(device, dtype=torch.float32)
            score = score.to(device, dtype=torch.float32)

        t0 = time.time()        
        out_1 = model(im1)
        out_2 = model(im2)
        loss = margin_loss(out_1, out_2, score)
        optimizer.zero_grad()
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        t1 = time.time()
        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Learning rate: lr={} || Timer: {:.4f} sec.".format(epoch, iteration, 
                          len(training_data_loader), loss.item(), optimizer.param_groups[0]['lr'], (t1 - t0)))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def checkpoint(epoch):
    model_out_path = opt.save_folder+hostname+opt.model_type+"_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


print('===> Loading datasets')
train_set = get_training_set(opt.data_dir, opt.im0_train_dataset, opt.im1_train_dataset, opt.im2_train_dataset)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

print('===> Building model ', opt.model_type)

model = vgg16().to(device)
margin_loss = nn.MarginRankingLoss(0.5).to(device)

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_model)
    if os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained model is loaded.')

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

milestones = []
for i in range(1, opt.nEpochs + 1):
    if i % opt.decay == 0:
        milestones.append(i)

scheduler = lrs.MultiStepLR(optimizer, milestones, opt.gamma)


for epoch in range(opt.start_iter, opt.nEpochs + 1):

    train(epoch)
    scheduler.step()
    checkpoint(epoch)


