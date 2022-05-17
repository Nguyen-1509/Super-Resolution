import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.optim as optim
import numpy as np
import math
from imutils import paths
import os
import PIL.Image as Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import transforms
from torch.utils.data import Dataset


class DivDataset(Dataset):

    def __init__(self, data_root, mode='train'):
        super(DivDataset, self).__init__()
        self.data_root = data_root
        self.list_images = [f for f in os.listdir(self.data_root) if f.endswith('.png')]
        self.mode = mode
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):

        image = cv2.imread(self.data_root + self.list_images[index], cv2.IMREAD_UNCHANGED)
        if self.mode == 'train':

            randomcrop = torchvision.transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(500)])
            hr_image = randomcrop(image)
            downscale = torchvision.transforms.Compose([transforms.Resize(256)])
            lr_image = downscale(hr_image)

            upscale = torchvision.transforms.Compose([transforms.Resize(500, interpolation=InterpolationMode.BICUBIC)])
            lr_image = upscale(lr_image)

            return self.transform(lr_image), self.transform(hr_image)

        elif self.mode == 'test':
            centercrop = torchvision.transforms.Compose([transforms.ToPILImage(), transforms.CenterCrop(500)])
            hr_image = centercrop(image)

            downscale = torchvision.transforms.Compose([transforms.Resize(256)])
            lr_image = downscale(hr_image)

            upscale = torchvision.transforms.Compose([transforms.Resize(500, interpolation=InterpolationMode.BICUBIC)])
            lr_image = upscale(lr_image)

            return self.transform(lr_image)
        else:
            raise AttributeError("The mode should be set to either 'train' or 'test' ")

    def __len__(self):
        return len(self.list_images)

class SRCNN(nn.Module):
  def __init__(self):
    super(SRCNN,self).__init__()
    #feature extraction layer
    self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
    self.relu = nn.ReLU(inplace=True)
    # Non-linear mapping layer.
    self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
    # Rebuild the layer.
    self.conv3 = nn.Conv2d(32,3,kernel_size=5, padding=2)
  def forward(self, X):
    out = self.relu(self.conv1(X))
    out = self.relu(self.conv2(out))
    out = self.conv3(out)
    return out
class Res_block(nn.Module):
  def __init__(self):
    super(Res_block,self).__init__()
    self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
    self.batch = nn.BatchNorm2d(64)
    self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
    self.relu = nn.ReLU(inplace=True)
  def forward(self,x):
    x_init = x
    out = self.relu(self.batch(self.conv1(x)))
    out = self.relu(self.batch(self.conv2(out)))
    out = torch.add(out,x_init)
    return out
class SRresnet(nn.Module):
  def __init__(self):
    super(SRresnet,self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
    self.relu = nn.ReLU(inplace=True)
    self.layer1 = Res_block()
    self.layer2 = Res_block()
    self.layer3 = Res_block()
    self.layer4 = Res_block()
    self.layer5 = Res_block()
    self.conv2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)
  def forward(self, x):
    x = self.relu(self.conv1(x))
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.layer5(out)
    out = self.conv2(out)
    return out


def PSNR(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


if __name__ == "__main__":
    train_path = './archive/DIV2K_train_HR/DIV2K_train_HR/'
    valid_path = './archive/DIV2K_valid_HR/DIV2K_valid_HR/'
    # img = cv2.imread(valid_path + '0900.png')

    # data = DivDataset(data_root=train_path, mode='train')
    # img, _ = data[0]
    # print(img)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyWindow("img")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    Train_dataset = DivDataset(data_root=train_path, mode='train')
    Test_dataset = DivDataset(data_root=valid_path, mode='train')
    Train_dataloader = DataLoader(Train_dataset, batch_size=3, shuffle=True, num_workers=1, pin_memory=True,
                                  drop_last=True, persistent_workers=True)
    Test_dataloader = DataLoader(Test_dataset, batch_size=3, shuffle=False, num_workers=1, pin_memory=True,
                                 drop_last=False, persistent_workers=False)
    model = SRCNN().to(device=device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.MSELoss().cuda()
    nb_epoch = 40
    # Training
    for epoch in range(0, nb_epoch):
        print('Epoch {}/{}'.format(epoch, nb_epoch - 1))
        print('-' * 20)
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
                running_loss = 0.0
                PSNR_train = 0.0
                for iter, data in enumerate(Train_dataloader):
                    optimizer.zero_grad()
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels).to(device=device)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    PSNR_train += PSNR(outputs, labels)
                running_loss = running_loss / len(Train_dataloader)
                PSNR_train = PSNR_train / len(Train_dataloader)
                print('Epoch {} --- Loss: {} -- PSNR: {}'.format(epoch, running_loss, PSNR_train))
            if phase == "test":
                model.eval()
                test_loss = 0.0
                PSNR_test = 0.0
                for iter, data in enumerate(Test_dataloader):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels).to(device=device)
                    test_loss += loss.item()
                    PSNR_test += PSNR(outputs, labels)
                test_loss = test_loss / len(Test_dataloader)
                PSNR_test = PSNR_test/len(Test_dataloader)
                print('Epoch {} --- Loss_Test: {}--PSNR:{}'.format(epoch, test_loss, PSNR_test))

