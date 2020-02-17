import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.autograd import Variable
from models import DnCNN, Net
# from DHDN_gray import Net
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs/RUN-S-15", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set68', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=15, help='noise level used on test set')
opt = parser.parse_args()

def normalize(data):
    return data/255.

def main():
    # Build model
    print('Loading model ...\n')
    net = Net()
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    a = torch.load(glob.glob(os.path.join(opt.logdir, '*.pth'))[0])
    model.load_state_dict(a)
    DHDN_flag = 4
    # model.load_state_dict(a["model"].state_dict())
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()
    # process data
    psnr_test = 0
    c = 1
    for f in files_source:
        # image
        Img = cv2.imread(f)
        a = Img.shape[0]
        b = Img.shape[1]
        Img2 = normalize(np.float32(Img[:,:,0]))
        Img2 = np.expand_dims(Img2, 0)
        Img2 = np.expand_dims(Img2, 1)
        RSource = torch.Tensor(Img2)
        if a % DHDN_flag != 0 or b % DHDN_flag != 0:
            h = DHDN_flag - (a % DHDN_flag)
            w = DHDN_flag - (b % DHDN_flag)
            Img = np.pad(Img, [(h//2, h-h//2), (w//2, w-w//2), (0, 0)], mode='edge')
        Img = normalize(np.float32(Img[:,:,0]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)
        # noise
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
        # noisy image
        INoisy = ISource + noise
        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
        with torch.no_grad(): # this can save much memory
            Out = torch.clamp(model(INoisy), 0., 1.)
        ## if you are using older version of PyTorch, torch.no_grad() may not be supported
        # ISource, INoisy = Variable(ISource.cuda(),volatile=True), Variable(INoisy.cuda(),volatile=True)
        # Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
        if a % DHDN_flag != 0 or b % DHDN_flag != 0:
            h = DHDN_flag - (a % DHDN_flag)
            w = DHDN_flag - (b % DHDN_flag)
            Out = Out[:,:,h//2:Img.shape[0]-(h-h//2+1), w//2:Img.shape[1]-(w-w//2+1)]
        save_image(Out, "denoise_" + str(c) + ".png")
        psnr = batch_PSNR(Out, RSource, 1.)
        psnr_test += psnr
        print("%s PSNR %f" % (f, psnr))
        c += 1
    psnr_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)

if __name__ == "__main__":
    main()
