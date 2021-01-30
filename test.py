
from __future__ import print_function
import argparse
import torch
from torch.utils.data import DataLoader
from vgg16 import vgg16
from data import get_eval_set
from torchvision.utils import save_image

# settings
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--input_dir', type=str, default='dataset/eval')
parser.add_argument('--model_type', type=str, default='IQA')
parser.add_argument('--model', default='Twice-Mixing-model.pth', help='Pretrained base model')

opt = parser.parse_args()
print(opt)
device = torch.device(opt.device)
cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

print('===> Loading datasets')
test_set = get_eval_set(opt.input_dir)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')

model = vgg16().to(device)

model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
print('Pre-trained model is loaded.')

model.eval()
torch.set_grad_enabled(False)
for batch in testing_data_loader:
    with torch.no_grad():
        input, name = batch[0], batch[1]
        if cuda:
            input = input.to(device)
            with torch.no_grad():
                score = model(input)
                score = score.squeeze().cpu().numpy()
                save_image(input.squeeze().data, './results/LQ/{:.4f}_{}'.format(score, name[0]))

