# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Main script to launch AugMix training on CIFAR-10/100.

Supports WideResNet, AllConv, ResNeXt models on CIFAR-10 and CIFAR-100 as well
as evaluation on CIFAR-10-C and CIFAR-100-C.

Example usage:
  `python cifar.py`
"""
from __future__ import print_function

import argparse
import os
import shutil
import time
# import cv2
from PIL import Image

import augmentation_with_WCT2 as augmentations
# import augmentation_with_WCT2_code as augmentations
# import augmentations
from models.cifar.allconv import AllConvNet
import numpy as np
from third_party.ResNeXt_DenseNet.models.densenet import densenet
from third_party.ResNeXt_DenseNet.models.resnext import resnext29
from third_party.WideResNet_pytorch.wideresnet import WideResNet

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

parser = argparse.ArgumentParser(
    description='Trains a CIFAR Classifier',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=['cifar10', 'cifar100'],
    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument(
    '--model',
    '-m',
    type=str,
    default='wrn',
    choices=['wrn', 'allconv', 'densenet', 'resnext'],
    help='Choose architecture.')
# Optimization options
parser.add_argument(
    '--epochs', '-e', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument(
    '--learning-rate',
    '-lr',
    type=float,
    default=0.1,
    help='Initial learning rate.')
parser.add_argument(
    '--batch-size', '-b', type=int, default=64, help='Batch size.')
parser.add_argument('--eval-batch-size', type=int, default=1000)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument(
    '--decay',
    '-wd',
    type=float,
    default=0.0005,
    help='Weight decay (L2 penalty).')
# WRN Architecture options
parser.add_argument(
    '--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='Widen factor')
parser.add_argument(
    '--droprate', default=0.0, type=float, help='Dropout probability')
# AugMix options
parser.add_argument(
    '--mixture-width',
    default=3,
    type=int,
    help='Number of augmentation chains to mix per augmented example')
parser.add_argument(
    '--mixture-depth',
    default=-1,
    type=int,
    help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument(
    '--aug-severity',
    default=3,
    type=int,
    help='Severity of base augmentation operators')
parser.add_argument(
    '--no-jsd',
    '-nj',
    action='store_true',
    help='Turn off JSD consistency loss.')
parser.add_argument(
    '--all-ops',
    '-all',
    action='store_true',
    help='Turn on all operations (+brightness,contrast,color,sharpness).')
# Checkpointing options
parser.add_argument(
    '--save',
    '-s',
    type=str,
    default=None,
    help='Folder to save checkpoints.')
# parser.add_argument(
#     '--resume',
#     '-r',
#     type=str,
#     default='',
#     help='Checkpoint path for resume / test.')
parser.add_argument('--evaluate', action='store_true', help='Eval only.')
parser.add_argument(
    '--print-freq',
    type=int,
    default=50,
    help='Training loss print frequency (batches).')
# Acceleration
parser.add_argument(
    '--num-workers',
    type=int,
    default=4,
    help='Number of pre-fetching threads.')

parser.add_argument(
    '--corrup',
    default=None,
    type=str,
    help='Corruption')

parser.add_argument(
    '--duel-corrup',
    type=str,
    default=None,
    help='Duel corruption')

parser.add_argument(
    '--gpu',
    type=int,
    default=1,
    help='GPU')

args = parser.parse_args()

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

class CorruptionDataset(torch.utils.data.Dataset):
  def __init__(self, imgdata, tagdata,
               transform=None, target_transform=None):
    self.imgdata = imgdata
    self.tagdata = tagdata
    self.transform = transform
    self.target_transform = target_transform
  
  def __getitem__(self, index):
    # return self.imgdata[index], self.tagdata[index]
    img, target = self.imgdata[index], self.tagdata[index]

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = Image.fromarray(img)
    target = int(target)

    if self.transform is not None:
        img = self.transform(img)

    if self.target_transform is not None:
        target = self.target_transform(target)

    return img, target

  def __len__(self):
    return len(self.tagdata)


def get_lr(step, total_steps, lr_max, lr_min):
  """Compute learning rate according to cosine annealing schedule."""
  return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                             np.cos(step / total_steps * np.pi))

def aug(image, preprocess, idx, lbl):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

  Returns:
    mixed: Augmented and mixed image.
  """
  aug_list = augmentations.augmentations
  if args.all_ops:
    aug_list = augmentations.augmentations_all

  ws = np.float32(np.random.dirichlet([1] * args.mixture_width))
  m = np.float32(np.random.beta(1, 1))

  mix = torch.zeros_like(preprocess(image))
  for i in range(args.mixture_width):
    image_aug = image.copy()
    depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(
        1, 4)
    op_code = []
    has_WCT2 = False
    for dep in range(depth):
      op = np.random.choice(aug_list)
      if op == aug_list[-1]:
        has_WCT2 = True
        continue
      op_code.append(op)
      # image_aug = op(image_aug, args.aug_severity, lbl)
      # image_aug = op(image_aug, args.aug_severity, idx, dep)
      # image_aug = op(image_aug, args.aug_severity)
    if has_WCT2:
      image_aug = aug_list[-1](image_aug, args.corrup, idx, dep)
    for op in op_code:
      image_aug = op(image_aug, args.aug_severity, idx, dep)

    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug)

  mixed = (1 - m) * preprocess(image) + m * mix
  return mixed


class AugMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess, no_jsd=False):
    self.dataset = dataset
    self.preprocess = preprocess
    self.no_jsd = no_jsd

  def __getitem__(self, i):
    x, y = self.dataset[i]
    if self.no_jsd:
      return aug(x, self.preprocess, i), y
    else:
      im_tuple = (self.preprocess(x), aug(x, self.preprocess, i, y),
                  aug(x, self.preprocess, i, y))
      return im_tuple, y

  def __len__(self):
    return len(self.dataset)


def train(net, train_loader, optimizer, scheduler):
  """Train for one epoch."""
  net.train()
  loss_ema = 0.
  for i, (images, targets) in enumerate(train_loader):
    optimizer.zero_grad()

    if args.no_jsd:
      images = images.cuda()
      targets = targets.cuda()
      logits = net(images)
      loss = F.cross_entropy(logits, targets)
    else:
      images_all = torch.cat(images, 0).cuda()
      targets = targets.cuda()
      logits_all = net(images_all)
      logits_clean, logits_aug1, logits_aug2 = torch.split(
          logits_all, images[0].size(0))

      # Cross-entropy is only computed on clean images
      loss = F.cross_entropy(logits_clean, targets)

      p_clean, p_aug1, p_aug2 = F.softmax(
          logits_clean, dim=1), F.softmax(
              logits_aug1, dim=1), F.softmax(
                  logits_aug2, dim=1)

      # Clamp mixture distribution to avoid exploding KL divergence
      p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
      loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    loss.backward()
    optimizer.step()
    scheduler.step()
    loss_ema = loss_ema * 0.9 + float(loss) * 0.1
    if i % args.print_freq == 0:
      print('Train Loss {:.3f}'.format(loss_ema))

  return loss_ema


def test(net, test_loader):
  """Evaluate network on given dataset."""
  net.eval()
  total_loss = 0.
  total_correct = 0
  with torch.no_grad():
    for images, targets in test_loader:
      images, targets = images.cuda(), targets.cuda()
      logits = net(images)
      loss = F.cross_entropy(logits, targets)
      pred = logits.data.max(1)[1]
      total_loss += float(loss.data)
      total_correct += pred.eq(targets.data).sum().item()

  return total_loss / len(test_loader.dataset), total_correct / len(
      test_loader.dataset)

      
def test_c_s(net, test_loader, corrup):
  """Evaluate network on given dataset."""
  # error_list_root = "/DATASET/__saika_data/new_Fewshot/augmix/error_list_from_normal_wrn/"
  # if not os.path.exists(error_list_root):
  #   os.mkdir(error_list_root)
  # save_error_path = error_list_root + corrup + '.npy'
  net.eval()
  total_loss = 0.
  total_correct = 0
  # tot_idx = 0
  # error_list = []
  with torch.no_grad():
    for images, targets in test_loader:
      tag_list = targets.numpy()
      images, targets = images.cuda(), targets.cuda()
      logits = net(images)
      loss = F.cross_entropy(logits, targets)
      pred = logits.data.max(1)[1]
      total_loss += float(loss.data)
      total_correct += pred.eq(targets.data).sum().item()
      # pred_list = pred.cpu().numpy()
      # for idx in range(len(tag_list)):
      #   if tag_list[idx] != pred_list[idx]:
      #     error_list.append(tot_idx)
      #   tot_idx += 1
    #   for idx in range()
    #   print(pred.eq(targets.data).sum().item())

  return total_loss / len(test_loader.dataset), total_correct / len(
      test_loader.dataset)


def test_c(net, test_data, base_path):
  """Evaluate network on given corrupted dataset."""
  corruption_accs = []
  for corruption in CORRUPTIONS:
    # Reference to original data is mutated
    test_data.data = np.load(base_path + corruption + '.npy')
    test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    test_loss, test_acc = test(net, test_loader)
    corruption_accs.append(test_acc)
    print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
        corruption, test_loss, 100 - 100. * test_acc))

  return np.mean(corruption_accs)

def test_c_failure_single_case(net, test_loader, base_path):
  """Evaluate network on given corrupted dataset."""
  corruption = args.corrup

  test_loss, test_acc = test_c_s(net, test_loader, corruption)

  return test_acc

def main():
  torch.manual_seed(1)
  np.random.seed(1)

  os.environ["CUDA_VISIBLE_DEVICES"]='{}'.format(args.gpu)

  if args.corrup is None:
    print("use --corrup")
    return

  # Load datasets
  train_transform = transforms.Compose(
      [transforms.RandomHorizontalFlip(),
       transforms.RandomCrop(32, padding=4)])
  preprocess = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize([0.5] * 3, [0.5] * 3)])
  test_transform = preprocess

  if args.dataset == 'cifar10':
    train_data = datasets.CIFAR10(
        './data/cifar', train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR10(
        './data/cifar', train=False, transform=test_transform, download=True)
    base_c_path = './data/cifar/CIFAR-10-C/'
    base_c_failure_case_path = '/EX_STORE/augmix/error_list_from_normal_{}_divide/'.format(args.model)
    num_classes = 10
  else:
    train_data = datasets.CIFAR100(
        './data/cifar', train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR100(
        './data/cifar', train=False, transform=test_transform, download=True)
    base_c_path = './data/cifar/CIFAR-100-C/'
    num_classes = 100

  train_data_dir = "/DATASET/__saika_experiment/transfer_shot/transfer_output/gaussian_noise/"
  train_data_list = os.listdir(train_data_dir)
  train_data_list.sort()
  train_img = []
  for train_data_name in train_data_list:
    # t_img = Image.open(r"{}".format(train_data_dir + train_data_name)).crop((0,0,32,32))
    t_img = cv2.imread(train_data_dir + train_data_name)
    # t_img = Image.fromarray(t_img)
    # t_img = train_transform(t_img)
    train_img.append(t_img)
  train_img = np.array(train_img)
  train_data.data = train_img

  test_failure_data = datasets.CIFAR10(
      './data/cifar', train=False, transform=test_transform, download=True)
  id_failure_list = np.load(base_c_failure_case_path + args.corrup + '_test.npy')
  img_failure_path = "/DATASET/__saika_experiment/transfer_shot/CIFAR-10-C/" + args.corrup + ".npy"
  img_failure_list = np.load(img_failure_path)
  test_failure_data.data = np.array(img_failure_list[id_failure_list])
  tag_failure_path = "/DATASET/__saika_experiment/transfer_shot/CIFAR-10-C/labels.npy"
  tag_failure_list = np.load(tag_failure_path)
  test_failure_data.targets = torch.LongTensor(np.array(tag_failure_list[id_failure_list]))

  train_data = AugMixDataset(train_data, preprocess, args.no_jsd)
  train_loader = torch.utils.data.DataLoader(
      train_data,
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=True)

  test_loader = torch.utils.data.DataLoader(
      test_data,
      batch_size=args.eval_batch_size,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=True)
    
  if args.corrup == "whole":
    img_failure_path = "/EX_STORE/augmix/error_data_whole_set/from_{}/whole_data.npy".format(args.model)
    img_failure_list = np.load(img_failure_path)
    tag_failure_path = "/EX_STORE/augmix/error_data_whole_set/from_{}/whole_label.npy".format(args.model)
    tag_failure_list = np.load(tag_failure_path)
    # id_failure_list = np.random.choice(len(tag_failure_list), 50000, replace=False)
    # img_failure_list = img_failure_list[id_failure_list]
    # tag_failure_list = tag_failure_list[id_failure_list]
    test_failure_data = CorruptionDataset(img_failure_list, tag_failure_list, transform=test_transform)
  elif args.corrup in CORRUPTIONS:
    id_failure_list = np.load(base_c_failure_case_path + args.corrup + '_test.npy')
    img_failure_path = "/EX_STORE/CIFAR-10-C/" + args.corrup + ".npy"
    img_failure_list = np.load(img_failure_path)
    img_failure_list = np.array(img_failure_list[id_failure_list])
    tag_failure_path = "/EX_STORE/CIFAR-10-C/labels.npy"
    tag_failure_list = np.load(tag_failure_path)
    tag_failure_list = np.array(tag_failure_list[id_failure_list])
    test_failure_data = CorruptionDataset(img_failure_list, tag_failure_list, transform=test_transform)
  else:
    print("!!! Wrong corruption !!!")
    return

  test_failure_loader = torch.utils.data.DataLoader(
      test_failure_data,
      batch_size=args.eval_batch_size,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=True)

  # Create model
  if args.model == 'densenet':
    net = densenet(num_classes=num_classes)
    # resume_path = "/EX_STORE/augmix/model_original_densenet/model_best.pth.tar"
  elif args.model == 'wrn':
    net = WideResNet(args.layers, num_classes, args.widen_factor, args.droprate)
    # resume_path = "/EX_STORE/augmix/model_original_wrn/model_best.pth.tar"
  elif args.model == 'allconv':
    net = AllConvNet(num_classes)
    # resume_path = "/EX_STORE/augmix/model_original_allconv/model_best.pth.tar"
  elif args.model == 'resnext':
    net = resnext29(num_classes=num_classes)

  optimizer = torch.optim.SGD(
      net.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.decay,
      nesterov=True)

  # Distribute model across all visible GPUs
  net = torch.nn.DataParallel(net).cuda()
  cudnn.benchmark = True

  start_epoch = 0

  do_resume = True
  if do_resume:
    if os.path.isfile(resume_path):
      checkpoint = torch.load(resume_path)
      start_epoch = checkpoint['epoch'] + 1
      best_acc = checkpoint['best_acc']
      net.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      print('Model restored from epoch:', start_epoch)

  scheduler = torch.optim.lr_scheduler.LambdaLR(
      optimizer,
      lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
          step,
          100 * len(train_loader),
          1,  # lr_lambda computes multiplicative factor
          1e-6 / args.learning_rate))

  if args.duel_corrup is None:
    save_root_path = './repair_with_WCT2_{}__{}'.format(args.model, args.corrup)       # <---------------------------------------------------------------- model's save path
  else:
    save_root_path = '.repair_with_/WCT2_{}__{}'.format(args.model, args.duel_corrup)
  if args.save is not None:
    save_root_path = args.save

  if not os.path.exists(save_root_path):
    os.makedirs(save_root_path)
  if not os.path.isdir(save_root_path):
    raise Exception('%s is not a dir' % save_root_path)

  log_path = os.path.join(save_root_path,
                          args.dataset + '_' + args.model + '_training_log.csv')
  with open(log_path, 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

  best_acc = 0
  best_fail_acc = 0
  n_epochs_stop = 5
  epochs_no_improve = 0
  early_stop = False
  min_test_loss = None
  print('Beginning training from epoch:', start_epoch + 1)
  for epoch in range(start_epoch, start_epoch+args.epochs):
    begin_time = time.time()

    train_loss_ema = train(net, train_loader, optimizer, scheduler)
    test_loss, test_acc = test(net, test_loader)
    test_c_acc = test_c_failure_single_case(net, test_failure_loader, base_c_failure_case_path)

    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)

    is_best_failure = test_c_acc > best_fail_acc
    best_fail_acc = max(test_c_acc, best_fail_acc)

    checkpoint = {
        'epoch': epoch,
        'dataset': args.dataset,
        'model': args.model,
        'state_dict': net.state_dict(),
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }

    save_path = os.path.join(save_root_path, 'checkpoint.pth.tar')
    torch.save(checkpoint, save_path)
    if is_best or is_best_failure:
      shutil.copyfile(save_path, os.path.join(save_root_path, 'model_best_pure_{:>.3f}_fail_{:>.3f}.pth.tar'.format(test_acc, test_c_acc)))

    with open(log_path, 'a') as f:
      f.write('%03d, %05d, %0.6f, %0.5f, %0.2f, %0.2f, %0.2f, %0.2f\n' % (
          (epoch + 1),
          time.time() - begin_time,
          train_loss_ema,
          test_loss,
          100 - 100. * test_acc,
          100. * test_c_acc,
          100. * best_acc,
          100. * best_fail_acc,
      ))

    print(
        'Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} |'
        ' Test Error {4:.2f} | Failure Case Accuracy {5:02f}'
        .format((epoch + 1), int(time.time() - begin_time), train_loss_ema,
                test_loss, 100 - 100. * test_acc, 100. * test_c_acc))
    
    if min_test_loss is None:
      min_test_loss = test_loss
    elif min_test_loss > test_loss:
      min_test_loss = test_loss
      epochs_no_improve = 0
    else:
      epochs_no_improve += 1

    if epoch >= n_epochs_stop and epochs_no_improve == n_epochs_stop:
      print("------ Early stop ------")
      break

  test_c_acc = test_c(net, test_data, base_c_path)
  print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))

  with open(log_path, 'a') as f:
    f.write('%03d,%05d,%0.6f,%0.5f,%0.2f,%0.2f\n' %
            (args.epochs + 1, 0, 0, 0, 100 - 100 * test_acc, 100. * test_c_acc))


if __name__ == '__main__':
  main()
