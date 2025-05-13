"""
This code is based on the following two repositories
* https://github.com/google-research/augmix
* https://github.com/erichson/NoisyMixup
"""

import argparse
import os

import augmentations  # 确保这个模块已正确导入
import numpy as np

from src.cifar_models.preresnet import preactwideresnet18, preactresnet18
#from src.cifar_models.wideresnet import wideresnet28  # 确保导入 wideresnet28
from torchvision.datasets import VisionDataset  # 修正后的导入

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from src.noisy_mixup import mixup_criterion  # 确保这个模块已正确导入
from src.tools import get_lr  # 确保这个模块已正确导入
from aug_utils import *  # 确保这个模块已正确导入

from PIL import Image  # 确保导入 PIL


parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'], help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--arch', '-m', type=str, default='preactresnet18',
    choices=['preactresnet18', 'preactwideresnet18', 'wideresnet28'], help='Choose architecture.')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs to train.')  # 建议增加 epochs 数量
parser.add_argument('--learning-rate', '-lr', type=float, default=0.1, help='Initial learning rate.')
parser.add_argument('--train-batch-size', type=int, default=128, help='Batch size.')
parser.add_argument('--test-batch-size', type=int, default=1000, help='Test batch size.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-wd', type=float, default=0.0005, help='Weight decay (L2 penalty).')

# AugMix options
parser.add_argument('--augmix', type=int, default=1, metavar='S', help='aug mixup (default: 1)')
parser.add_argument('--mixture-width', default=3, type=int, help='Number of augmentation chains to mix per augmented example')
parser.add_argument('--mixture-depth', default=-1, type=int, help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument('--aug-severity', default=3, type=int, help='Severity of base augmentation operators')
parser.add_argument('--jsd', type=int, default=1, metavar='S', help='JSD consistency loss (default: 1)')

parser.add_argument('--all-ops', '-all', action='store_true', help='Turn on all operations (+brightness,contrast,color,sharpness).')

# Noisy Feature Mixup options
parser.add_argument('--alpha', type=float, default=1.0, metavar='S', help='for mixup')  # 根据用户之前的运行命令设置为 1.0
parser.add_argument('--manifold_mixup', type=int, default=1, metavar='S', help='manifold mixup (default: 1)')
parser.add_argument('--add_noise_level', type=float, default=0.1, metavar='S', help='level of additive noise')
parser.add_argument('--mult_noise_level', type=float, default=0.1, metavar='S', help='level of multiplicative noise')
parser.add_argument('--sparse_level', type=float, default=0.2, metavar='S', help='sparse noise')

args = parser.parse_args()

# 定义噪声类型
NOISE_TYPES = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    
    "defocus_blur",
    "glass_blur",    
    "motion_blur",
    "zoom_blur",

    "snow",
    "frost",
    "fog",
    "brightness",
    
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]

# 定义 CIFARCorrupt 类
class CIFARCorrupt(VisionDataset):

    def __init__(self,
                 root="data/CIFAR-100-C",
                 severity=[1, 2, 3, 4, 5],
                 noise=None,
                 transform=None,
                 target_transform=None):
        super(CIFARCorrupt, self).__init__(root, transform=transform, target_transform=target_transform)

        noise = NOISE_TYPES if noise is None else noise

        X = []
        for n in noise:
            D = np.load(os.path.join(root, f"{n}.npy"))
            D_s = np.split(D, 5, axis=0)
            for s in severity:
                X.append(D_s[s - 1])
        X = np.concatenate(X, axis=0)
        Y = np.load(os.path.join(root, "labels.npy"))
        Y_s = np.split(Y, 5, axis=0)
        Y = np.concatenate([Y_s[s - 1] for s in severity])
        Y = np.repeat(Y, len(noise))

        self.data = X
        self.targets = Y
        self.noise_to_nsamples = (noise, X.shape, Y.shape)
        print(f"Loaded data with severities {severity} and noises {noise}: X {X.shape}, Y {Y.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # 将 numpy 数组转换为 PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

def test_on_cifar100c(net, cifar10c_root, transform, device='cuda'):
    """在 CIFAR-10C 上测试模型，并返回每种腐蚀类型和严重程度的准确率。"""
    net.eval()
    corruption_types = NOISE_TYPES
    severities = [1, 2, 3, 4, 5]
    results = {}

    for corruption in corruption_types:
        results[corruption] = {}
        for severity in severities:
            # 加载单一腐蚀类型和严重程度的数据
            test_set = CIFARCorrupt(
                root=cifar10c_root,
                severity=[severity],
                noise=[corruption],
                transform=transform
            )
            test_loader = torch.utils.data.DataLoader(
                test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True
            )

            total_correct = 0
            total = 0
            with torch.no_grad():
                for images, targets in test_loader:
                    images, targets = images.to(device), targets.to(device)
                    outputs = net(images)
                    _, preds = torch.max(outputs, 1)
                    total_correct += (preds == targets).sum().item()
                    total += targets.size(0)
            accuracy = total_correct / total
            results[corruption][severity] = accuracy
            print(f"Corruption: {corruption}, Severity: {severity}, Accuracy: {accuracy * 100:.2f}%")
    # Compute Overall Accuracy
    all_accuracies = []
    for corruption in corruption_types:
        for severity in severities:
            all_accuracies.append(results[corruption][severity])
    overall_accuracy = np.mean(all_accuracies) * 100
    print(f"\nOverall CIFAR-100-C Accuracy: {overall_accuracy:.2f}%")
    return results,overall_accuracy

class oldCustomLoss(nn.Module):
    def __init__(self, rho=0.05, q=1):
        """
        Initializes the adCustomLoss module.

        Args:
            rho (float): Weighting factor for the variation term.
            q (float): The norm degree for variation calculation. Supports any q >= 1 and float('inf') for infinity norm.
        """
        super(oldCustomLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.rho = rho
        self.training_mode = True
        self.q = q

    def forward(self, outputs, targets, model=None, images=None):
        """
        Computes the custom loss combining cross-entropy and gradient-based variation.

        Args:
            outputs (torch.Tensor): Model outputs (logits) of shape (batch_size, num_classes).
            targets (torch.Tensor or tuple): True labels or mixup targets.
            model (nn.Module, optional): The neural network model. Required if variation is computed.
            images (torch.Tensor, optional): Input images tensor of shape (batch_size, C, H, W). Required if variation is computed.

        Returns:
            torch.Tensor: The computed total loss.
        """
        # Compute Cross-Entropy Loss
        if isinstance(targets, tuple):
            y_a, y_b, lam = targets
            ce_loss = lam * self.cross_entropy(outputs, y_a) + (1 - lam) * self.cross_entropy(outputs, y_b)
        else:
            ce_loss = self.cross_entropy(outputs, targets)
        
        # Compute Variation if in training mode and model/images are provided
        if self.training_mode and model is not None and images is not None:
            variation = self.calculate_variation(model, images, targets)
            total_loss = ce_loss + self.rho * variation
        else:
            total_loss = ce_loss
        
        return total_loss

    def calculate_variation(self, model, images, targets):
        """
        Calculates the gradient-based variation metric.

        Args:
            model (nn.Module): The neural network model.
            images (torch.Tensor): Input images tensor of shape (batch_size, C, H, W).
            targets (torch.Tensor or tuple): True labels or mixup targets.

        Returns:
            float: The computed variation metric.
        """
        # Ensure the model is in evaluation mode to prevent unwanted behavior
        was_training = model.training
        model.eval()

        # Clone images to avoid modifying the original data and enable gradient computation
        images = images.clone().detach().requires_grad_(True).to(next(model.parameters()).device)

        # Forward pass
        outputs = model(images)
        
        # Compute loss
        if isinstance(targets, tuple):
            y_a, y_b, lam = targets
            loss = lam * self.cross_entropy(outputs, y_a) + (1 - lam) * self.cross_entropy(outputs, y_b)
        else:
            loss = self.cross_entropy(outputs, targets)
        
        # Zero existing gradients
        model.zero_grad()

        # Compute gradients w.r. to images
        grads = torch.autograd.grad(loss, images, create_graph=False, retain_graph=False)[0]  # Shape: (batch_size, C, H, W)

        # Reshape gradients to (batch_size, -1)
        gradients = grads.view(grads.size(0), -1)  # Shape: (batch_size, C*H*W)

        # Compute variation based on the specified norm (q)
        if self.q == float('inf'):
            # Infinity norm: maximum absolute gradient value across each sample
            variation_per_sample = gradients.abs().max(dim=1)[0]  # Shape: (batch_size,)
            variation = variation_per_sample.max().item()  # Scalar
        else:
            # q-norm for each sample
            variation_per_sample = gradients.norm(p=self.q, dim=1)  # Shape: (batch_size,)

            # Aggregate variations across the batch
            # Using mean to ensure batch size independence
            variation = variation_per_sample.pow(self.q).mean().pow(1.0 / self.q).item()

        # Restore the model's original training state
        if was_training:
            model.train()

        return variation

    def set_training_mode(self, mode):
        """
        Sets the training mode for variation calculation.

        Args:
            mode (bool): If True, variation is included in the loss. Otherwise, only cross-entropy is used.
        """
        self.training_mode = mode

def train(net, train_loader, optimizer, scheduler, device):
    """Train for one epoch and compute training loss and accuracy."""
    net.train()
    loss_ema = 0.
    correct = 0
    total = 0
    
    criterion = oldCustomLoss(rho=16, q=2).to(device)
    
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        
        
        if args.jsd == 0:
            images = images.to(device)
            targets = targets.to(device)

            if args.alpha == 0.0:   
                outputs = net(images)
                loss = criterion(outputs, targets, model=net, images=images)
            else:
                outputs, targets_a, targets_b, lam = net(
                    images, targets=targets, jsd=args.jsd,
                    mixup_alpha=args.alpha,
                    manifold_mixup=args.manifold_mixup,
                    add_noise_level=args.add_noise_level,
                    mult_noise_level=args.mult_noise_level,
                    sparse_level=args.sparse_level
                )
                loss = criterion(outputs, (targets_a, targets_b, lam), model=net, images=images)
            
            # Compute accuracy
            _, preds = torch.max(outputs, 1)
            if args.alpha > 0:
                correct += (preds == targets_a).sum().item() * lam + \
                           (preds == targets_b).sum().item() * (1 - lam)
            else:
                correct += preds.eq(targets).sum().item()
            total += targets.size(0)
        
        elif args.jsd == 1:
            images_all = torch.cat(images, 0).to(device)
            targets = targets.to(device)      
            
            if args.alpha == 0.0:   
                logits_all = net(images_all)
                logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
                loss = criterion(logits_clean, targets, model=net, images=images[0])
            else:
                logits_all, targets_a, targets_b, lam = net(
                    images_all, targets=targets, jsd=args.jsd, 
                    mixup_alpha=args.alpha,
                    manifold_mixup=args.manifold_mixup,
                    add_noise_level=args.add_noise_level,
                    mult_noise_level=args.mult_noise_level,
                    sparse_level=args.sparse_level
                )
                logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
                loss = criterion(logits_clean, (targets_a, targets_b, lam), model=net, images=images[0])
            
            # Compute accuracy
            _, preds = torch.max(logits_clean, 1)
            if args.alpha > 0:
                correct += (preds == targets_a).sum().item() * lam + \
                           (preds == targets_b).sum().item() * (1 - lam)
            else:
                correct += preds.eq(targets).sum().item()
            total += targets.size(0)
            
            # JSD Loss
            p_clean = F.softmax(logits_clean, dim=1)
            p_aug1 = F.softmax(logits_aug1, dim=1)
            p_aug2 = F.softmax(logits_aug2, dim=1)
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            jsd_loss = 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                             F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                             F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
            loss += jsd_loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_ema = loss_ema * 0.9 + float(loss) * 0.1
    
    train_acc = correct / total
    return loss_ema, train_acc



def test(net, test_loader, device):
    """Evaluate network on given dataset and compute loss and accuracy."""
    net.eval()
    total_loss = 0.
    total_correct = 0
    total = 0
    criterion = oldCustomLoss(rho=12, q=2).to(device)
    criterion.set_training_mode(False)  # Disable variation term during evaluation
    
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            logits = net(images)
            loss = criterion(logits, targets)  # No need to pass model and images
            _, preds = torch.max(logits, 1)
            total_loss += loss.item() * targets.size(0)
            total_correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    
    avg_loss = total_loss / total
    avg_acc = total_correct / total
    criterion.set_training_mode(True)
    return avg_loss, avg_acc


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 数据增强和预处理
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4)])
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5] * 3, [0.5] * 3)])
    test_transform = preprocess

    if args.augmix == 0:
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(32, padding=4),
             transforms.ToTensor(),
             transforms.Normalize([0.5] * 3, [0.5] * 3),
             ])

    # 加载训练集
    if args.dataset == 'cifar10':
        train_data = datasets.CIFAR10(
            './data/cifar', train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(
            './data/cifar', train=False, transform=test_transform, download=True)
        num_classes = 10
        # 应用 AugMix
        if args.augmix == 1:
            train_data = AugMixDataset(train_data, preprocess, args.jsd, args)
    else:
        train_data = datasets.CIFAR100(
            './data/cifar', train=True, transform=train_transform, download=True)
        num_classes = 100
        test_data = datasets.CIFAR100(
            './data/cifar', train=False, transform=test_transform, download=True)
        num_classes = 100
        # 这里可以添加类似的 AugMix 处理，如果需要的话
        if args.augmix == 1:
            train_data = AugMixDataset(train_data, preprocess, args.jsd, args)

    train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.train_batch_size,
            shuffle=True, num_workers=4, pin_memory=True)          
    test_loader = torch.utils.data.DataLoader(
          test_data, batch_size=args.test_batch_size,
          shuffle=False, num_workers=4, pin_memory=True)
    # 定义 CIFAR-10C 的根目录
    CIFAR100C_FOLDER = './data/CIFAR-100-C/'
 

    # 创建模型
    if args.arch == 'preactresnet18':
        net = preactresnet18(num_classes=num_classes)
    elif args.arch == 'preactwideresnet18':
        net = preactwideresnet18(num_classes=num_classes)
    elif args.arch == 'wideresnet28':
        net = wideresnet28(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")

    optimizer = torch.optim.SGD(net.parameters(),
        args.learning_rate, momentum=args.momentum,
        weight_decay=args.decay, nesterov=True)

    # 将模型分布到所有可用的 GPU
    net = torch.nn.DataParallel(net).to(device)

    start_epoch = 0

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(
            step, args.epochs * len(train_loader),
            1,  # lr_lambda 计算乘法因子
            1e-6 / args.learning_rate))

    # 初始化最佳验证准确率
    best_acc = 0
      
    for epoch in range(start_epoch, args.epochs):
            
                train_loss_ema, train_acc = train(net, train_loader, optimizer, scheduler, device)
                test_loss, test_acc = test(net, test_loader, device)

                is_best = test_acc > best_acc
                best_acc = max(test_acc, best_acc)
            
                if is_best:
                  DESTINATION_PATH = args.dataset + '_models/'
                  OUT_DIR = os.path.join(DESTINATION_PATH, f'best_arch_{args.arch}_augmix_{args.augmix}_jsd_{args.jsd}_alpha_{args.alpha}_manimixup_{args.manifold_mixup}_addn_{args.add_noise_level}_multn_{args.mult_noise_level}_seed_{args.seed}')
                  if not os.path.isdir(DESTINATION_PATH):
                            os.mkdir(DESTINATION_PATH)
                  torch.save(net, OUT_DIR+'.pt')  
            
                print(
                    'Epoch {0:3d} | Train Loss {1:.4f} |'
                    ' Test Accuracy {2:.2f}'
                    .format((epoch + 1), train_loss_ema, 100. * test_acc)) 
    # 保存最终模型
    DESTINATION_PATH = args.dataset + '_models/'
    OUT_DIR = os.path.join(DESTINATION_PATH, f'final_arch_{args.arch}_augmix_{args.augmix}_jsd_{args.jsd}_alpha_{args.alpha}_manimixup_{args.manifold_mixup}_addn_{args.add_noise_level}_multn_{args.mult_noise_level}_seed_{args.seed}')
    if not os.path.isdir(DESTINATION_PATH):
        os.mkdir(DESTINATION_PATH)
    torch.save(net.state_dict(), OUT_DIR+'.pt')
    print(f"Final model saved to {OUT_DIR+'.pt'}")

    # 使用最终模型在 CIFAR-10C 上进行评估
    print("开始在 CIFAR-10C 上进行评估...")
    results, overall_accuracy = test_on_cifar100c(net, CIFAR100C_FOLDER, test_transform, device=device)
    # 保存结果到文件
    with open(os.path.join(DESTINATION_PATH, 'cifar100c_results.txt'), 'w') as f:
        for corruption, severities_acc in results.items():
            for severity, acc in severities_acc.items():
                f.write(f"Corruption: {corruption}, Severity: {severity}, Accuracy: {acc * 100:.2f}%\n")
    print("CIFAR-100C 测试完成，结果已保存。")
    print(f"\nOverall CIFAR-100-C Accuracy: {overall_accuracy:.2f}%")

if __name__ == '__main__':
    main()
