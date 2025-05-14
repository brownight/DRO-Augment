"""
This code is based on the following two repositories
* https://github.com/google-research/augmix
* https://github.com/erichson/NoisyMixup
"""
import torchattacks 
import argparse
import os

import augmentations  
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms, models
from src.cifar_models.preresnet import preactwideresnet18, preactresnet18
#from src.cifar_models.wideresnet import wideresnet28  
from torchvision.datasets import VisionDataset 

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from src.noisy_mixup import mixup_criterion  
from src.tools import get_lr  
from aug_utils import *  

from PIL import Image  


parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--arch', '-m', type=str, default='preactresnet18',
    choices=['preactresnet18', 'preactwideresnet18', 'wideresnet28'], help='Choose architecture.')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=50, help='Number of epochs to train.')  # 建议增加 epochs 数量
parser.add_argument('--learning-rate', '-lr', type=float, default=0.001, help='Initial learning rate.')
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


def evaluate_under_attack(model, test_loader, device, atk):
    """Evaluate the model under PGD attack."""
    model.eval()
    total_correct = 0
    total = 0

    for images, targets in test_loader:
        images, targets = images.to(device), targets.to(device)
        adv_images = atk(images, targets)
        outputs = model(adv_images)
        _, preds = torch.max(outputs, 1)
        total_correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    accuracy = total_correct / total
    print(f"Model accuracy under PGD attack: {accuracy * 100:.2f}%")
    return accuracy

class CustomLoss(nn.Module):
    def __init__(self, rho=0.05, q=1):
        """
        Initializes the adCustomLoss module.

        Args:
            rho (float): Weighting factor for the variation term.
            q (float): The norm degree for variation calculation. Supports any q >= 1 and float('inf') for infinity norm.
        """
        super(CustomLoss, self).__init__()
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
# Define the new custom loss function with the regularization term
class oldCustomLoss(nn.Module):
    def __init__(self, C=0.5, N=100, regularization_weight=0.01, use_mcmc=False):
        """
        Initializes the CustomLoss module.

        Args:
            C (float): Constant in the regularization term.
            N (int): Number of samples for Monte Carlo estimation.
            regularization_weight (float): Weighting factor for the regularization term.
            use_mcmc (bool): Whether to use MCMC sampling.
        """
        super(CustomLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.C = C
        self.N = N
        self.regularization_weight = regularization_weight
        self.use_mcmc = use_mcmc
        self.training_mode = True

    def forward(self, outputs, targets, model=None, images=None):
        """
        Computes the custom loss combining cross-entropy and the regularization term.

        Args:
            outputs (torch.Tensor): Model outputs (logits) of shape (batch_size, num_classes).
            targets (torch.Tensor or tuple): True labels or mixup targets.
            model (nn.Module, optional): The neural network model.
            images (torch.Tensor, optional): Input images tensor.

        Returns:
            torch.Tensor: The computed total loss.
        """
        # Compute Cross-Entropy Loss
        if isinstance(targets, tuple):
            y_a, y_b, lam = targets
            ce_loss = lam * self.cross_entropy(outputs, y_a) + (1 - lam) * self.cross_entropy(outputs, y_b)
        else:
            ce_loss = self.cross_entropy(outputs, targets)

        # Compute Regularization Term if in training mode and images are provided
        if self.training_mode and images is not None and model is not None:
            reg_loss = self.compute_regularization(model, ce_loss, images, targets)
            total_loss = ce_loss + self.regularization_weight * reg_loss
        else:
            total_loss = ce_loss

        return total_loss

    def compute_regularization(self, model, loss, images, targets):
        """
        Computes the regularization term based on the provided mathematical expression.

        Args:
            model (nn.Module): The neural network model.
            loss (torch.Tensor): The cross-entropy loss.
            images (torch.Tensor): Input images tensor.
            targets (torch.Tensor): Targets tensor.

        Returns:
            torch.Tensor: The computed regularization term.
        """
        total_regularization = 0.0
        C = self.C
        N = self.N

        # Ensure the model is in evaluation mode
        was_training = model.training
        model.eval()

        # Enable gradients for images
        images = images.clone().detach().requires_grad_(True)

        # Forward pass
        outputs = model(images)
        if isinstance(targets, tuple):
            y_a, y_b, lam = targets
            loss = lam * self.cross_entropy(outputs, y_a) + (1 - lam) * self.cross_entropy(outputs, y_b)
        else:
            loss = self.cross_entropy(outputs, targets)

        # Compute gradient of loss w.r.t. images
        grad_images = torch.autograd.grad(loss, images, create_graph=True)[0]
        grad_images_flat = grad_images.view(grad_images.size(0), -1)  # Shape: (batch_size, num_features)
        n = grad_images_flat.size(1)  # Number of features

        batch_size = grad_images_flat.size(0)
        device = grad_images_flat.device

        # Sample N vectors from the unit ball for each sample in the batch
        if self.use_mcmc:
            # Use MCMC sampling
            B = mcmc_sample_unit_ball(N, batch_size, n, device)
        else:
            # Direct sampling
            # Sample unit vectors
            X = torch.randn(N, batch_size, n, device=device)
            X_norm = X.norm(dim=2, keepdim=True)  # Shape (N, batch_size, 1)
            X_unit = X / X_norm

            # Sample radii
            U = torch.rand(N, batch_size, 1, device=device)
            # Compute samples from the unit ball
            B = X_unit * U.pow(1.0 / n)

        # Compute dot products: Shape (N, batch_size)
        dot_products = torch.einsum('nbf,bf->nb', B, grad_images_flat)

        # Compute s_i: Shape (N, batch_size)
        s = torch.exp(C * dot_products - 2)

        # Compute E_p: Shape (batch_size,)
        E_p = s.mean(dim=0)

        # Avoid division by zero
        E_p = torch.clamp(E_p, min=1e-7)

        # Compute first term: Shape (batch_size,)
        first_term = torch.log(E_p)

        # Compute second term: Shape (batch_size,)
        second_term = (1 / C) * (s / E_p.pow(C)).mean(dim=0)

        # Compute regularization term for each sample
        regularization = first_term + second_term + (1 / C)

        # Average over the batch
        total_regularization = regularization.mean()

        # Restore the model's original training state
        if was_training:
            model.train()

        return total_regularization

    def set_training_mode(self, mode):
        """
        Sets the training mode for the regularization calculation.

        Args:
            mode (bool): If True, regularization is included in the loss. Otherwise, only cross-entropy is used.
        """
        self.training_mode = mode


def train(net, train_loader, optimizer, scheduler, device, criterion):
    """Train for one epoch and compute training loss and accuracy."""
    net.train()
    loss_ema = 0.
    correct = 0
    total = 0
    
    
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        if args.jsd == 0:
            images = images.to(device)
            targets = targets.to(device)
          
            if args.alpha == 0.0:   
                outputs = net(images)
            else:
                outputs, targets_a, targets_b, lam = net(images, targets=targets, jsd=args.jsd,
                                                         mixup_alpha=args.alpha,
                                                         manifold_mixup=args.manifold_mixup,
                                                         add_noise_level=args.add_noise_level,
                                                         mult_noise_level=args.mult_noise_level,
                                                         sparse_level=args.sparse_level)
        
            if args.alpha > 0:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                
                _, preds = torch.max(outputs, 1)
                correct += (preds == targets_a).sum().item() * lam + (preds == targets_b).sum().item() * (1 - lam)
                total += targets.size(0)
            else:
                loss = criterion(outputs, targets)
                _, preds = torch.max(outputs, 1)
                correct += preds.eq(targets).sum().item()
                total += targets.size(0)
        
        
        elif args.jsd == 1:
            images_all = torch.cat(images, 0).to(device)
            targets = targets.to(device)      
          
            if args.alpha == 0.0:   
                logits_all = net(images_all)
            else:
                logits_all, targets_a, targets_b, lam = net(images_all, targets=targets, jsd=args.jsd, 
                                                            mixup_alpha=args.alpha,
                                                            manifold_mixup=args.manifold_mixup,
                                                            add_noise_level=args.add_noise_level,
                                                            mult_noise_level=args.mult_noise_level,
                                                            sparse_level=args.sparse_level)
        
            if args.alpha > 0:
                logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
                loss = mixup_criterion(criterion, logits_clean, targets_a, targets_b, lam)
                
                _, preds = torch.max(logits_clean, 1)
                correct += (preds == targets_a).sum().item() * lam + (preds == targets_b).sum().item() * (1 - lam)
                total += targets.size(0)
            else:
                logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
                loss = criterion(logits_clean, targets)
                _, preds = torch.max(logits_clean, 1)
                correct += preds.eq(targets).sum().item()
                total += targets.size(0)
          
            # JSD Loss
            p_clean, p_aug1, p_aug2 = F.softmax(logits_clean, dim=1), F.softmax(logits_aug1, dim=1), F.softmax(logits_aug2, dim=1)
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                          F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                          F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
        
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
    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
            _, preds = torch.max(logits, 1)
            total_loss += loss.item() * targets.size(0)
            total_correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    
    avg_loss = total_loss / total
    avg_acc = total_correct / total
    return avg_loss, avg_acc

def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

   

    preprocess = transforms.Compose([
            transforms.Resize(32), 
            transforms.ToTensor(), 
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.2860, 0.2860, 0.2860), 
                         (0.3530, 0.3530, 0.3530))
    ])
    test_transform = preprocess
  
   
    train_dataset = datasets.FashionMNIST(root='./data', train=True, transform = test_transform, download= True)
                               #transform=train_transform, download=True)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, 
                              transform=test_transform, download=True)
    num_classes = 10 
    if args.augmix == 1:
            train_dataset = AugMixDataset(train_dataset, preprocess, args.jsd, args)
 

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.train_batch_size,
            shuffle=True, num_workers=8, pin_memory=True) 
    test_loader = torch.utils.data.DataLoader(
          test_dataset, batch_size=args.test_batch_size,
          shuffle=False, num_workers=8, pin_memory=True)

    
    
    
    if args.arch == 'preactresnet18':
        net = preactresnet18(num_classes=num_classes)
    elif args.arch == 'preactwideresnet18':
        net = preactwideresnet18(num_classes=num_classes)
    elif args.arch == 'wideresnet28':
        net = wideresnet28(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")

   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    optimizer = torch.optim.SGD(net.parameters(),
        args.learning_rate, momentum=args.momentum,
        weight_decay=args.decay, nesterov=True)

    
    #net = torch.nn.DataParallel(net).to(device)

    

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(
            step, args.epochs * len(train_loader),
            1, 
            1e-6 / args.learning_rate))
    
    
    criterion = CustomLoss(rho = 0.08, q=1).to(device)
    
    # Training loop
    for epoch in range(args.epochs):
        train_loss_ema, train_acc = train(net, train_loader, optimizer, scheduler, device,criterion)
        test_loss, test_acc = test(net, test_loader, device)
        print(
            'Epoch {0:3d} | Train Loss {1:.4f} |'
            ' Test Accuracy {2:.2f}%'
            .format((epoch + 1), train_loss_ema, 100. * test_acc))

    # Save final model
    DESTINATION_PATH = 'fmnist_models/'
    if not os.path.isdir(DESTINATION_PATH):
        os.mkdir(DESTINATION_PATH)
    model_path = os.path.join(DESTINATION_PATH, f'final_arch_{args.arch}_seed_{args.seed}.pt')
    torch.save(net.state_dict(), model_path)
    print(f"Final model saved to {model_path}")

      

     # Evaluate the model under PGD attack
    print("Evaluating the model under PGD attack...")
    attack1 = torchattacks.PGD(net, eps=4/255, alpha=0.5/255, steps=20, random_start=True)
    pgd_accuracy = evaluate_under_attack(net, test_loader, device ,attack1)
    print(f"Model accuracy under PGD 4/255 attack: {pgd_accuracy * 100:.2f}%")

    attack2 = torchattacks.PGD(net, eps=8/255, alpha=1/255, steps=20, random_start=True)
    pgd_accuracy = evaluate_under_attack(net, test_loader, device ,attack2)
    print(f"Model accuracy under PGD 8/255 attack: {pgd_accuracy * 100:.2f}%")

    attack3 = torchattacks.PGD(net, eps=16/255, alpha=2/255, steps=20, random_start=True)
    pgd_accuracy = evaluate_under_attack(net, test_loader, device, attack3)
    print(f"Model accuracy under PGD 16/255 attack: {pgd_accuracy * 100:.2f}%")
    
if __name__ == '__main__':
    main()
