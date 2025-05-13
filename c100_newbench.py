import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18,resnet34,resnet50,resnet101, ResNet18_Weights,ResNet34_Weights,ResNet50_Weights,ResNet101_Weights
from torch.utils.data import DataLoader, Dataset
from PIL import Image as PILImage
from io import BytesIO
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import ctypes
import cv2
import skimage as sk
from skimage.filters import gaussian
from scipy.ndimage import zoom as scizoom
from scipy.ndimage import map_coordinates
import random
import time

# 设置随机种子以确保可重复性
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# 忽略警告信息
import warnings
warnings.simplefilter("ignore", UserWarning)

# 定义辅助函数
def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]

# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)

def plasma_fractal(mapsize=32, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


# 定义腐蚀函数
def gaussian_noise(x, severity=1):
    c = [0.0035, 0.055, .075, .09, .12][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    c = [1450, 162, 95, 50, 30][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [.00055, .008, .016, .028, .036][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def defocus_blur(x, severity=1):
    c = [(0.35, 0.4), (0.4, 0.6), (0.44, 0.85), (0.9, 1.9), (1.55, 2.2)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x32x32 -> 32x32x3

    return np.clip(channels, 0, 1) * 255


def glass_blur1(x, severity=1):
    # sigma, max_delta, iterations
    c = [(0.0000001,1,1), (0.5,1,2), (1,1,1), (1.25,1,2), (1.5,1,2)][severity - 1]

    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0]) * 255)

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(32 - c[1], c[1], -1):
            for w in range(32 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255., sigma=c[0]), 0, 1) * 255


def glass_blur(x, severity=1):

    severity_settings = [
        (0.001, 1, 1, 0.03),  # Severity 1: Minimal blur and swapping
        (0.4,      1, 1, 0.16),  # Severity 2: Slight increase in blur and swapping
        (0.4,        1, 1, 0.3),  # Severity 3: Moderate blur and swapping
        (0.3,       1, 2, 0.21),  # Severity 4: Increased iterations and swapping
        (0.3,        1, 2, 0.29)   # Severity 5: Maximum blur and swapping
    ]
    
    if severity < 1 or severity > 5:
        raise ValueError("Severity must be an integer between 1 and 5.")
    
    sigma, max_delta, iterations, swap_ratio = severity_settings[severity - 1]

    # Apply initial Gaussian blur
    x_blur = gaussian(np.array(x) / 255.0, sigma=sigma)
    x_blur = np.uint8(x_blur * 255)

    # Locally shuffle pixels
    height, width = x_blur.shape[:2]
    
    for i in range(iterations):
        # Calculate the number of swaps based on swap_ratio
        total_pixels = (height - 2 * max_delta) * (width - 2 * max_delta)
        num_swaps = int(total_pixels * swap_ratio)
        
        for _ in range(num_swaps):
            # Randomly select a pixel within the valid range to avoid boundary issues
            h = np.random.randint(max_delta, height - max_delta)
            w = np.random.randint(max_delta, width - max_delta)
            
            # Randomly determine the displacement within the specified delta
            dx, dy = np.random.randint(-max_delta, max_delta + 1, size=2)
            h_prime, w_prime = h + dy, w + dx
            
            # Ensure the new coordinates are within image boundaries
            if (0 <= h_prime < height) and (0 <= w_prime < width):
                # Swap the pixels
                if x_blur.ndim == 3:
                    # For color images
                    temp = x_blur[h, w].copy()
                    x_blur[h, w] = x_blur[h_prime, w_prime]
                    x_blur[h_prime, w_prime] = temp
                else:
                    # For grayscale images
                    x_blur[h, w], x_blur[h_prime, w_prime] = x_blur[h_prime, w_prime], x_blur[h, w]

    # Apply final Gaussian blur
    final_blur = gaussian(x_blur / 255.0, sigma=sigma)
    final_blur = np.clip(final_blur, 0, 1) * 255
    return final_blur.astype(np.uint8)



def motion_blur(x, severity=1):
    c = [(1,0.6), (6,1.2), (6,2), (9,2.9), (7,5)][severity - 1]

    output = BytesIO()
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                     cv2.IMREAD_UNCHANGED)

    if x.shape != (32, 32):
        return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    else:  # greyscale to RGB
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)

def zoom_blur(x, severity=1):
    c = [np.arange(1, 1.04, 0.01), np.arange(1, 1.13, 0.01), np.arange(1, 1.22, 0.01),
         np.arange(1, 1.36, 0.01), np.arange(1, 1.70, 0.01)][severity - 1]

    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255

def snow(x, severity=1):
    c = [(0.001,0.07,1.0,0.1,5,5,0.95),
         (0.009,0.1,1.0,0.1,9,9,0.85),
         (0.009,0.16,1.25,0.2,10,20,0.65),
         (0.015,0.33,1.75,0.3,12,20,0.65),
         (0.025,0.47,1.75,0.4,12,18,0.62)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

    snow_layer = cv2.imdecode(np.frombuffer(snow_layer.make_blob(), np.uint8),
                              cv2.IMREAD_UNCHANGED) / 255.
    snow_layer = snow_layer[..., np.newaxis]

    x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(32, 32, 1) * 1.5 + 0.5)
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255

def frost(x, severity=1):
    c = [(1, 0.16), (1.1, 0.3), (1, 0.4), (0.8, 0.53), (0.9, 0.65)][severity - 1]
    idx = np.random.randint(5)
    filename = ['./frost1.png', './frost2.png', './frost3.png', './frost4.jpg', './frost5.jpg', './frost6.jpg'][idx]
    frost = cv2.imread(filename)
    frost = cv2.resize(frost, (0, 0), fx=0.2, fy=0.2)
    # randomly crop and convert to rgb
    x_start, y_start = np.random.randint(0, frost.shape[0] - 32), np.random.randint(0, frost.shape[1] - 32)
    frost = frost[x_start:x_start + 32, y_start:y_start + 32][..., [2, 1, 0]]

    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)

def fog(x, severity=1):
    c = [(.3,3.5), (.5,2.75), (.9,2.2), (1.15,2), (1.45,1.85)][severity - 1]

    x = np.array(x) / 255.
    max_val = x.max()
    x += c[0] * plasma_fractal(wibbledecay=c[1])[:32, :32][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255

def brightness(x, severity=1):
    c = [.12, .24, .36, .49, .62][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255

def contrast(x, severity=1):
    c = [.81, .59, .45, .39, 0.3][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def elastic_transform(image, severity=1):
    IMSIZE = 32
    c = [(IMSIZE*0, IMSIZE*0, IMSIZE*0.01),
         (IMSIZE*0.2, IMSIZE*0.2, IMSIZE*0.075),
         (IMSIZE*0.2, IMSIZE*0.2, IMSIZE*0.142),
         (IMSIZE*0.2, IMSIZE*0.2, IMSIZE*0.19),
         (IMSIZE*0.2, IMSIZE*0.2, IMSIZE*0.29)][severity - 1]

    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255

def pixelate(x, severity=1):
    c = [0.94, 0.67, 0.52, 0.45, 0.35][severity - 1]

    x = x.resize((int(32 * c), int(32 * c)), PILImage.BOX)
    x = x.resize((32, 32), PILImage.BOX)

    return x

def jpeg_compression(x, severity=1):
    c = [90, 35, 15, 8, 5][severity - 1]

    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = PILImage.open(output)

    return x

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409),  # CIFAR-100 的均值
                         (0.2673, 0.2564, 0.2762)), # CIFAR-100 的标准差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

batch_size = 128

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                         download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=8)

# 定义模型、损失函数和优化器
#net = resnet18(weights=ResNet18_Weights.DEFAULT)
#net = resnet34(weights=ResNet34_Weights.DEFAULT)
net = resnet50(weights=ResNet50_Weights.DEFAULT)
#net = resnet101(weights=ResNet101_Weights.DEFAULT)
net.fc = nn.Linear(net.fc.in_features, 100)

# 如果有 GPU，可将模型移动到 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# 训练模型
num_epochs = 200 # 根据需要调整训练轮数

for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if batch_idx % 100 == 99:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(trainloader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0
    
    scheduler.step()
    end_time = time.time()
    print(f'Epoch [{epoch+1}/{num_epochs}] completed in {end_time - start_time:.2f} seconds.')

# 保存模型
torch.save(net.state_dict(), 'cifar10_resnet18.pth')

# 定义评估函数
def evaluate_on_corrupted_dataset(corruption_func, severity):
    net.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in testloader:
            # 将输入转换为 PIL 图像
            inputs_pil = [transforms.ToPILImage()(inp.cpu()) for inp in inputs]
            # 对每张图像应用腐蚀
            inputs_corrupted = []
            for img_pil in inputs_pil:
                corrupted_img = corruption_func(img_pil, severity=severity)
                if isinstance(corrupted_img, PILImage.Image):
                    corrupted_img = corrupted_img
                else:
                    corrupted_img = PILImage.fromarray(np.uint8(corrupted_img))
                inputs_corrupted.append(transforms.ToTensor()(corrupted_img))
            # 将腐蚀后的图像转换为张量
            inputs_corrupted = torch.stack(inputs_corrupted).to(device)
            inputs_corrupted = transforms.Normalize((0.5071, 0.4865, 0.4409),  # CIFAR-100 的均值
                                                    (0.2673, 0.2564, 0.2762))(inputs_corrupted)
            targets = targets.to(device)
            outputs = net(inputs_corrupted)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    avg_loss = total_loss / total
    return acc, avg_loss

# 定义腐蚀类型
corruption_types = [
   'gaussian_noise',
   'shot_noise',
   'impulse_noise',
   'defocus_blur',
   'glass_blur',
   'motion_blur',
   'zoom_blur',
    'snow',
   'frost',
   'fog',
    'brightness',
    'contrast',
   'elastic_transform',
    'pixelate',
    'jpeg_compression',
]

# 映射腐蚀类型到函数
corruption_functions = {
    'gaussian_noise': gaussian_noise,
    'shot_noise': shot_noise,
    'impulse_noise': impulse_noise,
    'defocus_blur': defocus_blur,
    'glass_blur': glass_blur,
    'motion_blur': motion_blur,
    'zoom_blur': zoom_blur,
    'snow': snow,
    'frost': frost,
    'fog': fog,
    'brightness': brightness,
    'contrast': contrast,
    'elastic_transform': elastic_transform,
    'pixelate': pixelate,
    'jpeg_compression': jpeg_compression,
}

# 对每种腐蚀类型和严重程度进行评估
for corruption_name in corruption_types:
    print(f"\n{corruption_name}:")
    for severity in range(1, 6):
        acc, avg_loss = evaluate_on_corrupted_dataset(corruption_functions[corruption_name], severity)
        print(f"  Severity {severity}: Accuracy = {acc:.2f}%, Loss = {avg_loss:.4f}")
