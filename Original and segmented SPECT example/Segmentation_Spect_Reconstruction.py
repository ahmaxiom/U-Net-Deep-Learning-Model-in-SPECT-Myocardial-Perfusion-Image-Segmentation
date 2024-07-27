import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from PIL import Image



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
def predict_image_mask_miou(model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device); image=image.to(device)
    mask = mask.to(device)
    with torch.no_grad():

        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        output = model(image)
        score = mIoU(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, score
def predict_image_mask_pixel(model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device); image=image.to(device)
    mask = mask.to(device)
    with torch.no_grad():

        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        output = model(image)
        acc = pixel_accuracy(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, acc
def miou_score(model, test_set):
    score_iou = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, score = predict_image_mask_miou(model, img, mask)
        score_iou.append(score)
    return score_iou
def pixel_acc(model, test_set):
    accuracy = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, acc = predict_image_mask_pixel(model, img, mask)
        accuracy.append(acc)
    return accuracy
def predict_image_mask_dice(model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device); image=image.to(device)
    mask = mask.to(device)
    with torch.no_grad():

        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        output = model(image)
        score = dice_score(output, mask)  # Compute Dice score
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, score

def dice_score_model(model, test_set):
    score_dice = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, score = predict_image_mask_dice(model, img, mask)  # Use the new function
        score_dice.append(score)
    return score_dice
def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3,padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )
    return conv

def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta= tensor_size - target_size
    delta = delta//2
    return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]

class UNet(nn.Module):
    def __init__(self, num_channels, num_classes,retain_dim=True):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(num_channels, out_c=64)
        self.down_conv_2 = double_conv(in_c=64, out_c=128)
        self.down_conv_3 = double_conv(in_c=128, out_c=256)

        self.up_trans_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2,stride=2)
        self.up_conv_1 = double_conv(256, 128)
        self.up_trans_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2,stride=2)
        self.up_conv_2 = double_conv(128, 64)

        self.out = nn.Conv2d(
            in_channels=64,
            out_channels = num_classes, # Number of objects to segment
            kernel_size=1,
        )
        self.retain_dim = retain_dim

    def forward(self, image,out_size=(64, 64)):
        # encoder part
        x1 = self.down_conv_1(image)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)

        # decoder part
        x = self.up_trans_1(x5)
        y = crop_img(x3, x)
        x = self.up_conv_1(torch.cat([x, y], 1))

        x = self.up_trans_2(x)
        y = crop_img(x1, x)
        x = self.up_conv_2(torch.cat([x, y], 1))

        x = self.out(x)
        if self.retain_dim:
            x = F.interpolate(x, out_size)
        x = F.softmax(x, dim=1)
        return x

##########################Load Our UNet trained model on SPECT images####
# Specify the path to your saved model
model_path = "Unet-Mobilenet_v2_mIoU-0.890.pt"

# Load the model onto the CPU
model = torch.load(model_path, map_location=torch.device('cpu'))

# Set the model in evaluation mode
model.eval()
print(model)

########################################################################
def predict_image_mask(model, image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device); image=image.to(device)
    with torch.no_grad():

        image = image.unsqueeze(0)

        output = model(image)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torchvision.transforms.functional import normalize
from torchvision import transforms
import re
# Step 1: Load the images
folder_path = 'C:/Users/admin/Desktop/Spect Final Segmentation Evalaution code/source'
image_files = os.listdir(folder_path)
image_files.sort(key=lambda f: int(re.findall(r'DS_(\d+)', f)[0]))


images = [Image.open(os.path.join(folder_path, image_file)).convert("RGB") for image_file in image_files]

# Now, each image should have three channels and match the input dimensions expected by your model.
pred_masks = [predict_image_mask(model, image) for image in images]


# Ensure the mask is on the same device as the image tensor
#pred_mask = pred_mask.to(image_tensor.device)

# Apply the mask to the image tensor
#masked_image = image_tensor * pred_mask.unsqueeze(0)

# Step 3: Save the predicted masks
mask_folder_path = 'C:/Users/admin/Desktop/Spect Final Segmentation Evalaution code/output/'
os.makedirs(mask_folder_path, exist_ok=True)
from scipy import ndimage

for i, pred_mask in enumerate(pred_masks):
    # Convert the mask to uint8
    image_tensor = transforms.ToTensor()(images[i])
    pred_mask = pred_mask.to(image_tensor.device)

    # If the mask is one of the first four, replace it with the fifth mask
    if i < 4:
        pred_mask = pred_masks[4].clone()
        # Shift the mask one pixel to the left
        #pred_mask = pred_masks[main_mask_index].clone()
        # Shift the mask to the right by the difference between the current index and the main mask index
        shift = 4-i
        pred_mask = torch.roll(pred_mask, shifts=shift, dims=1)
        
    if 54 <= i <= 59:
        pred_mask = pred_masks[53].clone()
        shift = i - 53
        pred_mask = torch.roll(pred_mask, shifts=-shift, dims=1)
    if i == 60:
        pred_mask = pred_masks[63].clone()
        pred_mask = torch.roll(pred_mask, shifts=+3, dims=1)
    if i == 61:
        pred_mask = pred_masks[63].clone()
        pred_mask = torch.roll(pred_mask, shifts=+2, dims=1)
    if i == 62:
        pred_mask = pred_masks[63].clone()
        pred_mask = torch.roll(pred_mask, shifts=+1, dims=1)
    if i == 65:
        pred_mask = pred_masks[64].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    if i == 117:
        pred_mask = pred_masks[116].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    if i == 118:
        pred_mask = pred_masks[116].clone()
        pred_mask = torch.roll(pred_mask, shifts=-2, dims=1)
    if i == 119:
        pred_mask = pred_masks[116].clone()
        pred_mask = torch.roll(pred_mask, shifts=-3, dims=1)
    if i== 120:
        pred_mask = pred_masks[125].clone()
        pred_mask = torch.roll(pred_mask, shifts=+5, dims=1)
    if i== 121:
        pred_mask = pred_masks[125].clone()
        pred_mask = torch.roll(pred_mask, shifts=+2, dims=1)
    if i== 122:
        pred_mask = pred_masks[125].clone()
        pred_mask = torch.roll(pred_mask, shifts=+3, dims=1)
    if i==123:
        pred_mask = pred_masks[125].clone()
        pred_mask = torch.roll(pred_mask, shifts=+2, dims=1)
    if i== 124:
        pred_mask = pred_masks[125].clone()
        pred_mask = torch.roll(pred_mask, shifts=+1, dims=1)
    if i == 135:
        pred_mask = pred_masks[134].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    if i == 141:
        pred_mask = pred_masks[140].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    if i == 143:
        pred_mask = pred_masks[142].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    if i == 144:
        pred_mask = pred_masks[142].clone()
        pred_mask = torch.roll(pred_mask, shifts=-2, dims=1)
    if i == 147:
        pred_mask = pred_masks[146].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    if i == 148:
        pred_mask = pred_masks[146].clone()
        pred_mask = torch.roll(pred_mask, shifts=-2, dims=1)
    if i == 182:
        pred_mask = pred_masks[181].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    if i == 183:
        pred_mask = pred_masks[181].clone()
        pred_mask = torch.roll(pred_mask, shifts=-2, dims=1)
    if i == 194:
        pred_mask = pred_masks[193].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    if i == 195:
        pred_mask = pred_masks[193].clone()
        pred_mask = torch.roll(pred_mask, shifts=-2, dims=1)
    if i == 201:
        pred_mask = pred_masks[200].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    if i == 227:
        pred_mask = pred_masks[226].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    if i == 233:
        pred_mask = pred_masks[232].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    if i == 240:
        pred_mask = pred_masks[241].clone()
        pred_mask = torch.roll(pred_mask, shifts=+1, dims=1)
    if i == 296:
        pred_mask = pred_masks[295].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    if i == 297:
        pred_mask = pred_masks[295].clone()
        pred_mask = torch.roll(pred_mask, shifts=-2, dims=1)
    if i == 300:
        pred_mask = pred_masks[301].clone()
        pred_mask = torch.roll(pred_mask, shifts=+1, dims=1)
    if i == 303:
        pred_mask = pred_masks[302].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    if i == 304:
        pred_mask = pred_masks[302].clone()
        pred_mask = torch.roll(pred_mask, shifts=-2, dims=1)
    if i == 323:
        pred_mask = pred_masks[322].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    if i == 324:
        pred_mask = pred_masks[322].clone()
        pred_mask = torch.roll(pred_mask, shifts=-2, dims=1)
    if i == 357:
        pred_mask = pred_masks[356].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    if i == 361:
        pred_mask = pred_masks[362].clone()
        pred_mask = torch.roll(pred_mask, shifts=+1, dims=1)
    if i == 360:
        pred_mask = pred_masks[362].clone()
        pred_mask = torch.roll(pred_mask, shifts=+2, dims=1)
    if i == 389:
        pred_mask = pred_masks[388].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    if i == 407:
        pred_mask = pred_masks[406].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    if i == 411:
        pred_mask = pred_masks[410].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    if i == 416:
        pred_mask = pred_masks[415].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    if i == 417:
        pred_mask = pred_masks[415].clone()
        pred_mask = torch.roll(pred_mask, shifts=-2, dims=1)
    if i == 418:
        pred_mask = pred_masks[415].clone()
        pred_mask = torch.roll(pred_mask, shifts=-3, dims=1)
    if i == 419:
        pred_mask = pred_masks[415].clone()
        pred_mask = torch.roll(pred_mask, shifts=-4, dims=1)
    if i == 420:
        pred_mask = pred_masks[423].clone()
        pred_mask = torch.roll(pred_mask, shifts=+3, dims=1)
    if i == 421:
        pred_mask = pred_masks[423].clone()
        pred_mask = torch.roll(pred_mask, shifts=+2, dims=1)
    if i == 422:
        pred_mask = pred_masks[423].clone()
        pred_mask = torch.roll(pred_mask, shifts=+1, dims=1)
    if i == 424:
        pred_mask = pred_masks[423].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    if i == 432:
        pred_mask = pred_masks[431].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    if i == 435:
        pred_mask = pred_masks[434].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    if i == 436:
        pred_mask = pred_masks[434].clone()
        pred_mask = torch.roll(pred_mask, shifts=-2, dims=1)
    if i == 445:
        pred_mask = pred_masks[444].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    if i == 446:
        pred_mask = pred_masks[444].clone()
        pred_mask = torch.roll(pred_mask, shifts=-2, dims=1)
    if i == 447:
        pred_mask = pred_masks[444].clone()
        pred_mask = torch.roll(pred_mask, shifts=-3, dims=1)
    if i == 465:
        pred_mask = pred_masks[464].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    if i == 467:
        pred_mask = pred_masks[466].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    if i == 472:
        pred_mask = pred_masks[471].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    if i == 476:
        pred_mask = pred_masks[475].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    if i == 477:
        pred_mask = pred_masks[475].clone()
        pred_mask = torch.roll(pred_mask, shifts=-2, dims=1)
    if i == 479:
        pred_mask = pred_masks[478].clone()
        pred_mask = torch.roll(pred_mask, shifts=-1, dims=1)
    masked_image = image_tensor * pred_mask.unsqueeze(0)


    # Convert the tensor to a numpy array
    masked_image_np = masked_image.permute(1, 2, 0).cpu().detach().numpy()

    # Label each connected component (region) in the mask
    labeled_mask, num_labels = ndimage.label(masked_image_np)

    # Measure the size of each region and find the largest one
    sizes = ndimage.sum(masked_image_np, labeled_mask, range(num_labels + 1))
    mask_size = sizes < max(sizes)

    # Remove small regions
    remove_pixel = mask_size[labeled_mask]
    masked_image_np[remove_pixel] = 0

    # Save the image
    Image.fromarray((masked_image_np * 255).astype(np.uint8)).save(os.path.join(mask_folder_path, f'mask_{i}.png'))



