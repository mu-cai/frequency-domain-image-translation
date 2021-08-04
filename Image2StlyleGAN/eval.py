
import cv2
import numpy as np
import pandas as pd
import argparse
import os
import pdb

from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image
import math


def psnr(img1, img2, const=1):
    mse = np.mean( (img1/const - img2/const) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def mse(img1, img2, const=255):
    mse = np.mean( (img1/const - img2/const) ** 2 )
    return mse
    
def mae(img1, img2, const=255): 
    mae = np.mean( abs(img1/const - img2/const)  )
    return mae   


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def resize_short(image, imgsize=256):
    (w, h) = image.size
    if w <= h:
        h = int(h * imgsize / w)
        w = imgsize
    else:
        w = int(w * imgsize / h)
        h = imgsize
    return image.resize((w, h), Image.ANTIALIAS)

parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
parser.add_argument('--resolution', default=1024, type=int)
parser.add_argument('--weight_file', default="weight_files/pytorch/karras2019stylegan-ffhq-1024x1024.pt", type=str)
parser.add_argument('--image_folder_name', default="encode_freq_")
parser.add_argument('--image_dir', default="save_image_1024/")
parser.add_argument('--test_iter', default=4990, type=int)

imgsize = 1024
save_dir_resize = 'source_image_resize/'

def center_crop(img):
    """Returns center cropped image

    Args:Image Scaling
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped from center
    """
    width, height = img.shape[1], img.shape[0]
    dim = min(width, height)
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 =  int(dim/2) ,  int(dim/2) 
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img

def ssim(img_org, img1):
    return compare_ssim(img_org, img1,multichannel=True)

args = parser.parse_args()
dir_source= 'source_image/' 
list_name = os.listdir(dir_source)
result_mse = []
result_mae = []
result_psnr = []
result_ssim = []
for name in list_name:

    name_before = name.split(".")[0]
    name_after = 'png'
    freq_output= args.image_dir +args.image_folder_name + name_before + f'/{args.test_iter}.' +name_after
    img_org_dir=save_dir_resize + name_before + '.png'
    
    img_org = cv2.imread(img_org_dir)   

    img1 = cv2.imread(freq_output)
    print(img_org_dir, freq_output)

    result_mse.append(mse(img_org, img1))
    result_mae.append(mae(img_org, img1))
    result_ssim.append(ssim(img_org, img1))
    result_psnr.append(psnr(img_org, img1))

print(np.mean(result_mse), np.mean(result_mae), np.mean(result_psnr), np.mean(result_ssim)) 