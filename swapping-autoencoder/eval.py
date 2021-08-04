import argparse

import numpy as np
import torch
from PIL import Image
from torchvision import utils
import os
import model


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

def save_image(img, index, save_dir, dir = '/source/', save_ending =  '.png'):
    utils.save_image(
                img.cpu(),
                save_dir + dir+ str(index) + save_ending,
                normalize=True,
                range=(-1, 1)
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default="checkpoint/mountain.pt")
    parser.add_argument("--ckpt_freq", type=str, default="")
    parser.add_argument('--folder', type=str, default='img_mountains_val')
    parser.add_argument('--imgsize', type=int, default=256)
    parser.add_argument("--test_specific_pairs", action="store_true")
    parser.add_argument('--image_pair_dir', type=str, default='mountain_val/')

    args = parser.parse_args()

    if not os.path.exists(args.name):
        os.makedirs(args.name)

    imgsize=args.imgsize
    large_dir='test_sample/'

    imgs = []

    if args.test_specific_pairs:
        image_dir = args.image_pair_dir
        source_dir = [ image_dir + 'source/' + image_name for image_name in os.listdir( image_dir + 'source') ]
        ref_dir = [ image_dir + 'ref/' + image_name for image_name in os.listdir( image_dir + 'ref') ]
        image_list =  source_dir + ref_dir # os.listdir( image_dir + 'ref')
        args.folder = ''
    else:
        args.files=os.listdir(args.folder)
        image_list = args.files[: len(args.files) // 2 * 2]
    for imgpath in image_list:
        img = Image.open(os.path.join(args.folder, imgpath) ).convert("RGB")
        img =resize_short(img, imgsize)
        img =crop_center(img,imgsize,imgsize)
        img_a = (
            torch.from_numpy(np.array(img))
            .to(torch.float32)
            .div(255)
            .add_(-0.5)
            .mul_(2)
            .permute(2, 0, 1)
        )
        imgs.append(img_a)
    print('Image Num: ', len(imgs))

    imgs = torch.stack(imgs, 0).cuda()
    img1_whole, img2_whole = imgs.chunk(2, dim=0)

    ckpt_freq = torch.load(args.ckpt_freq, map_location=lambda storage, loc: storage)
    ckpt_args_freq = ckpt_freq["args"]
    ckpt_args = ckpt_freq["args"]

    imgsize = ckpt_args.size

    enc_freq = model.Encoder(ckpt_args_freq.channel).cuda()
    gen_freq = model.Generator(ckpt_args_freq.channel).cuda()
    enc_freq.load_state_dict(ckpt_freq["e_ema"])
    gen_freq.load_state_dict(ckpt_freq["g_ema"])
    enc_freq.eval()
    gen_freq.eval()

    for index in range(img1_whole.size(0)):
        with torch.no_grad():
            real_img_freq1= img1_whole[index].unsqueeze(0)
            real_img_freq2 = img2_whole[index].unsqueeze(0)
            struct1, texture1 = enc_freq(real_img_freq1) 
            struct2, texture2 = enc_freq(real_img_freq2)
            out12_freq = gen_freq(struct1, texture2)
            out21_freq = gen_freq(struct2, texture1)
            
            dirs=[large_dir,large_dir+args.name, large_dir+args.name+'/freq/', large_dir+args.name+'/source/' ]

            save_dir = large_dir+args.name
            for dir_one in dirs:
                if not os.path.exists(dir_one):
                    os.makedirs(dir_one)
            save_image(real_img_freq1, index, save_dir, dir = '/source/', save_ending =  '.png')
            save_image(out12_freq, index, save_dir, dir = '/freq/', save_ending =  '.png')

            save_image(real_img_freq2, index, save_dir, dir = '/source/', save_ending =  '_inv.png')
            save_image(out21_freq, index, save_dir, dir = '/freq/', save_ending =  '_inv.png')

    ################################################
