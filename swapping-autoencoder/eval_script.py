import os
import argparse

from metrics.fid import calculate_fid_given_paths

parser = argparse.ArgumentParser()

parser.add_argument('--GPU_ID', type=str, default='1', help='GPU ID')
parser.add_argument('--ckpt_freq', type=str, default='checkpoint/mountain.pt', help='ckpt name')
parser.add_argument('--save_name', type=str, default='church', help='svae place')
parser.add_argument('--img_size', type=int, default=256, help='image resolution')
parser.add_argument('--batch_size', type=int, default=64, help='batch size to use')
parser.add_argument('--folder_test', type=str, default='church_val_image', help='ckpt name')
parser.add_argument("--comp_fid", action="store_true")
parser.add_argument("--test_specific_pairs", action="store_true")
args = parser.parse_args()


path1='test_sample/'+args.save_name+'/source/'
path2='test_sample/'+args.save_name+'/freq/'

if args.test_specific_pairs:
    test_specific_pairs = ' --test_specific_pairs'
else:
    test_specific_pairs = ''


os.system('CUDA_VISIBLE_DEVICES='+ args.GPU_ID+' python eval.py  --ckpt_freq '+args.ckpt_freq+' --name '+args.save_name+' --folder '+ args.folder_test + test_specific_pairs) 
if args.comp_fid:
    fid_value = calculate_fid_given_paths((path1,path2), args.img_size, args.batch_size)
    print('FID value: ' + str(fid_value))