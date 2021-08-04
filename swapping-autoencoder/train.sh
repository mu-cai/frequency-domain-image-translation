CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 train.py --batch  4 \
dataset/img_mountains_swapae   --name mountain