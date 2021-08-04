CUDA_VISIBLE_DEVICES=1 \
python eval_script.py --ckpt_freq checkpoint/mountain.pt \
--save_name mountains_show  --folder_test img_mountains_val --comp_fid


CUDA_VISIBLE_DEVICES=1 \
python eval_script.py --ckpt_freq checkpoint/mountain.pt \
--save_name mountains_sample --test_specific_pairs