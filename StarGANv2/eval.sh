CUDA_VISIBLE_DEVICES=1 \
python main.py --mode eval --num_domains 2 --w_hpf 1 \
--lambda_reg 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 1 \
--resume_iter 100000 \
--train_img_dir data/celeba_hq_smile/train \
--val_img_dir data/celeba_hq_smile/val \
--checkpoint_dir checkpoints/ \
--eval_dir eval/  --exp_dir expr_smile_10w_recon_freq_decay_style_se/