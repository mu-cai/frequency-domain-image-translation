python main.py --mode train --num_domains 2 --w_hpf 1 \
               --lambda_reg 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 1 \
               --train_img_dir data/celeba_hq_smile/train \
               --val_img_dir data/celeba_hq_smile/val  --batch_size 8 \
               --exp_dir expr_smile/