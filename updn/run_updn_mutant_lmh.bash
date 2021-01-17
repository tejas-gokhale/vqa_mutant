 CUDA_VISIBLE_DEVICES=3 nohup python main_mutant.py \
 --output saved_models/updn_mutant_lmh \
 --seed 0 \
 --eval_each_epoch > updn_mutant_lmh.out &
