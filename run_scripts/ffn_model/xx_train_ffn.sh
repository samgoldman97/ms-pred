#mkdir results/ffn_baseline
CUDA_VISIBLE_DEVICES=2 python src/ms_pred/ffn_pred/train.py --seed 5 \
--num-workers 0 --batch-size 64 --max-epochs 200 --dataset-name canopus_train_public \
--dataset-labels labels.tsv --split-name split_1.tsv --learning-rate 4e-4 \
--layers 1 --hidden-size 64   --lr-decay-rate 0.77 \
--dropout 0.2  --gpu  --use-reverse --num-bins 3000 \
--loss-fn mse  --save-dir results/debug_ffn_reverse2/ --weight-decay 1e-7

#CUDA_VISIBLE_DEVICES=2 python src/ms_pred/ffn_pred/train.py --seed 5 \
#--num-workers 0 --batch-size 64 --max-epochs 200 --dataset-name canopus_train_public \
#--dataset-labels labels.tsv --split-name split_1.tsv --learning-rate 4e-4 \
#--layers 1 --hidden-size 64   --lr-decay-rate 0.77 \
#--dropout 0.2  --gpu  --num-bins 3000 \
#--loss-fn mse  --save-dir results/debug_ffn_no_reverse2/ --weight-decay 1e-7 &
