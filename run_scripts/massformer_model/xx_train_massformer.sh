#mkdir results/massformer_baseline
CUDA_VISIBLE_DEVICES=1 python src/ms_pred/massformer_pred/train.py --seed 5 \
--num-workers 0 --batch-size 16 --max-epochs 200 --dataset-name canopus_train_public \
--dataset-labels labels.tsv --split-name split_1.tsv --learning-rate 0.0006 \
--layers 3 --hidden-size 64 --lr-decay-rate 0.86   --weight-decay 1e-7 \
--dropout 0.0  --gpu  --num-bins 3000 --set-layers 3 \
--loss-fn mse --save-dir results/debug_gnn_reverse --mpnn-type GINE --use-reverse  \
 --pe-embed-k 0 --pool-op avg &#results/gnn_baseline/split_1

#mkdir results/gnn_baseline
CUDA_VISIBLE_DEVICES=2 python src/ms_pred/massformer_pred/train.py --seed 5 \
--num-workers 0 --batch-size 16 --max-epochs 200 --dataset-name canopus_train_public \
--dataset-labels labels.tsv --split-name split_1.tsv --learning-rate 0.0006 \
--layers 3 --hidden-size 64 --lr-decay-rate 0.86   --weight-decay 1e-7 \
--dropout 0.0  --gpu  --num-bins 3000 --set-layers 3 \
--loss-fn mse --save-dir results/debug_gnn_no_reverse --mpnn-type GINE   \
 --pe-embed-k 0 --pool-op avg & #results/gnn_baseline/split_1