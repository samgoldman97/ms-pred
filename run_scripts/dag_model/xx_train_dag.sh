#
#CUDA_VISIBLE_DEVICES=0 python src/ms_pred/dag_pred/train_gen.py 
#--seed 5 \
#--num-workers 0 \
#--batch-size 1 \
#--max-epochs 100 \
#--dataset-name canopus_train_public \
#--dataset-labels labels.tsv \
#--split-name split_1.tsv \
#--learning-rate 7e-4 \
#--layers 4 \
#--hidden-size 128  \
#--mpnn-type PNA \
#--set-layers 3 \
#--dropout 0 \
#--pe-embed-k 15 \
#--debug  \
#--gpu \
#--pool-op attn

CUDA_VISIBLE_DEVICES=0 python src/ms_pred/dag_pred/train_gen.py  \
--seed 5 \
--num-workers 8 \
--batch-size 16 \
--max-epochs 100 \
--dataset-name canopus_train_public \
--dataset-labels labels.tsv \
--split-name split_1.tsv \
--learning-rate 0.0009 \
--lr-decay-rate 0.8579 \
--layers 5 \
--hidden-size 128  \
--inject-early \
--mpnn-type GGNN  \
--set-layers 2 \
--dropout 0 \
--pe-embed-k 20 \
--gpu \
--pool-op avg \
--debug-overfit \
--root-encode fp \
--save-dir results/debug_overfit_dag_gen \
--weight-decay 1e-06 \
#--debug  \
