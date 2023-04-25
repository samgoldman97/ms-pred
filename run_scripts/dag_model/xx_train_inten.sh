#CUDA_VISIBLE_DEVICES=1 python -m pdb src/ms_pred/dag_pred/train_inten.py  \
#--seed 5 \
#--num-workers 0 \
#--batch-size 4 \
#--max-epochs 100 \
#--dataset-name canopus_train_public \
#--dataset-labels labels.tsv \
#--split-name split_1.tsv \
#--learning-rate 3e-4 \
#--layers 2 \
#--hidden-size 64 \
#--mpnn-type GGNN \
#--gpu \
#--set-layers 2 \
#--dropout 0 \
#--debug \
#--pe-embed-k 15 \
#--frag-attn-layers 2 \
#--loss-fn mse \
#--save results/debug_inten  \
#--magma-dag-folder results/dag_gen_gine/split_1/preds_train_inten/ \
#--root-encode fp \
##--inject-early  \

CUDA_VISIBLE_DEVICES=2 python src/ms_pred/dag_pred/train_inten.py  \
--seed 5 \
--num-workers 10 \
--batch-size 16 \
--max-epochs 200 \
--dataset-name canopus_train_public \
--dataset-labels labels.tsv \
--split-name split_1.tsv \
--learning-rate 3e-4 \
--lr-decay-rate 0.85 \
--layers 5 \
--hidden-size 128 \
--mpnn-type GINE \
--gpu \
--set-layers 0 \
--dropout 0.1 \
--pe-embed-k 15 \
--frag-attn-layers 2 \
--loss-fn mse \
--pool-op avg \
--save results/dag_inten_gine_persistent  \
--magma-dag-folder results/dag_gen_gine/split_1/preds_train_inten/ \
--root-encode gnn \
--inject-early  \
--weight-decay 1.0e-6 \
