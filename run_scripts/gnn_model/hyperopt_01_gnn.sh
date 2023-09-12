mkdir results/gnn_hyperopt
CUDA_VISIBLE_DEVICES=1,2,3 python src/ms_pred/gnn_pred/gnn_hyperopt.py \
--seed 1 \
--num-workers 8 \
--max-epochs 200 \
--dataset-name nist20 \
--dataset-labels labels.tsv \
--split-name hyperopt.tsv \
--gpu  \
--num-bins 15000  \
--save-dir results/gnn_hyperopt_v2  \
--cpus-per-trial 8 \
--gpus-per-trial 0.5  \
--num-h-samples 50 \
--grace-period 900 \
--max-concurrent 10 \
--embed-adduct \
--form-dir-name no_subform \

