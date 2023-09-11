CUDA_VISIBLE_DEVICES=0,1,3  python src/ms_pred/autoregr_gen/hyperopt.py \
--seed 5  \
--num-workers 10  \
--max-epochs 200 \
--dataset-name nist20 \
--dataset-labels labels.tsv \
--formula-folder magma_subform_50 \
--split-name hyperopt.tsv  \
 --gpu  \
 --save-dir results/autoregr_hyperopt \
--cpus-per-trial 10 \
--gpus-per-trial 0.5 \
--num-h-samples 50 \
--grace-period 900 \
--max-concurrent 10  \
--embed-adduct \
--use-reverse \
#--tune-checkpoint results/scarf_hyperopt/score_function_2023-02-01_21-09-00/