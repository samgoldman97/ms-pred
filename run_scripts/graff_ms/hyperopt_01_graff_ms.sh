mkdir results/graff_ms_hyperopt
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/ms_pred/graff_ms/graff_ms_hyperopt.py \
--seed 1 \
--num-workers 5 \
--dataset-name nist20 \
--dataset-labels labels.tsv \
--split-name hyperopt.tsv \
--gpu  \
--num-bins 15000  \
--save-dir results/graff_ms_hyperopt  \
--cpus-per-trial 5 \
--gpus-per-trial 0.5  \
--num-h-samples 50 \
--grace-period 1000 \
--max-concurrent 10 \
--embed-adduct \
--form-dir-name magma_subform_50_with_raw \
--max-epochs 200
