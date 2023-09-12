mkdir results/ffn_hyperopt
CUDA_VISIBLE_DEVICES=1,2,3 python src/ms_pred/ffn_pred/ffn_hyperopt.py \
        --seed 1 \
        --num-workers 8 \
        --max-epochs 200 \
        --dataset-name nist20 \
        --dataset-labels labels.tsv \
        --form-dir-name no_subform \
        --loss-fn cosine\
        --split-name hyperopt.tsv \
        --gpu  \
        --num-bins 15000 \
        --save-dir results/ffn_hyperopt \
        --cpus-per-trial 8 \
        --gpus-per-trial 1 \
        --num-h-samples 50 \
        --grace-period 900 \
        --max-concurrent 10 \
        --embed-adduct \
