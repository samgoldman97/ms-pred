# Note: An intermediate model must first be trained, used to make predictions, and assignments must be labeled to utilize this model

CUDA_VISIBLE_DEVICES=1,2,3 python src/ms_pred/dag_pred/inten_hyperopt.py  \
    --seed 5 \
    --num-workers 10 \
    --max-epochs 200 \
    --dataset-name nist20 \
    --dataset-labels labels.tsv \
    --split-name hyperopt.tsv \
    --gpu \
    --loss-fn cosine \
    --save-dir results/dag_inten_hyperopt \
    --magma-dag-folder results/dag_nist_hyperopt/hyperopt/preds_train_100_inten \
    --cpus-per-trial 10 \
    --gpus-per-trial 0.5 \
    --num-h-samples 50 \
    --grace-period 900 \
    --max-concurrent 10 \
    --binned-targs \
    --embed-adduct \
    --encode-forms \
    --add-hs \
