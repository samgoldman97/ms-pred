split_name_no_ext="canopus_hplus_100_0"
split_name="${split_name_no_ext}.tsv"
dataset_name="mist_augment_canopus_train"
save_dir="results/${dataset_name}_train_inten/$split_name_no_ext"
inten_dir="results/${dataset_name}_train_gen/$split_name_no_ext/preds_train_100_inten/"
CUDA_VISIBLE_DEVICES=0,1,2 python src/ms_pred/dag_pred/train_inten.py \
    --seed 1 \
    --num-workers 16 \
    --batch-size 32 \
    --max-epochs 200 \
    --dataset-name $dataset_name \
    --split-name $split_name \
    --learning-rate 0.000736 \
    --lr-decay-rate 0.825 \
    --dropout 0.1 \
    --mpnn-type GGNN \
    --pe-embed-k 0 \
    --pool-op avg \
    --set-layers 0 \
    --frag-set-layers 3 \
    --hidden-size 256 \
    --weight-decay 1e-7 \
    --gnn-layers 4 \
    --mlp-layers 1 \
    --root-encode gnn \
    --binned-targs \
    --loss-fn cosine \
    --encode-forms \
    --embed-adduct \
    --add-hs \
    --gpu \
    --save-dir $save_dir \
    --magma-dag-folder $inten_dir
