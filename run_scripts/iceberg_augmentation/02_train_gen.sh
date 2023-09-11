split_name_no_ext="canopus_hplus_100_0"
split_name="${split_name_no_ext}.tsv"
dataset_name="mist_augment_canopus_train"
save_dir="results/${dataset_name}_train_gen/$split_name_no_ext"
CUDA_VISIBLE_DEVICES=0,1,2 python src/ms_pred/dag_pred/train_gen.py \
    --seed 1 \
    --num-workers 8  \
    --batch-size 32 \
    --max-epochs 200 \
    --dataset-name $dataset_name \
    --split-name $split_name \
    --learning-rate 0.000996 \
    --lr-decay-rate 0.7214 \
    --dropout 0.2 \
    --mpnn-type GGNN \
    --pe-embed-k 14 \
    --pool-op avg \
    --set-layers 0 \
    --hidden-size 512 \
    --weight-decay 0 \
    --layers 6 \
    --root-encode gnn \
    --encode-forms \
    --embed-adduct \
    --add-hs \
    --gpu \
    --save-dir $save_dir
