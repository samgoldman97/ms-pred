split_name_no_ext="canopus_hplus_100_0"
split_name="${split_name_no_ext}.tsv"
dataset_name="mist_augment_canopus_train"
save_dir="results/${dataset_name}_train_gen/$split_name_no_ext/preds_train_100"
inten_dir="results/${dataset_name}_train_gen/$split_name_no_ext/preds_train_100_inten"
checkpoint="results/${dataset_name}_train_gen/$split_name_no_ext/version_0/best.ckpt"
true_dag_folder="data/spec_datasets/${dataset}/subformulae/no_subform/"

CUDA_VISIBLE_DEVICES=1,2,3 python3 src/ms_pred/dag_pred/predict_gen.py \
        --batch-size 64 \
        --dataset-name $dataset_name \
        --num-workers 0 \
        --threshold 0  \
        --max-nodes 100 \
        --split-name $split_name \
        --checkpoint-pth  $checkpoint \
        --save-dir $save_dir


python3 data_scripts/dag/add_dag_intens.py \
        --pred-dag-folder ${save_dir}/tree_preds/ \
        --true-dag-folder $true_dag_folder/  \
        --out-dag-folder $inten_dir \
        --num-workers 16 \
        --add-raw 
