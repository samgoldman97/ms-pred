split_name_no_ext="canopus_hplus_100_0"
dataset_short="canopus_train"
dataset_name="mist_augment_canopus_train"
input_labels="data/spec_datasets/${dataset_name}/biomols_filtered_smiles_${dataset_short}_labels.tsv"

save_dir="results/${dataset_name}_train_inten/$split_name_no_ext/full_preds/"
inten_model="results/${dataset_name}_train_inten/$split_name_no_ext/version_0/best.ckpt"
gen_model="results/${dataset_name}_train_gen/$split_name_no_ext/version_0/best.ckpt"
max_nodes=100

# Predict all smiles
python src/ms_pred/dag_pred/predict_smis.py \
    --batch-size 96 \
    --sparse-out \
    --sparse-k 100 \
    --max-nodes $max_nodes \
    --gen-checkpoint $gen_model \
    --inten-checkpoint $inten_model \
    --save-dir ${save_dir} \
    --dataset-labels $input_labels \
    --num-workers 96

# Convert to mgf
python data_scripts/dag/dag_to_mgf.py \
        --dag-folder $save_dir/tree_preds_inten \
        --out $save_dir/full_out.mgf \
        --num-workers 16

# Delete memory intensive tree preds inten
rm -r $save_dir/tree_preds_inten
