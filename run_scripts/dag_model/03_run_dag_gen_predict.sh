python launcher_scripts/run_from_config.py configs/iceberg/dag_gen_predict_train_canopus.yaml
python launcher_scripts/run_from_config.py configs/iceberg/dag_gen_predict_train_nist.yaml


# Assign intensities to prediction for next training run

python data_scripts/dag/add_dag_intens.py \
	--pred-dag-folder  results/dag_nist20/scaffold_1/preds_train_100/tree_preds \
	--true-dag-folder data/spec_datasets/nist20/subformulae/no_subform/ \
	--out-dag-folder results/dag_nist20/scaffold_1/preds_train_100_inten  \
	--num-workers 16 \
	--add-raw

python data_scripts/dag/add_dag_intens.py \
	--pred-dag-folder  results/dag_nist20/split_1_rnd1/preds_train_100/tree_preds \
	--true-dag-folder data/spec_datasets/nist20/subformulae/no_subform/ \
	--out-dag-folder results/dag_nist20/split_1_rnd1/preds_train_100_inten  \
	--num-workers 16 \
	--add-raw

python data_scripts/dag/add_dag_intens.py \
	--pred-dag-folder  results/dag_canopus_train_public/split_1_rnd1/preds_train_100/tree_preds \
	--true-dag-folder data/spec_datasets/canopus_train_public/subformulae/no_subform/ \
	--out-dag-folder results/dag_canopus_train_public/split_1_rnd1/preds_train_100_inten  \
	--num-workers 16 \
	--add-raw

python data_scripts/dag/add_dag_intens.py \
	--pred-dag-folder  results/dag_nist20/split_1_rnd2/preds_train_100/tree_preds \
	--true-dag-folder data/spec_datasets/nist20/subformulae/no_subform/ \
	--out-dag-folder results/dag_nist20/split_1_rnd2/preds_train_100_inten  \
	--num-workers 16 \
	--add-raw

python data_scripts/dag/add_dag_intens.py \
	--pred-dag-folder  results/dag_canopus_train_public/split_1_rnd2/preds_train_100/tree_preds \
	--true-dag-folder data/spec_datasets/canopus_train_public/subformulae/no_subform/ \
	--out-dag-folder results/dag_canopus_train_public/split_1_rnd2/preds_train_100_inten  \
	--num-workers 16 \
	--add-raw

python data_scripts/dag/add_dag_intens.py \
	--pred-dag-folder  results/dag_nist20/split_1_rnd3/preds_train_100/tree_preds \
	--true-dag-folder data/spec_datasets/nist20/subformulae/no_subform/ \
	--out-dag-folder results/dag_nist20/split_1_rnd3/preds_train_100_inten  \
	--num-workers 16 \
	--add-raw

python data_scripts/dag/add_dag_intens.py \
	--pred-dag-folder  results/dag_canopus_train_public/split_1_rnd3/preds_train_100/tree_preds \
	--true-dag-folder data/spec_datasets/canopus_train_public/subformulae/no_subform/ \
	--out-dag-folder results/dag_canopus_train_public/split_1_rnd3/preds_train_100_inten  \
	--num-workers 16 \
	--add-raw
