python launcher_scripts/run_from_config.py configs/scarf/scarf_gen_predict_train_canopus.yaml
python launcher_scripts/run_from_config.py configs/scarf/scarf_gen_predict_train_nist.yaml

# Assign intensities to enable easy training in next module
python data_scripts/forms/03_add_form_intens.py \
	--pred-form-folder results/scarf_nist20/split_1/preds_train_100/form_preds \
	--true-form-folder data/spec_datasets/nist20/subformulae/no_subform/ \
	--out-form-folder results/scarf_nist20/split_1/preds_train_100_inten  \
	--num-workers 16 \
	--add-raw \
	--binned-add


python data_scripts/forms/03_add_form_intens.py \
	--pred-form-folder results/scarf_nist20/scaffold_1/preds_train_100/form_preds \
	--true-form-folder data/spec_datasets/nist20/subformulae/no_subform/ \
	--out-form-folder results/scarf_nist20/scaffold_1/preds_train_100_inten  \
	--num-workers 16 \
	--add-raw \
	--binned-add


python data_scripts/forms/03_add_form_intens.py \
	--pred-form-folder results/scarf_canopus_train_public/split_1/preds_train_100/form_preds \
	--true-form-folder data/spec_datasets/canopus_train_public/subformulae/no_subform/ \
	--out-form-folder results/scarf_canopus_train_public/split_1/preds_train_100_inten  \
	--num-workers 16 \
	--add-raw \
	--binned-add
