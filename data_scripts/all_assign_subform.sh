# Canopus train public

ppm_diff=20
workers=64
python data_scripts/forms/01_assign_subformulae.py --data-dir data/spec_datasets/canopus_train_public/ --labels-file data/spec_datasets/canopus_train_public/labels.tsv --use-magma --mass-diff-thresh $ppm_diff --output-dir magma_subform_50

python data_scripts/forms/01_assign_subformulae.py --data-dir data/spec_datasets/canopus_train_public/ --labels-file data/spec_datasets/canopus_train_public/labels.tsv --use-all --output-dir no_subform

python data_scripts/forms/03_add_form_intens.py \
    --num-workers $workers \
    --pred-form-folder data/spec_datasets/canopus_train_public/subformulae/magma_subform_50/ \
    --true-form-folder data/spec_datasets/canopus_train_public/subformulae/no_subform/ \
    --add-raw \
    --binned-add \
    --out-form-folder data/spec_datasets/canopus_train_public/subformulae/magma_subform_50_with_raw



# NIST20 (commercial dataset)
python data_scripts/forms/01_assign_subformulae.py --data-dir data/spec_datasets/nist20/ --labels-file data/spec_datasets/nist20/labels.tsv --use-magma --mass-diff-thresh $ppm_diff --output-dir magma_subform_50
python data_scripts/forms/01_assign_subformulae.py --data-dir data/spec_datasets/nist20/ --labels-file data/spec_datasets/nist20/labels.tsv --use-all --output-dir no_subform

python data_scripts/forms/03_add_form_intens.py \
    --num-workers $workers \
    --pred-form-folder data/spec_datasets/nist20/subformulae/magma_subform_50/ \
    --true-form-folder data/spec_datasets/nist20/subformulae/no_subform/ \
    --add-raw \
    --binned-add \
    --out-form-folder data/spec_datasets/nist20/subformulae/magma_subform_50_with_raw

# CASMI22
# python data_scripts/forms/01_assign_subformulae.py --data-dir data/spec_datasets/casmi22/ --labels-file data/spec_datasets/casmi22/labels.tsv --use-all --output-dir no_subform