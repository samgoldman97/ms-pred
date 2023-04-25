# Canopus train public
python data_scripts/forms/01_assign_subformulae.py --data-dir data/spec_datasets/canopus_train_public/ --labels-file data/spec_datasets/canopus_train_public/labels.tsv --use-magma --mass-diff-thresh 20 --output-dir magma_subform_50
python data_scripts/forms/01_assign_subformulae.py --data-dir data/spec_datasets/canopus_train_public/ --labels-file data/spec_datasets/canopus_train_public/labels.tsv --use-all --output-dir no_subform

# NIST20 (commercial dataset)
python data_scripts/forms/01_assign_subformulae.py --data-dir data/spec_datasets/nist20/ --labels-file data/spec_datasets/nist20/labels.tsv --use-magma --mass-diff-thresh 20 --output-dir magma_subform_50
python data_scripts/forms/01_assign_subformulae.py --data-dir data/spec_datasets/nist20/ --labels-file data/spec_datasets/nist20/labels.tsv --use-all --output-dir no_subform
