

python data_scripts/make_splits.py --data-dir data/spec_datasets/nist20 --label-file data/spec_datasets/nist20/labels.tsv  --seed 1 --split-type fingerprint --split-name fingerprint_1.tsv  

python data_scripts/make_splits.py --data-dir data/spec_datasets/nist20/ --label-file data/spec_datasets/nist20/labels.tsv --seed 1 --split-type scaffold --split-name scaffold_1.tsv --greedy


python data_scripts/make_splits.py --data-dir data/spec_datasets/nist20/ --label-file data/spec_datasets/nist20/labels.tsv --seed 1
python data_scripts/make_splits.py --data-dir data/spec_datasets/canopus_train_public/ --label-file data/spec_datasets/canopus_train_public/labels.tsv --seed 1



python data_scripts/make_splits.py --data-dir data/spec_datasets/nist20/ --label-file data/spec_datasets/nist20/labels.tsv --seed 2
python data_scripts/make_splits.py --data-dir data/spec_datasets/canopus_train_public/ --label-file data/spec_datasets/canopus_train_public/labels.tsv --seed 2


python data_scripts/make_splits.py --data-dir data/spec_datasets/nist20/ --label-file data/spec_datasets/nist20/labels.tsv --seed 3
python data_scripts/make_splits.py --data-dir data/spec_datasets/canopus_train_public/ --label-file data/spec_datasets/canopus_train_public/labels.tsv --seed 3

# hyperopt
python data_scripts/make_splits.py --data-dir data/spec_datasets/nist20/ --label-file data/spec_datasets/nist20/labels.tsv  --seed 1 --split-name hyperopt.tsv --test-frac 0.5

