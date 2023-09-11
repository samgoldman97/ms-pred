dataset=mist_augment_canopus_train
max_peaks=50
python3 src/ms_pred/magma/run_magma.py  \
    --spectra-dir data/spec_datasets/$dataset/spec_files  \
    --output-dir data/spec_datasets/$dataset/magma_outputs  \
    --spec-labels data/spec_datasets/$dataset/labels.tsv \
    --max-peaks $max_peaks \

python data_scripts/forms/01_assign_subformulae.py \
    --data-dir data/spec_datasets/$dataset \
    --labels-file data/spec_datasets/$dataset/labels.tsv \
    --use-all \
    --output-dir no_subform \
    --max-formulae 50
