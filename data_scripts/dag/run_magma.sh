#dataset=gnps2015_debug
dataset=canopus_train_public
dataset=nist20
max_peaks=50
ppm_diff=20
python3 src/ms_pred/magma/run_magma.py  \
--spectra-dir data/spec_datasets/$dataset/spec_files  \
--output-dir data/spec_datasets/$dataset/magma_outputs  \
--spec-labels data/spec_datasets/$dataset/labels.tsv \
--max-peaks $max_peaks  \
--ppm-diff $ppm_diff

### 
dataset=nist20
max_peaks=50
ppm_diff=20
python3 src/ms_pred/magma/run_magma.py  \
--spectra-dir data/spec_datasets/$dataset/spec_files  \
--output-dir data/spec_datasets/$dataset/magma_outputs  \
--spec-labels data/spec_datasets/$dataset/labels.tsv \
--max-peaks $max_peaks \
--ppm-diff $ppm_diff
