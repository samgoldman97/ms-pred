#dataset=gnps2015_debug
dataset=canopus_train_public
dataset=nist20
max_peaks=50
python3 src/ms_pred/magma/run_magma.py  \
--spectra-dir data/spec_datasets/$dataset/spec_files  \
--output-dir data/spec_datasets/$dataset/magma_outputs  \
--spec-labels data/spec_datasets/$dataset/labels.tsv \
--max-peaks $max_peaks \

### 
#dataset=nist20
#max_peaks=50
#python3 src/ms_pred/magma/run_magma.py  \
#--spectra-dir data/spec_datasets/$dataset/spec_files  \
#--output-dir data/spec_datasets/$dataset/magma_outputs  \
#--spec-labels data/spec_datasets/$dataset/labels.tsv \
#--max-peaks $max_peaks
