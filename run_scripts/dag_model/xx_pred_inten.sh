overfit_ckpt=/home/samlg/projects/iceberg/results/debug_hurdle_model_inten/version_0/epoch=888-val_loss=0.00.ckpt

python src/ms_pred/dag_pred/predict_inten.py  --batch-size 1  --dataset-name \
gnps2015_debug --dataset-labels labels.tsv --split-name split_22.tsv \
--subset-datasets train_only --inten-hurdle 0.2 \
--checkpoint $overfit_ckpt --save-dir \
results/debug_hurdle_model_inten/
