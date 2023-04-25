overfit_ckpt=/home/samlg/projects/ms_pred/results/2022_12_23_ffn/version_22/epoch=964-val_loss=0.00.ckpt



python src/ms_pred/ffn_pred/predict.py  --batch-size 32  --dataset-name \
gnps2015_debug --dataset-labels labels.tsv --split-name split_22.tsv \
--subset-datasets train_only \
--checkpoint $overfit_ckpt --save-dir results/2022_12_23_ffn_pred/
