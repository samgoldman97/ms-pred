overfit_ckpt="/home/samlg/projects/iceberg/results/2022_12_15_tree_pred/version_0/epoch=751-val_loss=0.00.ckpt"

overfit_ckpt=/home/samlg/projects/iceberg/results/2022_12_16_tree_pred/version_0/epoch=860-val_loss=0.00.ckpt
overfit_ckpt=/home/samlg/projects/iceberg/results/2022_12_16_tree_pred/version_1/epoch=871-val_loss=0.00.ckpt

overfit_ckpt=/home/samlg/projects/iceberg/results/2022_12_16_tree_pred/version_3/epoch=593-val_loss=0.00.ckpt

overfit_ckpt=/home/samlg/projects/iceberg/results/2022_12_16_tree_pred/version_4/epoch=827-val_loss=0.00.ckpt #with val loss of 4.111225644010119e-05

overfit_ckpt=/home/samlg/projects/iceberg/results/2022_12_17_tree_pred/version_0/epoch=874-val_loss=0.00.ckpt
overfit_ckpt=/home/samlg/projects/iceberg/results/2022_12_17_tree_pred/version_2/epoch=845-val_loss=0.00.ckpt

overfit_ckpt='results/2022_12_17_tree_pred/version_2/epoch=845-val_loss=0.00.ckpt'

overfit_ckpt=/home/samlg/projects/iceberg/results/2022_12_18_tree_pred/version_1/epoch=830-val_loss=0.00.ckpt

overfit_ckpt=/home/samlg/projects/iceberg/results/2022_12_18_tree_pred/version_2/epoch=139-val_loss=0.00.ckpt

overfit_ckpt=/home/samlg/projects/iceberg/results/2022_12_18_tree_pred/version_0/epoch=660-val_loss=0.00.ckpt
#With val loss of 3.169644332956523e-05

overfit_ckpt=/home/samlg/projects/iceberg/results/2022_12_18_tree_pred/version_1/epoch=755-val_loss=0.00.ckpt
#With val loss of 3.805951928370632e-05
overfit_ckpt=results/2022_12_23_tree_pred/version_0/epoch\=526-val_loss\=0.00.ckpt
overfit_ckpt=/home/samlg/projects/iceberg/results/2022_12_27_tree_pred/version_0/epoch=97-val_loss=0.00.ckpt

python src/ms_pred/dag_pred/predict_gen.py  --batch-size 1  --dataset-name \
gnps2015_debug --dataset-labels labels.tsv --split-name split_22.tsv \
--subset-datasets train_only --thresh 0.08 \
--checkpoint $overfit_ckpt --save-dir \
results/debug_overfit_gen

#python src/ms_pred/tree_pred/predict.py  --batch-size 1  --dataset-name \
#gnps2015_debug --dataset-labels labels.tsv --split-name split_22.tsv \
#--subset-datasets train_only --thresh 0.1 \
#--checkpoint $overfit_ckpt --save-dir \
#results/2022_12_15_tree_pred/overfit_debug_preds_01/
#
#python src/ms_pred/tree_pred/predict.py  --batch-size 1  --dataset-name \
#gnps2015_debug --dataset-labels labels.tsv --split-name split_22.tsv \
#--subset-datasets train_only --thresh 0.2 \
#--checkpoint $overfit_ckpt --save-dir \
#results/2022_12_15_tree_pred/overfit_debug_preds_02/
#
#python src/ms_pred/tree_pred/predict.py  --batch-size 1  --dataset-name \
#gnps2015_debug --dataset-labels labels.tsv --split-name split_22.tsv \
#--subset-datasets train_only --thresh 0.3 \
#--checkpoint $overfit_ckpt --save-dir \
#results/2022_12_15_tree_pred/overfit_debug_preds_03/
