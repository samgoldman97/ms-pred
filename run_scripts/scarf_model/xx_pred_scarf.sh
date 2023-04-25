#overfit_ckpt='results/debug_overfit_scarf_gen/version_0/best.ckpt'
overfit_ckpt='results/scarf/split_1/version_0/best.ckpt'

python -m pdb src/ms_pred/scarf_pred/predict_gen.py \
--batch-size 5  \
--dataset-name  canopus_train_public \
--dataset-labels labels.tsv \
--split-name split_1.tsv \
--checkpoint $overfit_ckpt \
--save-dir results/scarf/split_1/  \
--subset-datasets test_only 

#--subset-datasets debug_special   \

#--subset-datasets debug_special   \
#--subset-datasets debug_special   \
#--save-dir results/debug_overfit_scarf_gen \
