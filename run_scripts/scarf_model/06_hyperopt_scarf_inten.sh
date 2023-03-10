CUDA_VISIBLE_DEVICES=1,2,3 python src/ms_pred/scarf_pred/inten_hyperopt.py \
--seed 5  \
--num-workers 10  \
--max-epochs 200 \
--dataset-name nist20 \
--dataset-labels labels.tsv \
--loss-fn cosine \
--split-name hyperopt.tsv  \
 --gpu  \
 --save-dir results/scarf_inten_hyperopt_v4 \
 --formula-folder results/scarf_nist_hyperopt_adduct_mol/hyperopt/preds_train_100_inten/ \
--cpus-per-trial 10 \
--gpus-per-trial 0.5 \
--num-h-samples 100 \
--grace-period 900 \
--max-concurrent 10  \
--binned-targs \
--embed-adduct \


 #--formula-folder results/scarf_nist_hyperopt/hyperopt/preds_train_100_inten/ \

