. run_scripts/dag_model/01_run_dag_gen_train.sh
python run_scripts/dag_model/02_sweep_gen_thresh.py
. run_scripts/dag_model/03_run_dag_gen_predict.sh
. run_scripts/dag_model/04_train_dag_inten.sh
python run_scripts/dag_model/05_predict_dag_inten.py
python run_scripts/dag_model/06_run_retrieval.py
python run_scripts/dag_model/08_export_preds.py

