. run_scripts/scarf_model/01_run_scarf_gen_train.sh
python run_scripts/scarf_model/02_sweep_scarf_gen_thresh.py
. run_scripts/scarf_model/03_scarf_gen_predict.sh
. run_scripts/scarf_model/04_train_scarf_inten.sh
python run_scripts/scarf_model/05_predict_form_inten.py
python run_scripts/scarf_model/06_run_retrieval.py
python run_scripts/scarf_model/08_export_forms.py
