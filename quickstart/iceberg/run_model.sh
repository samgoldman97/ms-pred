#gen_model="quickstart/iceberg/models/nist_iceberg_generate.ckpt"
#score_model="quickstart/iceberg/models/nist_iceberg_score.ckpt"

gen_model="quickstart/iceberg/models/canopus_iceberg_generate.ckpt"
score_model="quickstart/iceberg/models/canopus_iceberg_score.ckpt"

labels="data/spec_datasets/sample_labels.tsv"
python src/ms_pred/dag_pred/predict_smis.py \
    --batch-size 16 \
    --max-nodes 100 \
    --gen-checkpoint $gen_model \
    --inten-checkpoint $score_model \
    --save-dir quickstart/iceberg/out \
    --dataset-labels $labels \
    --num-workers 0 \

    # --gpu
    # --binned-out
