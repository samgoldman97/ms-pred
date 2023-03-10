#thread_model="results/scarf_nist20/split_1/version_0/best.ckpt"
#weave_model="results/scarf_inten_nist20/split_1/version_0/best.ckpt"
thread_model="quickstart/models/nist_thread_model.ckpt"
weave_model="quickstart/models/nist_weave_model.ckpt"

labels="data/spec_datasets/sample_labels.tsv"
python src/ms_pred/scarf_pred/predict_smis.py \
    --batch-size 32 \
    --sparse-out \
    --max-nodes 300 \
    --gen-checkpoint $thread_model \
    --inten-checkpoint $weave_model \
    --save-dir quickstart/out \
    --dataset-labels $labels \
    --num-workers 0 \

    # --gpu
    # --binned-out
