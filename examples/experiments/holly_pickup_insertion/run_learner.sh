export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=holly_pickup_insertion \
    --checkpoint_path=../../experiments/holly_pickup_insertion/debug \
    --demo_path=../../experiments/holly_pickup_insertion/demo_data/holly_pickup_insertion_20_demos_2025-04-25_21-02-03.pkl \
    --learner \