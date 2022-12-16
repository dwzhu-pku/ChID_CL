path_to_dataset=./data/ChID
data_size=1w
path_to_output=./results/bart_baseline_${data_size}
# path_to_output=./results/bart_baseline_0
log_path=${path_to_output}/log.txt
# debug_mode="-m debugpy --listen 127.0.0.1:6673 --wait-for-client"
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --multi_gpu --mixed_precision=fp16 ${debug_mode} ./src/bart_baseline.py \
    --model_name_or_path fnlp/bart-large-chinese \
    --train_file ${path_to_dataset}/train_data_${data_size}.json \
    --valid_file ${path_to_dataset}/test_data.json \
    --num_beams 5 \
    --do_eval once \
    --do_train \
    --do_predict \
    --preprocessing_num_workers 4 \
    --output_dir ${path_to_output} \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --weight_decay 0.01 \
    --num_train_epochs 10 \
    --resume_from_checkpoint ${path_to_output}/epoch_4 \
    --report_to wandb \
    --checkpointing_steps epoch 2>&1 | tee -a ${log_path}
    # --with_tracking \
    # --max_predict_samples 100 \