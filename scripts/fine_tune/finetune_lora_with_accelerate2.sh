MODEL_SIZE=1
MODEL_NAME=models/llama3.2-triviaqa12-${MODEL_SIZE}b
TRAIN_FILE=../../data/train/triviaqa/sampled_train12.jsonl
NUM_GPUS=2
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=128
NUM_EPOCHS=2
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE}b using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
echo "Output dir: ${MODEL_NAME}"

export MASTER_PORT=29504

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ../../open_instruct/ds_configs/stage3_no_offloading_accelerate.conf \
    ../../open_instruct/finetune.py \
    --model_name_or_path meta-llama/Llama-3.2-${MODEL_SIZE}B \
    --tokenizer_name meta-llama/Llama-3.2-${MODEL_SIZE}B \
    --use_lora \
    --use_slow_tokenizer \
    --train_file $TRAIN_FILE \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs $NUM_EPOCHS \
    --checkpointing_steps epoch \
    --logging_steps 1 \
    --output_dir $MODEL_NAME \
