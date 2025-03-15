#!/bin/bash

# 设置参数
LORA_MODEL_PATH="/home/wangyatong/proxy-tuning/scripts/eval/models/llama2-triviaqa11-7b"  # LoRA 模型的路径
BASE_MODEL_PATH="meta-llama/Llama-2-7b-hf"  # 基础模型的路径
TOKENIZER_PATH="/home/wangyatong/proxy-tuning/scripts/eval/models/llama2-triviaqa11-7b"    # 分词器的路径（可选）
OUTPUT_DIR="/home/wangyatong/proxy-tuning/scripts/eval/models/llama2-triviaqa11-7b-m"     # 合并后模型的输出目录
USE_QLORA=false                 # 是否使用 QLoRA
SAVE_TOKENIZER=true             # 是否保存分词器
USE_FAST_TOKENIZER=true         # 是否使用快速分词器

# 调用 Python 脚本
python merge_lora.py \
    --lora_model_name_or_path "$LORA_MODEL_PATH" \
    --base_model_name_or_path "$BASE_MODEL_PATH" \
    --tokenizer_name_or_path "$TOKENIZER_PATH" \
    --output_dir "$OUTPUT_DIR" \
    $([ "$USE_QLORA" = true ] && echo "--qlora") \
    $([ "$SAVE_TOKENIZER" = true ] && echo "--save_tokenizer") \
    $([ "$USE_FAST_TOKENIZER" = true ] && echo "--use_fast_tokenizer")