# 启动第一个进程并获取其 PID
nohup ./finetune_lora_with_accelerate1.sh > finetune_triviaqa_llama3_11.log 2>&1 &
PID1=$!  # 获取第一个进程的 PID

# 等待第一个进程结束
wait $PID1

# 启动第二个进程并获取其 PID
nohup ./finetune_lora_with_accelerate2.sh > finetune_triviaqa_llama3_12.log 2>&1 &
PID2=$!  # 获取第二个进程的 PID

# 等待第二个进程结束
wait $PID2

# 启动第三个进程
nohup ./finetune_lora_with_accelerate3.sh > finetune_triviaqa_llama3_13.log 2>&1 &
PID3=$!


wait $PID3
nohup ./finetune_lora_with_accelerate.sh > finetune_triviaqa_llama3.log 2>&1 &