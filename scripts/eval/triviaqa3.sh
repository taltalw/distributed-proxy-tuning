export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/wangyatong/proxy-tuning"

# # Evaluating DExperts with triviaqa expert
# size=3
# echo "Evaluating DExperts with multi triviaqa expert"
# echo "Results dir: results/triviaqa/multi-dexperts-${size}B"
# python -m eval.triviaqa.run_eval \
#     --data_dir ../../data/eval/triviaqa/ \
#     --save_dir ../../results/triviaqa/multi-dexperts-${size}B3 \
#     --base_model_name_or_path meta-llama/Llama-3.2-${size}B \
#     --expert_model_name_or_path models/llama3.2-triviaqa11-1b \
#     --expert_model_name_or_path_2 models/llama3.2-triviaqa12-1b \
#     --expert_model_name_or_path_3 models/llama3.2-triviaqa13-1b \
#     --eval_batch_size 8

# size=3
# echo "Evaluating DExperts with triviaqa expert"
# echo "Results dir: results/triviaqa/dexperts-${size}B"
# python -m eval.triviaqa.run_eval \
#     --data_dir ../../data/eval/triviaqa/ \
#     --base_model_name_or_path meta-llama/Llama-3.2-${size}B \
#     --expert_model_name_or_path models/llama3.2-triviaqa-1b \
#     --save_dir ../../results/triviaqa/dexperts-triviaqa-${size}B \
#     --eval_batch_size 8

# size=3
# echo "Evaluating DExperts with chat expert"
# echo "Results dir: results/triviaqa/dexperts-chat-${size}B"
# python -m eval.triviaqa.run_eval \
#     --data_dir ../../data/eval/triviaqa/ \
#     --base_model_name_or_path meta-llama/Llama-3.2-${size}B \
#     --expert_model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
#     --save_dir ../../results/triviaqa/dexperts-triviaqa-INSTRUCT-${size}B \
#     --eval_batch_size 8


# Evaluating Llama 2
size=3
echo "Evaluating llama3.2 3b"
echo "Results dir: results/triviaqa/llama3.2-${size}B"
python -m eval.triviaqa.run_eval \
    --data_dir ../../data/eval/triviaqa/ \
    --model_name_or_path meta-llama/Llama-3.2-${size}B \
    --save_dir ../../results/triviaqa/llama3.2-${size}B \
    --eval_batch_size 8


# Evaluating Llama 2 chat
size=3
echo "Evaluating llama3.2 3b INSTRUCT"
echo "Results dir: results/triviaqa/llama3.2-INSTRUCT-${size}B"
python -m eval.triviaqa.run_eval \
    --data_dir ../../data/eval/triviaqa/ \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --save_dir ../../results/triviaqa/llama3.2-INSTRUCT-${size}B \
    --eval_batch_size 8



# Evaluating expert
size=3
echo "Evaluating llama3.2 3b triviaqa"
python -m eval.triviaqa.run_eval \
    --data_dir ../../data/eval/triviaqa/ \
    --model_name_or_path models/llama3.2-triviaqa-${size}b \
    --save_dir ../../results/triviaqa/llama3.2-triviaqa-${size}B \
    --eval_batch_size 8


#---7b---
# Evaluating Llama 2 7b
size=1
echo "Evaluating llama3.2 1b"
echo "Results dir: results/triviaqa/llama3.2-${size}B"
python -m eval.triviaqa.run_eval \
    --data_dir ../../data/eval/triviaqa/ \
    --model_name_or_path meta-llama/Llama-3.2-${size}B \
    --save_dir ../../results/triviaqa/llama3.2-${size}B \
    --eval_batch_size 8


# Evaluating Llama 2 chat
size=1
echo "Evaluating llama3.2 1b chat"
echo "Results dir: results/triviaqa/llama3.2-INSTRUCT-${size}B"
python -m eval.triviaqa.run_eval \
    --data_dir ../../data/eval/triviaqa/ \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --save_dir ../../results/triviaqa/llama3.2-INSTRUCT-${size}B \
    --eval_batch_size 8



# Evaluating expert
size=1
echo "Evaluating llama3.1 1b triviaqa"
python -m eval.triviaqa.run_eval \
    --data_dir ../../data/eval/triviaqa/ \
    --model_name_or_path models/llama3.2-triviaqa-${size}b \
    --save_dir ../../results/triviaqa/llama3.2-triviaqa-${size}B \
    --eval_batch_size 8


