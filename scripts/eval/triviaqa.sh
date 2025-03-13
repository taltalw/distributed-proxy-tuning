export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="/data6/wangyatong/proxy-tuning"
export CUDA_VISIBLE_DEVICES=0,3

# Evaluating DExperts with triviaqa expert
size=13
echo "Evaluating DExperts with multi triviaqa expert"
echo "Results dir: results/triviaqa/multi-dexperts-${size}B"
python -m eval.triviaqa.run_eval \
    --data_dir ../../data/eval/triviaqa/ \
    --save_dir ../../results/triviaqa/multi-dexperts-${size}B3 \
    --base_model_name_or_path meta-llama/Llama-2-${size}b-hf \
    --expert_model_name_or_path leah1014/llama2-triviaqa11-7b \
    --expert_model_name_or_path_2 leah1014/llama2-triviaqa12-7b \
    --expert_model_name_or_path_3 leah1014/llama2-triviaqa13-7b \
    --eval_batch_size 8

# # size=13
# # echo "Evaluating DExperts with triviaqa expert"
# # echo "Results dir: results/triviaqa/dexperts-${size}B"
# # python -m eval.triviaqa.run_eval \
# #     --data_dir ../../data/eval/triviaqa/ \
# #     --base_model_name_or_path meta-llama/Llama-2-${size}b-hf \
# #     --expert_model_name_or_path models/llama2-triviaqa-7b \
# #     --save_dir ../../results/triviaqa/dexperts-triviaqa-${size}B \
# #     --eval_batch_size 8

# # size=13
# # echo "Evaluating DExperts with chat expert"
# # echo "Results dir: results/triviaqa/dexperts-chat-${size}B"
# # python -m eval.triviaqa.run_eval \
# #     --data_dir ../../data/eval/triviaqa/ \
# #     --base_model_name_or_path meta-llama/Llama-2-${size}b-hf \
# #     --expert_model_name_or_path meta-llama/Llama-2-7b-chat-hf \
# #     --save_dir ../../results/triviaqa/dexperts-triviaqa-chat-${size}B \
# #     --eval_batch_size 8


# # # Evaluating Llama 2
# # size=13
# # echo "Evaluating llama2 13b"
# # echo "Results dir: results/triviaqa/llama2-${size}B"
# # python -m eval.triviaqa.run_eval \
# #     --data_dir ../../data/eval/triviaqa/ \
# #     --model_name_or_path meta-llama/Llama-2-${size}b-hf \
# #     --save_dir ../../results/triviaqa/llama2-${size}B \
# #     --eval_batch_size 8


# # # Evaluating Llama 2 chat
# # size=13
# # echo "Evaluating llama2 13b chat"
# # echo "Results dir: results/triviaqa/llama2-chat-${size}B"
# # python -m eval.triviaqa.run_eval \
# #     --data_dir ../../data/eval/triviaqa/ \
# #     --model_name_or_path meta-llama/Llama-2-${size}b-chat-hf \
# #     --save_dir ../../results/triviaqa/llama2-chat-${size}B \
# #     --eval_batch_size 8



# # Evaluating expert
# size=13
# echo "Evaluating llama2 13b triviaqa"
# python -m eval.triviaqa.run_eval \
#     --data_dir ../../data/eval/triviaqa/ \
#     --model_name_or_path models/llama2-triviaqa-${size}b \
#     --save_dir ../../results/triviaqa/llama2-triviaqa-${size}B \
#     --eval_batch_size 8


# # #---7b---
# # # Evaluating Llama 2 7b
# # size=7
# # echo "Evaluating llama2 7b"
# # echo "Results dir: results/triviaqa/llama2-${size}B"
# # python -m eval.triviaqa.run_eval \
# #     --data_dir ../../data/eval/triviaqa/ \
# #     --model_name_or_path meta-llama/Llama-2-${size}b-hf \
# #     --save_dir ../../results/triviaqa/llama2-${size}B \
# #     --eval_batch_size 8


# # # Evaluating Llama 2 chat
# # size=7
# # echo "Evaluating llama2 7b chat"
# # echo "Results dir: results/triviaqa/llama2-chat-${size}B"
# # python -m eval.triviaqa.run_eval \
# #     --data_dir ../../data/eval/triviaqa/ \
# #     --model_name_or_path meta-llama/Llama-2-${size}b-chat-hf \
# #     --save_dir ../../results/triviaqa/llama2-chat-${size}B \
# #     --eval_batch_size 8



# # Evaluating expert
# size=7
# echo "Evaluating llama2 7b triviaqa"
# python -m eval.triviaqa.run_eval \
#     --data_dir ../../data/eval/triviaqa/ \
#     --model_name_or_path models/llama2-triviaqa-${size}b \
#     --save_dir ../../results/triviaqa/llama2-triviaqa-${size}B \
#     --eval_batch_size 8


