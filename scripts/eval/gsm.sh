export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/mnt/data1/wangyatong/proxy-tuning"

# # Evaluating DExperts with GSM expert and realtimeqa expert
# size=13
# echo "Evaluating DExperts with GSM expert and realtimeqa expert"
# echo "Results dir: results/gsm/multi-dexperts-${size}B"
# python -m eval.gsm.run_eval \
#     --data_dir ../../data/eval/gsm/ \
#     --save_dir ../../results/gsm/dexperts-${size}B \
#     --base_model_name_or_path meta-llama/Llama-2-${size}b-hf \
#     --expert_model_name_or_path ../models/llama2-gsm1-7b \
#     --expert_model_name_or_path_2 ../models/llama2-gsm2-7b \
#     --expert_model_name_or_path_3 ../models/llama2-gsm3-7b \
#     --eval_batch_size 8

# # Evaluating DExperts with chat expert
# size=13
# echo "Evaluating DExperts with chat expert"
# echo "Results dir: results/gsm/dexperts-${size}B"
# python -m eval.gsm.run_eval \
#     --data_dir ../../data/eval/gsm/ \
#     --save_dir ../../results/gsm/dexperts-${size}B \
#     --base_model_name_or_path meta-llama/Llama-2-${size}b-hf \
#     --expert_model_name_or_path meta-llama/Llama-2-7b-chat-hf \
#     --eval_batch_size 8


# # Evaluating DExperts with GSM expert
# size=13
# echo "Evaluating DExperts with GSM expert"
# echo "Results dir: results/gsm/dexperts-gsm-${size}B"
# python -m eval.gsm.run_eval \
#     --data_dir ../../data/eval/gsm/ \
#     --save_dir ../../results/gsm/dexperts-gsm-${size}B \
#     --base_model_name_or_path meta-llama/Llama-2-${size}b-hf \
#     --expert_model_name_or_path ../models/llama2-gsm-7b \
#     --eval_batch_size 8


# # Evaluating Llama 2 13b
# size=13
# echo "Evaluating Llama 2 13B"
# echo "Results dir: results/gsm/llama2-${size}B"
# python -m eval.gsm.run_eval \
#     --data_dir ../../data/eval/gsm/ \
#     --save_dir ../../results/gsm/llama2-${size}B \
#     --model_name_or_path meta-llama/Llama-2-${size}b-hf \
#     --eval_batch_size 8


# # Evaluating Llama 2 13B chat
# size=13
# echo "Evaluating Llama 2 chat"
# echo "Results dir: results/gsm/llama2-chat-${size}B"
# python -m eval.gsm.run_eval \
#     --data_dir ../../data/eval/gsm/ \
#     --save_dir ../../results/gsm/llama2-chat-${size}B \
#     --model_name_or_path meta-llama/Llama-2-${size}b-chat-hf \
#     --eval_batch_size 8 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format


# Evaluating GSM expert
size=7
echo "Evaluating GSM123 expert"
echo "Results dir: results/gsm/llama2-gsm1-${size}B"
python -m eval.gsm.run_eval \
    --data_dir ../../data/eval/gsm/ \
    --save_dir results/gsm/llama2-gsm1-${size}B \
    --model_name_or_path ../models/llama2-gsm1-merged \
    --eval_batch_size 8

size=7
echo "Evaluating GSM123 expert"
echo "Results dir: results/gsm/llama2-gsm12-${size}B"
python -m eval.gsm.run_eval \
    --data_dir ../../data/eval/gsm/ \
    --save_dir results/gsm/llama2-gsm12-${size}B \
    --model_name_or_path ../models/llama2-gsm12-merged \
    --eval_batch_size 8

size=7
echo "Evaluating GSM123 expert"
echo "Results dir: results/gsm/llama2-gsm123-${size}B"
python -m eval.gsm.run_eval \
    --data_dir ../../data/eval/gsm/ \
    --save_dir results/gsm/llama2-gsm123-${size}B \
    --model_name_or_path ../models/llama2-gsm123-merged \
    --eval_batch_size 8

# # Evaluating llama2 7b
# size=7
# echo "Evaluating Llama 2 7B"
# echo "Results dir: results/gsm/llama2-${size}B"
# python -m eval.gsm.run_eval \
#     --data_dir ../../data/eval/gsm/ \
#     --save_dir ../../results/gsm/llama2-${size}B \
#     --model_name_or_path meta-llama/Llama-2-${size}b-hf \
#     --eval_batch_size 8 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format

# # Evaluating llama2 7b chat
# size=7
# echo "Evaluating Llama 2 chat"
# echo "Results dir: results/gsm/llama2-chat-${size}B"
# python -m eval.gsm.run_eval \
#     --data_dir ../../data/eval/gsm/ \
#     --save_dir ../../results/gsm/llama2-chat-${size}B \
#     --model_name_or_path meta-llama/Llama-2-${size}b-chat-hf \
#     --eval_batch_size 8 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format