# Evaluating DExperts
size=13
echo "Results dir: results/truthfulqa/dexperts-${size}B-helpful-prompt"
python -m eval.truthfulqa.run_eval \
    --base_model_name_or_path meta-llama/Llama-2-${size}b-hf \
    --expert_model_name_or_path_1  \
    --expert_model_name_or_path_2  \
    --expert_model_name_or_path_3  \
    --save_dir results/truthfulqa/dexperts-${size}B-helpful-prompt \
    --settings open mc \
    --system_prompt "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information." \
    --gpt_true_model_name deepseek-reasoner \
    --gpt_info_model_name deepseek-chat \
    --eval_batch_size 4


# Evaluating Llama 2
size=13
echo "Results dir: results/truthfulqa/llama2-${size}B"
python -m eval.truthfulqa.run_eval \
    --model_name_or_path meta-llama/Llama-2-${size}b-hf \
    --save_dir results/truthfulqa/llama2-${size}B \
    --settings open mc \
    --gpt_true_model_name deepseek-reasoner \
    --gpt_info_model_name deepseek-chat \
    --eval_batch_size 4


# Evaluating Llama 2 chat
size=13
echo "Results dir: results/truthfulqa/llama2-chat-${size}B-helpful-prompt"
python -m eval.truthfulqa.run_eval \
    --model_name_or_path meta-llama/Llama-2-${size}b-chat-hf \
    --save_dir results/truthfulqa/llama2-chat-${size}B-helpful-prompt \
    --settings open mc \
    --system_prompt "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information." \
    --gpt_true_model_name deepseek-reasoner \
    --gpt_info_model_name deepseek-chat \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format
