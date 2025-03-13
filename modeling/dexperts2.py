from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
import torch.nn.functional as F
from transformers.generation.utils import (
    ModelOutput,
    # top_k_top_p_filtering,
    StoppingCriteriaList,
    LogitsProcessorList
)
from collections import defaultdict
import torch.nn.functional as F

B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


class DExpertsLlama2:
    def __init__(
        self,
        base_model_name_or_path: str,
        expert_model_name_or_path: str,
        expert_model_name_or_path_2: str,
        expert_model_name_or_path_3: str,
        antiexpert_model_name_or_path: str,
        tokenizer: PreTrainedTokenizer,
        system_prompt: str = None,
        alpha: float = 1.0,
        chat_response_prefix: str = None,
        model_kwargs: Dict[str, Any] = None
    ):
        """
        chat_response_prefix: For llama chat models, it can be helpful for the response
        to start with a certain prefix to constrain the generation to directly answer
        the question. This makes evaluation on MC datasets easier.
        """

        self.base = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path, **model_kwargs
        )
        self.expert_1 = AutoModelForCausalLM.from_pretrained(
            expert_model_name_or_path, **model_kwargs
        )
        self.expert_2 = AutoModelForCausalLM.from_pretrained(
            expert_model_name_or_path_2, **model_kwargs
        )

        self.expert_3 = AutoModelForCausalLM.from_pretrained(
            expert_model_name_or_path_3, **model_kwargs
        )

        self.antiexpert = AutoModelForCausalLM.from_pretrained(
            antiexpert_model_name_or_path, **model_kwargs
        )

        self.base.eval()
        self.expert_1.eval()
        self.expert_2.eval()
        self.expert_3.eval()
        self.antiexpert.eval()

        self.tokenizer = tokenizer
        self.alpha = alpha
        self.device = self.base.device
        self.chat_response_prefix = chat_response_prefix

        # Llama chat experts need different formatting
        self.use_chat_format_for_expert = True if 'chat' in expert_model_name_or_path.lower() else False

        if self.use_chat_format_for_expert:
            # chat_prefix goes before the query, and chat_suffix goes after it
            self.chat_prefix = "[INST]"
            self.chat_suffix = "[/INST]"

            if system_prompt:
                self.chat_prefix += f"{B_SYS}{system_prompt}{E_SYS}"

            if self.chat_response_prefix:
                self.chat_suffix += f" {chat_response_prefix}"

    def forward(
        self,
        base_inputs,
        expert_inputs_1,
        expert_inputs_2,
        expert_inputs_3,
        antiexpert_inputs,
        return_dict=None
    ):
        base_outputs = self.base(**base_inputs, return_dict=return_dict)
        expert_outputs_1 = self.expert_1(**expert_inputs_1, return_dict=return_dict)
        expert_outputs_2 = self.expert_2(**expert_inputs_2, return_dict=return_dict)
        expert_outputs_3 = self.expert_3(**expert_inputs_3, return_dict=return_dict)
        antiexpert_outputs = self.antiexpert(**antiexpert_inputs, return_dict=return_dict)

        return base_outputs, expert_outputs_1, expert_outputs_2, expert_outputs_3, antiexpert_outputs


    def _get_tokenized_chat_inputs(self, input_ids):
        """Decode input_ids and encode again to insert chat formatting"""

        prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # remove response_prefix (e.g., "Answer:") from the prompt if it's already there
        if self.chat_response_prefix:
            cleaned_prompts = []
            for p in prompts:
                if self.chat_response_prefix in p:
                    p = p.replace(self.chat_response_prefix, '').rstrip()
                cleaned_prompts.append(p)
        else:
            cleaned_prompts = prompts

        chat_prompts = [f'{self.chat_prefix} {p} {self.chat_suffix}' for p in cleaned_prompts]
        # print('DExperts expert prompt', flush=True)
        # print(chat_prompts[0], flush=True)
        chat_inputs = self.tokenizer(
            chat_prompts, padding="longest", return_tensors="pt",
            add_special_tokens=True
        )
        chat_inputs.input_ids = chat_inputs.input_ids.to(self.device)
        chat_inputs.attention_mask = chat_inputs.attention_mask.to(self.device)

        return chat_inputs

    def update_analysis_data(self, analysis_data, next_tokens, next_token_logits_dict):
        analysis_data['tokens'].append([self.tokenizer.decode(t) for t in next_tokens])
        analysis_data['token_ids'].append(next_tokens)

        # logits from each model for the next token
        for model in next_token_logits_dict.keys():
            analysis_data[f'logits_{model}'].append(next_token_logits_dict[model].unsqueeze(dim=1))

        return analysis_data

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = 100,
        do_sample: bool = False,
        top_p: float = 1.0,
        temperature: float = 1.0,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        return_logits_for_analysis: bool = False,
        **kwargs
    ):
        base_kwargs = kwargs.copy()
        expert_kwargs_1= kwargs.copy()
        expert_kwargs_2= kwargs.copy()
        expert_kwargs_3= kwargs.copy()
        antiexpert_kwargs = kwargs.copy()

        # prepare inputs for expert model
        if self.use_chat_format_for_expert:
            chat_inputs = self._get_tokenized_chat_inputs(input_ids)
            expert_input_ids = chat_inputs.input_ids.to(input_ids.device)
            expert_kwargs_1['attention_mask'] = chat_inputs.attention_mask
            expert_kwargs_2['attention_mask'] = chat_inputs.attention_mask
            expert_kwargs_3['attention_mask'] = chat_inputs.attention_mask
        else:
            expert_input_ids = input_ids.to(input_ids.device)

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        eos_token_id_tensor = torch.tensor([self.tokenizer.eos_token_id]).to(input_ids.device)

        if return_logits_for_analysis:
            analysis_data = defaultdict(list)

        for step in range(max_new_tokens):
            # prepare model inputs with past_key_values and attention_mask
            base_inputs = self.base.prepare_inputs_for_generation(input_ids, **base_kwargs)
            expert_inputs_1 = self.expert_1.prepare_inputs_for_generation(expert_input_ids, **expert_kwargs_1)
            expert_inputs_2 = self.expert_2.prepare_inputs_for_generation(expert_input_ids, **expert_kwargs_2)
            expert_inputs_3 = self.expert_3.prepare_inputs_for_generation(expert_input_ids, **expert_kwargs_3)

            antiexpert_inputs = self.antiexpert.prepare_inputs_for_generation(input_ids, **antiexpert_kwargs)

            # DExperts
            base_outputs, expert_outputs_1, expert_outputs_2, expert_outputs_3, antiexpert_outputs = self.forward(
                base_inputs, expert_inputs_1, expert_inputs_2, expert_inputs_3, antiexpert_inputs, return_dict=True
            )

            base_next_token_logits = base_outputs.logits[..., -1, :]
            expert_next_token_logits_1 = expert_outputs_1.logits[..., -1, :]
            expert_next_token_logits_2 = expert_outputs_2.logits[..., -1, :]
            expert_next_token_logits_3 = expert_outputs_3.logits[..., -1, :]
            antiexpert_next_token_logits = antiexpert_outputs.logits[..., -1, :]

            # sometimes our experts have extra (irrelevant) tokens at the end of the normal vocabulary
            expert_next_token_logits_1 = expert_next_token_logits_1[:, :base_next_token_logits.shape[-1]]
            expert_next_token_logits_2 = expert_next_token_logits_2[:, :base_next_token_logits.shape[-1]]
            expert_next_token_logits_3 = expert_next_token_logits_3[:, :base_next_token_logits.shape[-1]]


            # DExperts!

            # average
            # total_logits_offsets = (expert_next_token_logits_1 + expert_next_token_logits_2 + expert_next_token_logits_3 - 3 * antiexpert_next_token_logits) /3

            # parameter
            # total_logits_offsets = (expert_next_token_logits_1 - antiexpert_next_token_logits) * 2 / 7 + \
            #            (expert_next_token_logits_2 - antiexpert_next_token_logits) * 2 / 7 + \
            #            (expert_next_token_logits_3 - antiexpert_next_token_logits) * 3 / 7

            # # prob voting
            # # 将 logits 转换为概率分布
            # expert_probs_1 = F.softmax(expert_next_token_logits_1, dim=-1)
            # expert_probs_2 = F.softmax(expert_next_token_logits_2, dim=-1)
            # expert_probs_3 = F.softmax(expert_next_token_logits_3, dim=-1)

            # # 使用概率分布进行比较
            # cond_1 = (expert_probs_1 + expert_probs_2) > (2 * expert_probs_3)
            # cond_2 = (expert_probs_1 + expert_probs_3) > (2 * expert_probs_2)
            # cond_3 = (expert_probs_2 + expert_probs_3) > (2 * expert_probs_1) 

            # # # 计算 next_token_logits
            # total_logits_offsets = torch.where(
            #     cond_1, (expert_next_token_logits_1 + expert_next_token_logits_2) / 2,
            #     torch.where(
            #         cond_2, (expert_next_token_logits_1 + expert_next_token_logits_3) / 2,
            #         torch.where(
            #             cond_3, (expert_next_token_logits_2 + expert_next_token_logits_3) / 2,
            #             (expert_next_token_logits_1 + expert_next_token_logits_2 + expert_next_token_logits_3) / 3
            #         )
            #     )
            # )

            # entropy voting
            # 计算每个专家的预测熵
            entropy_1 = -torch.sum(F.softmax(expert_next_token_logits_1, dim=-1) * F.log_softmax(expert_next_token_logits_1, dim=-1), dim=-1)
            entropy_2 = -torch.sum(F.softmax(expert_next_token_logits_2, dim=-1) * F.log_softmax(expert_next_token_logits_2, dim=-1), dim=-1)
            entropy_3 = -torch.sum(F.softmax(expert_next_token_logits_3, dim=-1) * F.log_softmax(expert_next_token_logits_3, dim=-1), dim=-1)

            # 根据熵计算权重
            total_entropy = entropy_1 + entropy_2 + entropy_3
            weight_1 = (1 / entropy_1) / total_entropy
            weight_2 = (1 / entropy_2) / total_entropy
            weight_3 = (1 / entropy_3) / total_entropy

            weight_1 = weight_1.unsqueeze(-1) 
            weight_2 = weight_2.unsqueeze(-1) 
            weight_3 = weight_3.unsqueeze(-1) 

            # 加权投票
            total_logits_offsets =  (expert_next_token_logits_1 - antiexpert_next_token_logits) * weight_1 + \
                                    (expert_next_token_logits_2 - antiexpert_next_token_logits) * weight_2 + \
                                    (expert_next_token_logits_3 - antiexpert_next_token_logits) * weight_3

            # # 置信度
            # # 计算每个专家的预测置信度
            # confidence_1 = torch.max(F.softmax(expert_next_token_logits_1, dim=-1), dim=-1).values
            # confidence_2 = torch.max(F.softmax(expert_next_token_logits_2, dim=-1), dim=-1).values
            # confidence_3 = torch.max(F.softmax(expert_next_token_logits_3, dim=-1), dim=-1).values

            # # 根据置信度计算权重
            # total_confidence = confidence_1 + confidence_2 + confidence_3
            # weight_1 = confidence_1 / total_confidence
            # weight_2 = confidence_2 / total_confidence
            # weight_3 = confidence_3 / total_confidence

            # # 加权投票
            # total_logits_offsets = (expert_next_token_logits_1 * weight_1 +
            #                         expert_next_token_logits_2 * weight_2 +
            #                         expert_next_token_logits_3 * weight_3 -
            #                         3 * antiexpert_next_token_logits)


            average_logits_offsets = self.alpha * total_logits_offsets

            next_token_logits = (
                base_next_token_logits + average_logits_offsets
            )

            # next_token_logits = (base_next_token_logits + self.alpha * (expert_next_token_logits - antiexpert_next_token_logits))

            # pre-process logits
            if logits_processor:
                next_token_logits = logits_processor(input_ids, next_token_logits)

            # warp logits
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            # if top_p < 1.0:
            #     next_token_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p)

            # decode
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            next_tokens = (
                next_tokens * unfinished_sequences +
                self.tokenizer.pad_token_id * (1 - unfinished_sequences)
            )

            if return_logits_for_analysis:
                next_token_logits_dict = {
                    'dexperts': next_token_logits,
                    'base': base_next_token_logits,
                    'expert_1': expert_next_token_logits_1,
                    'expert_2': expert_next_token_logits_2,
                    'expert_3': expert_next_token_logits_3,
                    'antiexpert': antiexpert_next_token_logits
                }
                analysis_data = self.update_analysis_data(analysis_data, next_tokens, next_token_logits_dict)

            # update model inputs for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            expert_input_ids = torch.cat([expert_input_ids, next_tokens[:, None]], dim=-1)

            # update kwargs
            base_kwargs = self._update_model_kwargs_for_generation(base_outputs, base_kwargs)
            expert_kwargs_1 = self._update_model_kwargs_for_generation(expert_outputs_1, expert_kwargs_1)
            expert_kwargs_2 = self._update_model_kwargs_for_generation(expert_outputs_2, expert_kwargs_2)
            expert_kwargs_3 = self._update_model_kwargs_for_generation(expert_outputs_3, expert_kwargs_3)

            antiexpert_kwargs = self._update_model_kwargs_for_generation(antiexpert_outputs, antiexpert_kwargs)

            # stopping criteria
            if stopping_criteria and stopping_criteria(input_ids, None):
                break

            # if eos_token was found in one sentence, set sentence to finished
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                break

        if return_logits_for_analysis:
            for k in analysis_data.keys():
                if k.startswith('logits'):
                    analysis_data[k] = torch.cat(analysis_data[k], dim=1)
            return input_ids, analysis_data

        return input_ids

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        # update past_key_values
        kwargs["past_key_values"] = outputs.past_key_values

        # update attention mask
        if "attention_mask" in kwargs:
            attention_mask = kwargs["attention_mask"]
            kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        return kwargs
