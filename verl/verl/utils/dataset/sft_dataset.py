# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

import pandas as pd
import torch
from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset

    Arguments:
        config (OmegaConf): the data config
    """

    def __init__(self, parquet_files: str | ListConfig, tokenizer, config):
        prompt_key = config.get("prompt_key", "prompt")
        prompt_dict_keys = config.get("prompt_dict_keys", None)
        response_key = config.get("response_key", "response")
        response_dict_keys = config.get("response_dict_keys", None)
        truncation = config.get("truncation", "error")
        use_shm = config.get("use_shm", False)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.training_mode = config.get("training_mode",'all')# all or stoch or deter
        assert truncation in ["error", "left", "right"]
        self.truncation = truncation
        self.use_shm = use_shm
        self.ratio = config.get("ratio", [0,0,0.9,0.1,0.0])

        if not isinstance(parquet_files, ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key if isinstance(prompt_key, tuple | list) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, tuple | list) else [response_key]
        self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []
        self.response_dict_keys = response_dict_keys if response_dict_keys else []

        self.max_length = config.get("max_length", 2048)

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, pandas.core.series.Series | numpy.ndarray) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        self.prompts = self.dataframe[self.prompt_key]
        self.references = self.dataframe.get('sample_reference',[""]*len(self.dataframe))
        self.explorations = self.dataframe.get('exploration',[""]*len(self.dataframe))
        self.executions = self.dataframe.get('re_execution',[""]*len(self.dataframe))
        self.raw_executions = self.dataframe.get('execution',[""]*len(self.dataframe))
        self.type1 = self.dataframe.get('type1',[""]*len(self.dataframe))
        self.type2 = self.dataframe.get('type2',[""]*len(self.dataframe))   
        self.thinkings = self.dataframe.get('cot',[""]*len(self.dataframe))
        self.answers = self.dataframe.get('answer',[""]*len(self.dataframe))
        if isinstance(self.explorations, pd.DataFrame):
            self.explorations = self.explorations.squeeze()
        if isinstance(self.answers, pd.DataFrame):
            self.answers = self.answers.squeeze()
        if isinstance(self.executions, pd.DataFrame):
            self.executions = self.executions.squeeze()
            
        # self.references = self.references.to_list()
        # self.explorations = self.explorations.to_list()
        # self.executions = self.executions.to_list()
        # self.answers = self.answers.to_list()
        
        for key in self.prompt_dict_keys:
            # type(x): pandas.core.series.Series
            # type(x[0]): numpy.ndarray
            # type(x[0][0]): dict
            try:
                self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.prompts={self.prompts}")
                raise
        if isinstance(self.prompts, pd.DataFrame):
            self.prompts = self.prompts.squeeze()
        self.prompts = self.prompts.tolist()
        self.responses = self.dataframe.get(self.response_key, [""] * len(self.dataframe))
        for key in self.response_dict_keys:
            try:
                self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.responses={self.responses}")
                raise
        if isinstance(self.responses, pd.DataFrame):
            self.responses = self.responses.squeeze()
        self.responses = self.responses

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item,mode=None):
        item = min(item, len(self.prompts)-1)
        tokenizer = self.tokenizer
        prompt = self.prompts[item]
        response = self.responses[item]
        # exploration  = self.explorations[item]
        # execution  = self.executions[item]
        # answer = self.answers[item]
        thinking = self.thinkings[item][:self.max_length]
        # raw_execution = self.raw_executions[item]
        special_inserted_tokens = []
        
        last_newline = thinking.rfind('\n')
        if last_newline != -1:
            thinking = thinking[:last_newline]
        
        # reference = self.references[item]
        # 0.8,0.1,0.1,0,0
        # by default reference = execution in dataset
        # training_type=['1','2','3','4','5'][torch.multinomial(torch.tensor(self.ratio),1).item()]
        # if mode is not None:
        #     training_type=mode
        
        prompt_chat = [{"role": "user", "content": prompt}]
        # string
        prompt_chat_str = tokenizer.apply_chat_template(
            prompt_chat, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
        )
        # print(prompt_chat_str)
        # print("++++++++++++++++++++++++++++++++++++")
        
        # t1_begin = "<think>"
        
        # if t1_begin not in prompt_chat_str:
        #     # prompt_chat_str=prompt_chat_str+t1_begin
        
        # t1_end = "</think>"
        # t0_begin = "<EXPLORATION>"
        # t0_end = "</EXPLORATION>"
        # t2_begin = "<EXECUTION>"
        # t2_end = "</EXECUTION>"
        # t3_begin = "\nanswer:"
        # if training_type == '1':
        #     # 全部
        #     prompt_chat_str=prompt_chat_str+t1_begin
        #     response=exploration+t1_end+t2_begin+execution+t2_end+t3_begin+answer+tokenizer.eos_token
        # elif training_type == '2':
        #     # EXECUTION
        #     prompt_chat_str=prompt_chat_str+t2_begin
        #     response = reference+t2_end+t3_begin+answer+tokenizer.eos_token
        # elif training_type == '3':
        #     prompt_chat_str=prompt_chat_str+t1_begin+t1_end+t2_begin
        #     response = reference+t2_end+t3_begin+answer+tokenizer.eos_token
        # elif training_type == '4': # 计算KL
        #     prompt_chat_str=prompt_chat_str+t2_begin
        #     response = reference+t2_end+t3_begin+answer+tokenizer.eos_token
        # elif training_type == '5':
        #     prompt_chat_str=prompt_chat_str+t1_begin
        #     response = exploration+t1_end+t2_begin
        # pred_answer = execution.split("boxed{")[-1]
        # last_right_brace = pred_answer.rfind("}")
        # if last_right_brace != -1:
        #     pred_answer = pred_answer[:last_right_brace]
        # else:
        #     pred_answer = answer[:10]
            
        # execution = execution.replace("<think>","").replace("</think>","")
        # if training_type == '1':
        #     # 全部
        #     prompt_chat_str=prompt_chat_str+t0_begin
        #     response="I need to make a step-by-step plan.\n"+exploration+t2_begin+"I need to carefully and step-by-step execute this plan to get the answer.\n"+execution+t2_end+tokenizer.eos_token
        # if training_type == '2':
        #     prompt_chat_str=prompt_chat_str+t0_begin+"I need to make a step-by-step plan.\n1.Integrate and reason based on existing information."+t2_begin
        #     response="There is no explicit plan.\nThen I will reason step by step to get the answer.\n"+raw_execution+t2_end+tokenizer.eos_token
        # elif training_type == '3':
        #     prompt_chat_str=prompt_chat_str+t1_begin+t0_begin
        #     response=exploration+t0_end+t2_begin+execution+t2_end+tokenizer.eos_token
        # elif training_type == '4':
        #     prompt_chat_str=prompt_chat_str+t1_begin+t0_begin+t0_end
        #     response=t2_begin+execution+t2_end+tokenizer.eos_token
        
        # if training_type == '1':
        #     prompt_chat_str=prompt_chat_str+'<think>\n\n</think>\n\n'
        #     response+=tokenizer.eos_token
        
        # if training_type == '2':
        #     response = '<think>\n'+ thinking+'\n</think>\n'+response
        #     response+=tokenizer.eos_token
        
        # print(f"训练类型: {training_type}, prompt 长度: {len(prompt_chat_str)}, response 长度: {len(response)}")
        # print("prompt:", prompt_chat_str)
        # print("response:", response)
        # if training_type == '1':
            # response = self.type1[item] + "<EXECUTION>I need to carefully and step-by-step execute this plan to get the answer.\n"
        # elif training_type == '2':
            # response = self.type2[item]
            # response+=tokenizer.eos_token
            
        
        
        # print(prompt_chat_str)
        # print("========")
        # print(response)
        response_chat_str = response
        # print()
        
        # print(len(response_chat_str)+len(prompt_chat_str))
        
        # print(prompt_chat_str)
        # print("===================")
        # print(response_chat_str)
        # tokenize
        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False,padding_side="left")
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]
        response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False,padding_side="left")
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]

        # print(f"训练类型: {training_type}, prompt 长度: {prompt_ids.shape[0]}, response 长度: {response_ids.shape[0]}")
        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = (
                torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype)
                * self.tokenizer.pad_token_id
            )
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt for SFT.
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        # mask out the last token in response
        for token in special_inserted_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            # print(f"特殊标记 {token} 的 ID 是 {token_id}")
            # 使用 nonzero 替代 non
            token_pos = (input_ids == token_id).nonzero(as_tuple=True)
            
            # 如果没有找到匹配的 token_id，跳过
            if len(token_pos[0]) == 0:
                continue
            # print(f"找到特殊标记 {token} 在位置 {token_pos[0][0].item()}")
            
            # 修改 loss_mask，假设 token_pos[0][0] 是找到的第一个位置
            loss_mask[token_pos[0][0] - 1] = 0
                
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0
        special_masked_attention_mask = attention_mask.clone()
        special_tokens = []
        for token in special_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            print(token,f"的 ID 是 {token_id}")
            token_pos = (input_ids == token_id).nonzero(as_tuple=True)
            if len(token_pos[0]) == 0:
                continue
            special_masked_attention_mask[token_pos[0][0]] = 0
        
        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "special_attention_mask":special_masked_attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
        # if not (mode=='4'):
        #     data_ref = self.__getitem__(item,mode='4')
        #     data['input_ids_ref']=data_ref['input_ids']
        #     data['attention_mask_ref']=data_ref['attention_mask']
        #     data['special_attention_mask_ref']=data_ref['special_attention_mask']
        #     data['position_ids_ref']=data_ref['position_ids']
        #     data['loss_mask_ref']=data_ref['loss_mask']
        return data
    
# torchrun --nproc_per_node=8 eval.py --config-path config --config-name eval-qwen-no-think && torchrun --nproc_per_node=8 eval.py --config-path config --config-name eval-qwen-hint-thinking
