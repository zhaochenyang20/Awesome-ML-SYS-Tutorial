# Wandba Weave Trace in VERL

## 环境

```bash
$WANDB_API_KEY
```

## bash

当前版本，获得trace需要配置的参数有：
``` cmd
+trainer.rollout_trace.backend=weave: only wandb weave is support in this PR. Leave the reset of trace tool to the community.
+trainer.rollout_trace.token2text=True: whether append decoded text in result of run method.
```
对于SGLang，仅在async rollout，multi-turn enabled时生效：
``` cmd
actor_rollout_ref.rollout.mode=async \
actor_rollout_ref.rollout.multi_turn.enable=true
```
## 数据集
数据集需增加一列agent_name，可以再
```python
def make_map_fn(split):
        def process_fn(example, idx):
            #...
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                # 增加一列
                "agent_name": "tool_agent",
            }
            return data

        return process_fn
```

## 复现脚本

```bash
# run on 8xH100
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='gsm8k_async_rl' \
    trainer.experiment_name='qwen2.5-3b_function_rm-gsm8k-sgl-multi-w-tool-verify-n16' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml" \
    trainer.total_epochs=15 \
    +trainer.rollout_trace.backend=weave \
    +trainer.rollout_trace.token2text=True \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.enable=true

```
数据集处理

```bash=
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/gsm8k")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "openai/gsm8k"
    dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = "Let's think step by step and output the final answer after `####`."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": (
                            "You are a math expert. You are given a question and you need to solve it step by step. "
                            "Reasoning step by step before any tool call. "
                            "You should use the `calc_gsm8k_reward` tool after step by step solving the question, "
                            "before generate final answer at least once and refine your answer if necessary. "
                            "Put your final answer in the format of `#### <answer>`."
                        ),
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "calc_gsm8k_reward": {
                            "create_kwargs": {"ground_truth": solution},
                            # "execute_kwargs": {},
                            # "calc_reward_kwargs": {},
                            # "release_kwargs": {},
                        },
                    },
                    "interaction_kwargs": {
                        "query": question,
                        "ground_truth": solution,
                    },
                },
                # new column for weave
                "agent_name": "tool_agent",
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)


```
