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

def correct_format(text):
    if text.count('<stochastic>') == 1 and text.count('<deterministic>') == 1 and text.count('<answer>') == 1:
        if text.index('<stochastic>') < text.index('<deterministic>') < text.index('<answer>'):
            return True
    return  False
def compute_score(solution_str, ground_truth, format_score=0.05, score=1.0,**kwargs):
    print(solution_str)
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """

    format_correct = correct_format(solution_str)   
    if format_correct==False:
        return 0
    if format_correct==True:
        answer = solution_str.split('<answer>')[-1].strip()
        if answer == ground_truth:
            return score
        else:
            return format_score
