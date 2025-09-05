#!/usr/bin/env python
# coding=utf-8
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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

import math

import torch
import torch.nn as nn

class MatLoRA(nn.Module):
    def __init__(
        self,
        maximum_rank: int = 1,
    ):
        '''
        maximum_rank: maximum rank of the input matrix
        '''
        super(MatLoRA, self).__init__()
        self.maximum_rank = maximum_rank
        self.current_rank = maximum_rank

    def get_dimension(self):
        return self.maximum_rank
    
    def get_rank(self):
        return self.current_rank
    
    def set_rank(self, rank):
        self.current_rank = max(0, min(rank, self.get_dimension())) + 1 # ensure rank in [1, maximum_rank]

    def forward(self, inputs):
        
        return inputs[:,:self.get_rank()] * math.sqrt(self.get_dimension()/(self.get_rank())) 