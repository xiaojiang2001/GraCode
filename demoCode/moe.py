import torch
import torch.nn as nn
import torch.nn.functional as F


# 专家网络
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x):
        return self.linear(x)

class MoELayer(nn.Module):
    def __init__(self, num_experts, in_features, out_features):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Linear(in_features, out_features) for _ in range(num_experts)])
        self.gate = nn.Linear(in_features, num_experts)
        
    def forward(self, x):
        gate_outputs = self.gate(x)
        gate_outputs = F.softmax(gate_outputs, dim=-1)

        # 2. 获取每个专家的输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  


        # 3. 门控值扩展维度以匹配专家输出
        gate_outputs = gate_outputs.unsqueeze(-1)   

        # 4. 专家输出与门控值相乘
        weighted_outputs = expert_outputs * gate_outputs

        # 5. 将所有专家的加权输出求和
        output = weighted_outputs.sum(dim=1)  

        # 获取每个专家的输出
        # expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # output = torch.bmm(gate_outputs.unsqueeze(1), expert_outputs).squeeze(1)
        # output = torch.sum(expert_outputs * gate_outputs.unsqueeze(-1), dim=-1)
        print("Gate weights:", gate_outputs.squeeze(-1)[0])  # 查看第一个样本的门控权重
        print("Expert outputs:", expert_outputs[0])          # 查看第一个样本的专家输出
        print("Weighted outputs:", weighted_outputs[0])      # 查看第一个样本的加权输出
        
        return output
    
# MoEAdapter：包含多个专家和门控网络，并在输出后添加激活函数
input_channels = 5
input_features = 3
num_experts = 4
batch_size = 10

model = MoELayer(num_experts, input_features, input_channels)
demo = torch.randn(batch_size, input_features)  # Batch size of 2

output = model(demo)
print("Output shape:", output.shape)  # Should be (batch_size, input_channels)
# Output shape: torch.Size([2, 5])
# Output: torch.Size([2, 5])    