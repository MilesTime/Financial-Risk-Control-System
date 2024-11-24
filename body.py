import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# LSTM模型定义
class CreditRiskLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(CreditRiskLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)

# 图神经网络（GNN）定义
class CreditRiskGNN(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(CreditRiskGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)

# 数据示例
edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)  # 图结构
x = torch.tensor([[1, 2], [2, 3], [3, 4]], dtype=torch.float)  # 节点特征

# 模型训练
gnn_model = CreditRiskGNN(num_features=2, hidden_channels=4)
output = gnn_model(x, edge_index)
print("风险预测:", output)
