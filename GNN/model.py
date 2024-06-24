# main.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np


X = ['seq'] 
y = ['sst3']  

amino_acid_to_int = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                     'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}

X_numeric = [[amino_acid_to_int[aa] for aa in seq] for seq in X]

X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

def convert_to_pyg_data(X, y):
    data_list = []
    for sequence, label in zip(X, y):
        edge_index = torch.tensor([[i, i + 1] for i in range(len(sequence) - 1)], dtype=torch.long).t().contiguous()
        x = torch.tensor(sequence, dtype=torch.float).view(-1, 1)
        y = torch.tensor([label], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    return data_list

train_data = convert_to_pyg_data(X_train, y_train)
test_data = convert_to_pyg_data(X_test, y_test)

input_dim = 1  
hidden_dim = 64
output_dim = 3  
gnn_model = GNNModel(input_dim, hidden_dim, output_dim)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(gnn_model.parameters(), lr=0.001)

for epoch in range(20):
    gnn_model.train()
    for data in train_data:
        optimizer.zero_grad()
        outputs = gnn_model(data)
        loss = loss_function(outputs, data.y.view(-1))
        loss.backward()
        optimizer.step()

    gnn_model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_data:
            outputs = gnn_model(data)
            _, predicted = torch.max(outputs, 1)
            total += data.y.size(0)
            correct += (predicted == data.y.view(-1)).sum().item()

    accuracy = correct / total
    print(f"Epoch {epoch + 1}, Test Accuracy: {accuracy:.4f}")

torch.save(gnn_model.state_dict(), 'gnn_model.pth')
