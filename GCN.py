# %%
import os
import sys
import time
import torch
from torch import nn
from torch.nn.parameter import Parameter
import math
import numpy as np
import scipy.sparse as sp
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, roc_curve, auc

# %%
# 将节点标签从字符串类型转换为整型索引
def encode_labels(labels):
    classes = sorted(list(set(labels)))
    label2index = {label: idx for idx, label in enumerate(classes)}
    indices = [label2index[label] for label in labels]
    indices = np.array(indices, dtype=np.int32)
    return indices

# %%
# 建立无向图，不关注谁引用谁，只关注是否有关系，只要有关系就是1
# 使得邻接矩阵是对称的
def build_symmetric_adj(edges, node_num):
    adj = np.zeros((node_num, node_num), dtype=np.float32)
    for i, j in edges:
        # 无向图，所以两个方向都要有边
        adj[i][j] = 1
        adj[j][i] = 1
    for i in range(node_num):
        adj[i][i] = 1 # 自己到自己的边
    return adj

# %%
# 归一化特征矩阵
def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

# %%
# 将邻接矩阵转换为稀疏矩阵
def adj_to_sparse_tensor(adj):
    adj = sp.coo_matrix(adj)
    indices = torch.from_numpy(
        np.vstack((adj.row, adj.col)).astype(np.int64)
    ).long()
    values = torch.from_numpy(adj.data).float()
    shape = torch.Size(adj.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# %%
# 读取Cora数据集
def load_cora_data(data_path):
    print('Loading cora data...')
    # 使用np.genfromtxt读取数据得到一个content数组
    content = np.genfromtxt(os.path.join(data_path, 'cora.content'), dtype=np.dtype(str))
    
    idx = content[:, 0].astype(np.int32) # 节点id
    features = content[:, 1:-1].astype(np.float32) # 节点的特征向量
    # 将节点标签从字符串类型转换为整型索引
    labels = encode_labels(content[:, -1]) # 节点标签
    node_num = len(idx)
    print(f"node_num: {node_num}")
    # 读取Cora的边数据建立邻接矩阵
    cites = np.genfromtxt(os.path.join(data_path, 'cora.cites'), dtype=np.int32)
    # 将节点的ID映射到索引
    idx_map = {j: i for i, j in enumerate(idx)}
    print(f"idx_map: {idx_map}")
    edges = [(idx_map[i], idx_map[j]) for i, j in cites] # 边的列表
    edges = np.array(edges, dtype=np.int32)
    print(f"edges_num: {len(edges)}")
    # 建立cora对应的无向图adj
    adj = build_symmetric_adj(edges, node_num)
    print(f"adj: {adj.shape}")
    
    features = normalize(features)
    adj = normalize(adj)
    # 转换为torch的tensor
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    adj = adj_to_sparse_tensor(adj)
    
    return features, labels, adj

# %%
# 实现图卷积
class GraphConvolution(nn.Module):
    def __init__(self, feature_num, hidden_size):
        super(GraphConvolution, self).__init__()
        # 定义权重参数
        self.w = Parameter(torch.FloatTensor(feature_num, hidden_size))
        # 定义偏置参数
        self.b = Parameter(torch.FloatTensor(hidden_size))
        # 初始化参数
        stdv = 1. / math.sqrt(self.w.size(0))
        self.w.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)
        
    # 传入特征和邻接矩阵，计算图卷积
    def forward(self, x, adj):
        # 计算图卷积
        x = torch.mm(x, self.w) # 计算x和w的矩阵乘法，线性变换
        output = torch.spmm(adj, x) # 计算adj和x的稀疏矩阵乘法
        return output + self.b

# %%
# 实现GCN模型
class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(GCN, self).__init__()
        # 定义两层图卷积
        self.gc1 = GraphConvolution(input_size, hidden_size) # 图卷积
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout) # dropout
        self.gc2 = GraphConvolution(hidden_size, output_size)
        
    def forward(self, x, adj):
        # 定义前向传播
        x = self.gc1(x, adj)
        x = self.relu(x)
        x = self.drop(x)
        x = self.gc2(x, adj)
        return x

# %%
from torch import optim

def plot_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()

# %%
def plot_accuracy(accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.show()

# %%
def plot_f1_score(f1_scores):
    plt.figure(figsize=(10, 5))
    plt.plot(f1_scores, label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Over Epochs')
    plt.legend()
    plt.show()

# %%
def plot_roc_curve(roc_auc, roc_fpr, roc_tpr):
    plt.figure(figsize=(10, 5))
    plt.plot(roc_fpr, roc_tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

# %%
if __name__ == '__main__':
    # 定义当前设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    # 读取数据，得到样本的特征，标签和邻接矩阵
    data_path = './data/cora/'
    features, labels, adj = load_cora_data(data_path)
    print(f"features: {features.shape}")
    print(f"labels: {labels.shape}")
    print(f"adj: {adj.shape}")
    
    features = features.to(device)
    labels = labels.to(device)
    adj = adj.to(device)
    assert len(features) == len(labels)
    assert len(features) == len(adj)
    
    sample_num = features.shape[0]
    # 使用50%的数据作为训练集
    train_num = int(sample_num * 0.5)
    # 使用剩下的数据作为验证集
    test_num = sample_num - train_num
    print(f"train_num: {train_num}")
    print(f"test_num: {test_num}")
    
    features_num = features.shape[1]
    hidden_size = 16
    class_num = labels.max().item() + 1 # 样本的类别数
    dropout = 0.5
    print(f"features_num: {features_num}")
    print(f"hidden_size: {hidden_size}")
    print(f"class_num: {class_num}")
    
    # 创建GCN模型
    model = GCN(features_num, hidden_size, class_num, dropout).to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 创建 TensorBoard 记录器
    writer = SummaryWriter(log_dir='./logs')
    
    n_epochs = 1000
    losses = []
    accuracies = []
    f1_scores = []
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        outputs = model(features, adj)
        loss = criterion(outputs[:train_num], labels[:train_num])
        loss.backward()
        optimizer.step()
        
        # 记录损失到 TensorBoard
        writer.add_scalar('Loss/train', loss.item(), epoch)
        
        # 计算准确率和F1得分
        model.eval()
        with torch.no_grad():
            outputs = model(features, adj)
            predicted = torch.argmax(outputs[train_num:], dim=1)
            correct = (predicted == labels[train_num:]).sum().item()
            accuracy = 100 * correct / test_num
            f1 = f1_score(labels[train_num:].cpu(), predicted.cpu(), average='weighted')
        
        model.train()
        
        # 记录准确率和F1得分
        writer.add_scalar('Accuracy/train', accuracy, epoch)
        writer.add_scalar('F1_Score/train', f1, epoch)
        
        losses.append(loss.item())
        accuracies.append(accuracy)
        f1_scores.append(f1)
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{n_epochs} Loss: {loss.item()} Accuracy: {accuracy:.2f}% F1 Score: {f1:.2f}')
    
    # 保存模型权重
    torch.save(model.state_dict(), 'gcn_model.pth')
    
    # 计算ROC曲线
    with torch.no_grad():
        outputs = model(features, adj)
        probabilities = torch.softmax(outputs[train_num:], dim=1)
        fpr, tpr, _ = roc_curve(labels[train_num:].cpu().numpy(), probabilities[:, 1].cpu().numpy(), pos_label=1)
        roc_auc = auc(fpr, tpr)
    
    writer.close()
    
    # 分别绘制所有指标
    plot_loss(losses)
    plot_accuracy(accuracies)
    plot_f1_score(f1_scores)
    plot_roc_curve(roc_auc, fpr, tpr)

    print(f'Accuracy: {accuracy:.2f}%')

# %%
# 加载权重文件的测试函数
def test_model(model_path, data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features, labels, adj = load_cora_data(data_path)
    
    features = features.to(device)
    labels = labels.to(device)
    adj = adj.to(device)
    
    features_num = features.shape[1]
    hidden_size = 16
    class_num = labels.max().item() + 1
    dropout = 0.5
    
    model = GCN(features_num, hidden_size, class_num, dropout).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        outputs = model(features, adj)
        predicted = torch.argmax(outputs, dim=1)
        accuracy = (predicted == labels).sum().item() / len(labels)
    
    print(f'Test Accuracy: {accuracy:.2f}%')

# %%
# 调用测试函数
test_model('gcn_model.pth', './data/cora/')


