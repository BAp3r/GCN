import os
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import  random,jit,grad
from jax.example_libraries import stax
from sklearn.metrics import f1_score, roc_curve, auc


# 将节点标签从字符串类型转换为整型索引
def encode_labels(labels):
    classes = sorted(list(set(labels)))
    label2index = {label: idx for idx, label in enumerate(classes)}
    indices = [label2index[label] for label in labels]
    indices = np.array(indices, dtype=np.int32)
    return indices

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

# 归一化特征矩阵
def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

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
    
    return features, labels, adj

# 实现图卷积
class GraphConvolution:
    def __init__(self, feature_num, hidden_size, rng_key):
        self.W = random.normal(rng_key, (feature_num, hidden_size)) * jnp.sqrt(2.0 / feature_num)
        self.b = jnp.zeros(hidden_size)
        
    def __call__(self, x, adj):
        x = jnp.dot(x, self.W)
        x = jnp.dot(adj, x)
        return x + self.b

# 实现GCN模型
class GCN:
    def __init__(self, input_size, hidden_size, output_size, dropout_rate, rng_key):
        rng_key, subkey1, subkey2 = random.split(rng_key, 3)
        self.gc1 = GraphConvolution(input_size, hidden_size, subkey1)
        self.gc2 = GraphConvolution(hidden_size, output_size, subkey2)
        self.dropout_rate = dropout_rate

    def __call__(self, x, adj, train=True):
        x = self.gc1(x, adj)
        x = jax.nn.relu(x)
        if train:
            x = stax.dropout(x, rate=self.dropout_rate)
        x = self.gc2(x, adj)
        return jax.nn.log_softmax(x)

# 损失函数
def loss_fn(params, x, adj, y, train):
    logits = model(params, x, adj, train)
    return -jnp.mean(jnp.sum(logits * y, axis=1))

# 准确率计算
def accuracy(logits, labels):
    return jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))

# 初始化模型
key = random.PRNGKey(0)
input_size = 1433
hidden_size = 16
output_size = 7
dropout_rate = 0.5

# 模型参数
params = {
    'gc1': {
        'W': random.normal(key, (input_size, hidden_size)) * jnp.sqrt(2.0 / input_size),
        'b': jnp.zeros(hidden_size)
    },
    'gc2': {
        'W': random.normal(key, (hidden_size, output_size)) * jnp.sqrt(2.0 / hidden_size),
        'b': jnp.zeros(output_size)
    }
}

# 编译模型
model = GCN(input_size, hidden_size, output_size, dropout_rate, key)
loss_fn = jit(loss_fn)
grad_fn = jit(grad(loss_fn))

# 加载数据
data_path = './data/cora/'
features, labels, adj = load_cora_data(data_path)
features = jnp.array(features)
labels = jax.nn.one_hot(jnp.array(labels), output_size)
adj = jnp.array(adj)

# 划分训练和测试数据
train_size = int(0.5 * len(features))
x_train, x_test = features[:train_size], features[train_size:]
y_train, y_test = labels[:train_size], labels[train_size:]
adj_train, adj_test = adj, adj  # 对于整个图的训练和测试，邻接矩阵相同

# 训练模型
epochs = 1000
learning_rate = 0.01
for epoch in range(epochs):
    grads = grad_fn(params, x_train, adj_train, y_train, True)
    params['gc1']['W'] -= learning_rate * grads['gc1']['W']
    params['gc1']['b'] -= learning_rate * grads['gc1']['b']
    params['gc2']['W'] -= learning_rate * grads['gc2']['W']
    params['gc2']['b'] -= learning_rate * grads['gc2']['b']
    
    if epoch % 100 == 0:
        train_logits = model(params, x_train, adj_train, False)
        test_logits = model(params, x_test, adj_test, False)
        train_acc = accuracy(train_logits, y_train)
        test_acc = accuracy(test_logits, y_test)
        print(f'Epoch {epoch}: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

# 评估模型
test_logits = model(params, x_test, adj_test, False)
test_pred = jnp.argmax(test_logits, -1)
test_true = jnp.argmax(y_test, -1)
test_f1 = f1_score(test_true, test_pred, average='weighted')
print(f'F1 Score: {test_f1:.4f}')

# 绘制 ROC 曲线
test_probs = jax.nn.softmax(test_logits)
fpr, tpr, _ = roc_curve(test_true, test_probs[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
