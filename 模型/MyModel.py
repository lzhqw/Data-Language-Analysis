import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class MyModel(nn.Module):
    def __init__(self, vocab_size, embed_size, gru_hidden_size, num_layers,
                 numeric_size, fc_hidden_size):
        super(MyModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(input_size=embed_size,
                          hidden_size=gru_hidden_size,
                          num_layers=num_layers)
        self.fc1 = nn.Linear(numeric_size, fc_hidden_size)
        self.fc2 = nn.Linear(gru_hidden_size, fc_hidden_size)
        self.fc3 = nn.Linear(fc_hidden_size, 1)
        self.bn = nn.BatchNorm1d(fc_hidden_size)

    def forward(self, text, numeric):
        # (batch_size, seq_len) -----> (batch_size, seq_len, embed_size)
        # (batch_size, seq_len, embed_size) -----> (seq_len, batch_size, embed_size)
        embedded = self.embedding(text).permute(1, 0, 2)
        # (seq_len, batch_size, embed_size) -----> (seq_len, batch_size, hidden_size)
        output, state = self.gru(embedded)
        # (seq_len, batch_size, hidden_size) -----> (batch_size, hidden_size)
        text_features = output[-1, :, :]
        numeric_feature = self.fc1(numeric)
        numeric_feature = self.bn(numeric_feature)
        numeric_feature = F.relu(numeric_feature)

        document_feature = self.fc2(text_features)
        document_feature = self.bn(document_feature)
        document_feature = F.relu(document_feature)

        feature = numeric_feature + document_feature
        # feature = torch.concat([document_feature, numeric_feature], dim=1)
        score = self.fc3(feature)
        score = torch.sigmoid(score)
        return score


def train_one_epoch(net, optimizer, loss, train_loader, device):
    metrics = [0, 0, 0]
    net.train()
    t = tqdm(train_loader)
    for batch in t:
        # -------------------------------------------- #
        # 训练
        # -------------------------------------------- #
        text, numeric, y = [x.to(device) for x in batch]
        y = torch.unsqueeze(y, dim=1)
        scores = net(text, numeric)
        l = loss(scores, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        # -------------------------------------------- #
        # 计算准确率, loss求和
        # -------------------------------------------- #
        for i in range(y.shape[0]):
            res = 1 if scores[i, 0] >= 0.5 else 0
            if y[i, 0] == res:
                metrics[1] += 1
            metrics[0] += l.item()
            metrics[2] += 1
        t.set_postfix(train_loss=metrics[0] / metrics[2], train_acc=metrics[1] / metrics[2])
    return metrics


def val_one_epoch(net, loss, test_loader, device):
    net.eval()
    metrics = [0, 0, 0]
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # -------------------------------------------- #
            # 前向传播
            # -------------------------------------------- #
            text, numeric, y = [x.to(device) for x in batch]
            y = torch.unsqueeze(y, dim=1)
            scores = net(text, numeric)
            l = loss(scores, y)
            # -------------------------------------------- #
            # 计算准确率, loss求和
            # -------------------------------------------- #
            for i in range(y.shape[0]):
                res = 1 if scores[i, 0] >= 0.5 else 0
                if y[i, 0] == res:
                    metrics[1] += 1
                metrics[0] += l.item()
                metrics[2] += 1
    return metrics


def train(net, train_loader_params, test_loader, lr, epochs, device, weight_path, logpath=''):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    # -------------------------------------------- #
    # 网络、优化器、loss
    # -------------------------------------------- #
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.BCELoss()
    # -------------------------------------------- #
    # 记录训练loss acc, 测试 loss acc
    # -------------------------------------------- #
    train_history = []
    test_history = []
    for epoch in range(epochs):
        train_loader = DataLoader(**train_loader_params)
        metrics = train_one_epoch(net=net,
                                  optimizer=optimizer,
                                  loss=loss,
                                  device=device,
                                  train_loader=train_loader)
        # -------------------------------------------- #
        # 平均loss 平均 acc
        # -------------------------------------------- #
        train_loss = metrics[0] / metrics[2]
        train_acc = metrics[1] / metrics[2]
        train_history.append((train_loss, train_acc))
        print(f'epoch:{epoch} train_loss:{train_loss:.3f} train_acc:{train_acc}')
        with open(logpath + 'log_train.txt', mode='a') as f:
            f.write(f'{epoch},{train_loss},{train_acc}\n')
        f.close()

        if epoch % 1 == 0:
            metrics = val_one_epoch(net=net, test_loader=test_loader, loss=loss, device=device)
            # -------------------------------------------- #
            # 平均loss 平均 acc
            # -------------------------------------------- #
            test_loss = metrics[0] / metrics[2]
            test_acc = metrics[1] / metrics[2]
            test_history.append((test_loss, test_acc))
            print(f'epoch:{epoch} test_loss:{test_loss:.3f} test_acc:{test_acc}')
            with open(logpath + 'log_test.txt', mode='a') as f:
                f.write(f'{epoch},{test_loss},{test_acc}\n')
            f.close()
            torch.save(net.state_dict(),
                       '{}/eopch_{}_loss_{:.4f}_acc{}.pth'.format(weight_path, epoch, test_loss, test_acc))
    return train_history, test_history
