from transformers import BertModel
import torch
import torch.nn as nn
from torch import sigmoid
from tqdm import tqdm
import torch.nn.functional as F

class BertCls(nn.Module):
    def __init__(self, model_path, numeric_size, dense_hidden_size=32):
        super(BertCls, self).__init__()

        self.BERT = BertModel.from_pretrained(model_path)
        # self.fc = nn.Linear(256 + numeric_size, 1)
        # self.dropout = nn.Dropout(0.3)
        
        
        
        self.fc1 = nn.Linear(256, dense_hidden_size)
        self.fc2 = nn.Linear(numeric_size, dense_hidden_size)
        self.fc3 = nn.Linear(2 * dense_hidden_size, 1)
        self.dropout = nn.Dropout(0.5)
        
        self.bn1 = nn.BatchNorm1d(2 * dense_hidden_size)
        self.bn2 = nn.BatchNorm1d(dense_hidden_size)

    def forward(self, text, numeric):
        # text {'input_ids': 1, sentence_num, word_per_sentence, ...}
        text = {'input_ids': text['input_ids'][0, :, :],
                'token_type_ids': text['token_type_ids'][0, :, :],
                'attention_mask': text['attention_mask'][0, :, :]
                }
        # numeric (1, numeric_size)
        numeric = numeric[0, :].to(torch.float32)
        # (sentence_num, word_per_sentence) ----> (sentence_num, word_per_sentence, hidden_size)
        BERT_output = self.BERT(**text)
        
        pooler_output = BERT_output.pooler_output
        article_vector = torch.mean(pooler_output, dim=0)
        
        
        document_feature = self.fc1(article_vector)
        # document_feature = self.bn2(document_feature)
        document_feature = F.relu(document_feature)

        numeric_feature = self.fc2(numeric)
        # numeric_feature = self.bn2(numeric_feature)
        numeric_feature = F.relu(numeric_feature)

        feature = torch.concat([document_feature, numeric_feature])
        # scores = self.bn1(feature)
        scores = self.fc3(feature)
        scores = torch.sigmoid(scores)
        
        
        
        
        # # (hidden_size) ----> (hidden_size + numeric_size)
        # features = torch.concat([article_vector, numeric])
        # # (hidden_size + numeric_size) ----> 1
        # output = sigmoid(self.dropout(self.fc(features)))
        return scores


def train_one_epoch(net, optimizer, loss, train_loader, device):
    metrics = [0, 0, 0]
    net.train()
    t = tqdm(train_loader)
    for batch in t:
        # -------------------------------------------- #
        # 训练
        # -------------------------------------------- #
        text, numeric, y = [x.to(device) for x in batch]
        output = net(text, numeric)
        l = loss(output, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        # -------------------------------------------- #
        # 计算准确率, loss求和
        # -------------------------------------------- #
        res = 1 if output[0] >= 0.5 else 0
        if y == res:
            metrics[1] += 1
        metrics[0] += l.cpu().detach().numpy().item()
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
            output = net(text, numeric)
            l = loss(output, y)
            # -------------------------------------------- #
            # 计算准确率, loss求和
            # -------------------------------------------- #
            res = 1 if output[0] >= 0.5 else 0
            if y == res:
                metrics[1] += 1
            metrics[0] += l.cpu().detach().numpy().item()
            metrics[2] += 1
    return metrics

def train(net, train_loader, test_loader, lr, epochs, device, weight_path, logpath=''):
    # -------------------------------------------- #
    # 网络、优化器、loss
    # -------------------------------------------- #
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.BCELoss(reduction='sum')
    # -------------------------------------------- #
    # 记录训练loss acc, 测试 loss acc
    # -------------------------------------------- #
    train_history = []
    test_history = []
    for epoch in range(epochs):
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
        with open(logpath+'log_train.txt', mode='a') as f:
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
