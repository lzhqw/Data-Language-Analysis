from transformers import LongformerModel
import torch
import torch.nn as nn
from torch import sigmoid
from tqdm import tqdm


class BertCls(nn.Module):
    def __init__(self, model_path, numeric_size):
        super(BertCls, self).__init__()

        self.BERT = LongformerModel.from_pretrained(model_path)

        max_pos = 4032
        max_pos += 2
        current_max_pos, embed_size = self.BERT.embeddings.position_embeddings.weight.shape
        new_pos_embed = self.BERT.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
        k = 2
        step = current_max_pos - 2
        while k < max_pos - 1:
            new_pos_embed[k:(k + step)] = self.BERT.embeddings.position_embeddings.weight[2:]
            k += step
        self.BERT.embeddings.position_embeddings.weight.data = new_pos_embed

        self.fc = nn.Linear(256 + numeric_size, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, text, numeric):
        # print(text['input_ids'].shape)
        # numeric (1, numeric_size)
        numeric = numeric.to(torch.float32)
        # (sentence_num, word_per_sentence) ----> (sentence_num, word_per_sentence, hidden_size)
        BERT_output = self.BERT(**text)
        # print(BERT_output)
        pooler_output = BERT_output.pooler_output
        # print(pooler_output.shape)
        # print(numeric.shape)
        features = torch.concat([pooler_output, numeric], dim=1)
        # print(features.shape)
        # (hidden_size + numeric_size) ----> 1
        output = sigmoid(self.dropout(self.fc(features)))
        return output


def train_one_epoch(net, optimizer, loss, train_loader, device):
    metrics = [0, 0, 0]
    net.train()
    t = tqdm(train_loader)
    for batch in t:
        # -------------------------------------------- #
        # 训练
        # -------------------------------------------- #
        text = {'input_ids': batch['input_ids'].to(device), 'position_ids': torch.arange(4032).to(device)}
        numeric = batch['numeric'].to(device)
        y = batch['targets'].to(device)
        y = torch.unsqueeze(y, dim=1)
        output = net(text, numeric)
        # print(output.shape, y.shape)
        l = loss(output, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        # -------------------------------------------- #
        # 计算准确率, loss求和
        # -------------------------------------------- #
        for i in range(y.shape[0]):
            res = 1 if output[i, 0] >= 0.5 else 0
            if y[i, 0] == res:
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
            text = {'input_ids': batch['input_ids'].to(device), 'position_ids': torch.arange(4096).to(device)}
            numeric = batch['numeric'].to(device)
            y = batch['targets'].to(device)
            y = torch.unsqueeze(y, dim=1)
            output = net(text, numeric)
            l = loss(output, y)
            # -------------------------------------------- #
            # 计算准确率, loss求和
            # -------------------------------------------- #
            for i in range(y.shape[0]):
                res = 1 if output[i, 0] >= 0.5 else 0
                if y[i, 0] == res:
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
        with open(logpath + 'log_train.txt', mode='a') as f:
            f.write(f'{epoch},{train_loss},{train_acc}\n')
        f.close()

        if epoch % 5 == 0:
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
