import torch
import torch.nn as nn
import torch.nn.functional as F


def train(model, criterion, optimizer, train_iter, num_epochs=20):
    # 加载数据集并进行训练
    losses = []

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_iter):
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'.\
                    format(epoch+1, num_epochs, i+1, len(train_iter), loss.item()))
        losses.append(loss.item())
    return model, losses
