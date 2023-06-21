import pickle
import torch

def saveModel(model, optimizer, epochs, losses, save_path):

    # 定义字典，将模型、优化器和其他变量存储起来
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epochs,
        'loss': losses,
    }

    # 保存模型和优化器
    torch.save(checkpoint, save_path)

def loadModel(model, optimizer, load_path):
    # 加载模型和优化器
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs = checkpoint['epoch']
    losses = checkpoint['loss']

    return model, optimizer, epochs, losses

def saveDataset(train_iter, test_iter, train_save_path, test_save_path):
    
    # 将数据集保存为 pickle 文件
    with open(train_save_path, 'wb') as f:
        pickle.dump(train_iter.dataset, f)
    with open(test_save_path, 'wb') as f:
        pickle.dump(test_iter.dataset, f)

def loadDataset(train_load_path, test_load_path):
    # 从 pickle 文件中加载数据集
    with open(train_load_path, 'rb') as f:
        train_dataset = pickle.load(f)
    with open(test_load_path, 'rb') as f:
        test_dataset = pickle.load(f)

    return train_dataset, test_dataset