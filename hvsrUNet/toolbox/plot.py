import matplotlib.pyplot as plt
import os
import torch

def setArray(depth, freq, true_v, syn_v, X):
    """
    Set the x and y coordinates of the model layers.
    """

    # 创建子图
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle(f'H/V Spectra and Velocity Model (depth: 0-{depth[-1]}m, freq: 0-{freq[-1]}Hz)')
    ax1 = fig.add_subplot(1, 2, 1)

    # 绘制第一条曲线
    color = 'tab:red'
    ax1.set_xlabel('depth [m]')
    ax1.set_ylabel('true velocity [m/s]', color=color)
    ax1.plot(depth, true_v, label='true', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([80, 920])

    # 创建第二个坐标轴
    ax2 = ax1.twinx()

    # 绘制第二条曲线
    color = 'tab:green'
    ax2.set_ylabel('model velocity [m/s]', color=color)
    ax2.plot(depth, syn_v, label='synthetic', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([80, 920])

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.set_title('Velocity Model')

    # 创建子图
    ax3 = fig.add_subplot(1, 2, 2)
    ax3.set_xlabel('frequence [Hz]')
    ax3.set_ylabel('amplitude')
    ax3.plot(freq, X[0][0][0], label='HVSR', color='tab:blue')
    ax3.set_title('H/V Spectra')
    ax3.legend(loc='upper right')

    # 调整布局
    fig.tight_layout()

    # 显示图像
    plt.show()

def plotTest(model, test_iter, depth_end=200., freqs_end=10, style=False):
    X, y = next(iter(test_iter))
    yyy = model(X)
    j = 30

    true_v = 100*y[j][0][0].detach().numpy()
    syn_v = 100*yyy[j][0][0].detach().numpy()
    depth = torch.linspace(0, depth_end, len(true_v))
    freq = torch.linspace(0, freqs_end, len(true_v))

    if style == True:
        print(os.getcwd())
        # 获取当前模块的绝对路径
        module_path = os.path.abspath(__file__)

        mpl_path = os.path.join(os.path.dirname(module_path), 'tm.mplstyle')


        with plt.style.context(mpl_path):
            setArray(depth, freq, true_v, syn_v, X)
    else:
        with plt.style.context('ggplot'):
            setArray(depth, freq, true_v, syn_v, X)

    

def plotLoss(losses):
    with plt.style.context('ggplot'):
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(losses, label='loss', color='red')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_title('Loss Curve')
        ax.legend()
        plt.show()

def plotHVSR(X, freqs_end=10.):
    i = 0
    freq = torch.linspace(0, freqs_end, len(X))
    x = torch.linspace(0, 10, len(X[i, 0, 0, :]))
    with plt.style.context('ggplot'):
        plt.plot(x, X[i, 0, 0, :].numpy(), label='HVSR')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.title('HVSR')
        plt.show()

def plotModel(y, depth_end=200.):
    v = 100*y[0][0][0].detach().numpy()
    depth = torch.linspace(0, depth_end, len(v))

    with plt.style.context('ggplot'):
        plt.plot(depth, v, label='true', color='tab:red')
        plt.xlabel('depth [m]')
        plt.ylabel('true velocity [m/s]')
        plt.ylim([80, 920])
        plt.legend(loc='upper left')
        plt.title('Velocity Model')
        plt.show()