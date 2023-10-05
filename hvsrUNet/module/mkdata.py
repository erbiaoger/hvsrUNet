from typing import Any
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
# from disba import PhaseDispersion           #计算相速度频散
#from disba import GroupDispersion          #计算群速度频散
# from disba._helpers import resample 

class MkData():
    """
    Make the dataset.
    """
    def __init__(self):
        pass

    def __call__(self, sampleNum=512, depth_end=200., freqs_end=50, vv=None, HH=None):
        self.sampleNum = sampleNum

        self.depth_end = depth_end
        self.freqs_end = freqs_end
        self.HVSR = np.zeros((4, 8, 2, self.num_h1, self.num_h2, self.sampleNum))
        self.VVs = np.zeros((4, 8, 2, self.num_h1, self.num_h2, self.sampleNum))
        self.vv1 = vv[0]
        vv2 = np.arange(400, 600, 50)
        vv3 = np.arange(600, 700, 50)
        HH1 = np.arange(30, 90, 4)
        HH2 = np.arange(30, 90, 4)
        HH3 = np.arange(30, 90, 4)
        return self.getIter()


    def set_array(self, thickness, velocity, depth_max):
        """
        Set the x and y coordinates of the model layers.

        Args:
            thickness (list): List of layer thicknesses.
            velocity (list): List of layer velocities.
            depth_max (float): Maximum depth.

        Returns:
            x (list): List of x-coordinates.
            y (list): List of y-coordinates.
        """
        if len(thickness) < len(velocity):
            thickness.append(0)
        
        j = 1
        x = [velocity[0]]
        y = [0]
        
        j += 1
        x.append(velocity[0])
        y.append(thickness[0])
        
        for i in range(1, len(velocity)):
            j += 1
            x.append(velocity[i])
            y.append(y[j-2])
            
            j += 1
            x.append(velocity[i])
            y.append(y[j-2] + thickness[i])
        
        y[-1] = depth_max
        
        return x, y

    def calc_hvsr(self, vel, thi, den, damp, freq):
        """
        Calculate the horizontal-to-vertical spectral ratio (HVSR).

        Args:
            vel (list): List of shear wave velocities of layers.
            thi (list): List of thicknesses of layers.
            den (list): List of densities of layers.
            damp (list): List of damping ratios of layers.
            freq (list): List of frequencies.

        Returns:
            hvsr (list): List of HVSR values.
        """
        a = np.ones(len(freq))
        b = np.ones(len(freq))
        
        for jm in range(len(vel)-1):
            a1 = den[jm]
            a2 = den[jm+1]
            alfa = (a1*vel[jm]*(1+1j*damp[jm])) / (a2*vel[jm+1]*(1+1j*damp[jm+1]))
            ksH = 2 * np.pi * freq * thi[jm] / (vel[jm] + damp[jm]*1j*vel[jm])
            A = 0.5 * a * (1 + alfa) * np.exp(1j*ksH) + 0.5 * b * (1 - alfa) * np.exp(-1j*ksH)
            B = 0.5 * a * (1 - alfa) * np.exp(1j*ksH) + 0.5 * b * (1 + alfa) * np.exp(-1j*ksH)
            
            a = A
            b = B
        
        hvsr = np.abs(1 / A)
        
        return hvsr

    def den(self, Vs):
        Den = np.zeros(len(Vs))
        for i, vs in enumerate(Vs):
            Den[i] = 0.85 * vs**0.14
        return Den

    def damp(self, Vs):
        Damp = np.zeros(len(Vs))
        for i, vs in enumerate(Vs):
            if vs < 1000:
                Damp[i] = 1/(2 * 0.06 * vs)
            elif vs < 2000:
                Damp[i] = 1/(2 * 0.04 * vs)   
            else:
                Damp[i] = 1/(2 * 0.16 * vs)
        return Damp
    

    def get_velocity_model(self, velocity_model):
        # velocity_model = np.array([
        # [H1/1000, 2*v1/1000, v1/1000, 2.00],
        # [H2/1000, 2*v2/1000, v2/1000, 2.00],
        # [H3/1000, 2*v3/1000, v3/1000, 2.00]
        # ])
        velocity_model = np.array(velocity_model)

        ## 0.1.对原始速度模型进行重采样
        dz = 0.005

        velocity_model_thickness=velocity_model[:, 0]
        velocity_model = resample(velocity_model_thickness.T, velocity_model, dz)
        velocity_model = np.array(velocity_model[1])
        velocity_model.T[0] = dz

        # Periods must be sorted starting with low periods
        t = np.logspace(-1.0, 1.0, 512)          #创建等比数列(周期)
        #t = np.linspace(0.0, 3.0, 100)
        # Compute the 3 first Rayleigh- and Love- wave modal dispersion curves 计R/L频散曲线
        # Fundamental mode corresponds to mode 0(基阶mode=0)
        pd = PhaseDispersion(*velocity_model.T)
        #pd = GroupDispersion(*velocity_model.T)
        #cpr = [pd(t, mode=i, wave="rayleigh") for i in range(3)]
        cpr = pd(t, mode=0, wave="rayleigh")
        #cpl = [pd(t, mode=i, wave="love") for i in range(3)]

        return cpr

    def getIter(self):

        sampleNum = self.sampleNum
        freqs_end = self.freqs_end
        depth_end = 200
        ddx = depth_end / sampleNum

        vv1 = self.vv1
        vv2 = self.vv2
        vv3 = self.vv3
        HH1 = self.HH1
        HH2 = self.HH2
        HH3 = self.HH3


        HVSR = np.zeros(len(HH1), len(HH2), len(vv1), len(vv2), len(vv3), sampleNum)
        VVs = np.zeros(len(HH1), len(HH2), len(vv1), len(vv2), len(vv3), sampleNum)
        disp = np.zeros(len(HH1), len(HH2), len(vv1), len(vv2), len(vv3), sampleNum)

        for ii, h1 in enumerate(HH1):
            for jj, h2 in enumerate(HH2):
                for kk, h3 in enumerate(HH3):
                    for i, v1 in enumerate(vv1):
                        for j, v2 in enumerate(vv2):
                            for k, v3 in enumerate(vv3):
                                h3 = depth_end - h1 - h2 
                                H = [h1, h2, h3]
                                Vs = [v1, v2, v3]
                                Den = self.den(Vs)
                                Damp = self.damp(Vs)
                                freqs = np.linspace(0, freqs_end, sampleNum)
                                try:
                                    # disp[ii][jj][i][j][k] = self.get_velocity_model(round(h1,2), round(h2,2), round(H3,2), \
                                    #                     round(v1,2), round(v2,2), round(v3,2))[1]   # 将h1保留两位小数
                                    HVSR[ii, jj, i, j, k, :] = self.calc_hvsr(Vs, H, Den, Damp, freqs)
                                except :
                                    print("***", h1, h2, h3, v1, v2, v3)
                                    continue

                                VVs[ii, jj, i, j, k, 0:int(h1/ddx)] =  v1
                                VVs[ii, jj, i, j, k, int(h1/ddx):int((h1+h2)/ddx)] =  v2
                                VVs[ii, jj, i, j, k, int((h1+h2)/ddx):] =  v3

        VVs = VVs.reshape(-1, sampleNum); HVSR = HVSR.reshape(-1, sampleNum)
        VVs = VVs.reshape(-1, 1, 1, sampleNum); HVSR = HVSR.reshape(-1, 1, 1, sampleNum)

        # 将HVSR制成训练集， VVs制成标签
        VVs = VVs / 1000.

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(HVSR, VVs, test_size=0.3, random_state=42)
        # train_dataset = np.utils.data.TensorDataset(X_train, y_train)
        # test_dataset = np.utils.data.TensorDataset(X_test, y_test)
        # train_iter = np.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)  
        # test_iter = np.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

        return X_train, X_test, y_train, y_test

if __name__ == '__main__':

    def generate_combinations_v(v, current_combination=[]):
        if not v:
            # 如果v为空列表，打印当前组合
            #print(*current_combination)
            V.append(current_combination)
            return
        else:
            # 递归生成组合
            for item in v[0]:
                generate_combinations_v(v[1:], current_combination + [item])

    def generate_combinations_h(h, current_combination=[]):
        if not h:
            # 如果v为空列表，打印当前组合
            #print(*current_combination)
            H.append(current_combination)
            return
        else:
            # 递归生成组合
            for item in h[0]:
                generate_combinations_h(h[1:], current_combination + [item])

    # 你的输入数据
    v1 = [200, 250, 300, 350, 400]
    v2 = [100, 120, 140]
    v3 = [300, 400, 500, 600]
    v4 = [200]
    h1 = [20, 30, 40, 50]
    h2 = [21, 31, 41, 51]
    h3 = [22, 32, 42, 52]
    h4 = [23]
    v = [v1, v2, v3, v4]
    h = [h1, h2, h3, h4]

    # v1 = [300, 400, 440, 460, 500]
    # v2 = [450, 500, 600]
    # v3 = [750, 800, 1000, 1100]
    # h1 = [30, 40, 50, 60]
    # h2 = [30, 40, 44, 58]
    # h3 = [44, 55, 58, 60]
    # v = [v1, v2, v3]
    # h = [h1, h2, h3]

    H = []
    V = []

    generate_combinations_v(v)
    generate_combinations_h(h)

    H = np.array(H)
    V = np.array(V)

    print('H.shape: ', H.shape, '\tV.shape: ', V.shape)

    velocity_model = []

    for HH in H:
        for VV in V:
            velocity_model1 = []
            for h, v in zip(HH, VV):
                velocity_model1.append([h/1000, 2*v/1000, v/1000, 2.00])
                
            velocity_model.append(velocity_model1)

    velocity_model = np.array(velocity_model)
    print('velocity_model.shape: ', velocity_model.shape)

    freqs_end = 50
    sampleNum = 512
    freqs = np.linspace(0, freqs_end, sampleNum)

    mk = MkData()

    disp = []
    HVSR = []
    VVs = []
    Depth = []

    for vel in velocity_model:
        Vs = vel[:, 2]*1000
        H = vel[:, 0]*1000
        Den = mk.den(Vs)
        Damp = mk.damp(Vs)
        depth_end = sum(H)
        ddx = depth_end / sampleNum
        try:
            # disp.append(mk.get_velocity_model(vel)[1])   # 将h1保留两位小数
            HVSR.append(mk.calc_hvsr(Vs, H, Den, Damp, freqs))
            
            h_last = 0
            vv = np.zeros(sampleNum)
            for v, h in zip(Vs, H):
                vv[int(h_last/ddx):int((h_last+h)/ddx)] = v
                h_last += h
            VVs.append(vv)
            Depth.append(depth_end)
        except :
            print("***", vel)
            continue


    HVSR = np.array(HVSR)
    # disp = np.array(disp)
    VVs = np.array(VVs)
    Depth = np.array(Depth)

    # print(HVSR.shape, disp.shape, VVs.shape, Depth.shape)


    # np.savez('datasets_4.npz', disp=disp, HVSR=HVSR, VVs=VVs, Depth=Depth)