from typing import Any
from sklearn.model_selection import train_test_split
import torch
  
class MkData():
    """
    Make the dataset.
    """
    def __init__(self):
        # self.sampleNum = sampleNum
        # self.layerNum = layerNum
        # self.num_h1 = num_h1
        # self.num_h2 = num_h2
        # self.dx = dx
        # self.depth_end = depth_end
        # self.freqs_end = freqs_end
        # self.HVSR = torch.zeros((4, 8, 2, self.num_h1, self.num_h2, self.sampleNum))
        # self.VVs = torch.zeros((4, 8, 2, self.num_h1, self.num_h2, self.sampleNum))
        pass

    def __call__(self, sampleNum=512, layerNum=3, num_h1=20, num_h2=20, dx=0.8, depth_end=200., freqs_end=50):
        self.sampleNum = sampleNum
        self.layerNum = layerNum
        self.num_h1 = num_h1
        self.num_h2 = num_h2
        self.dx = dx
        self.depth_end = depth_end
        self.freqs_end = freqs_end
        self.HVSR = torch.zeros((4, 8, 2, self.num_h1, self.num_h2, self.sampleNum))
        self.VVs = torch.zeros((4, 8, 2, self.num_h1, self.num_h2, self.sampleNum))

        train_iter, test_iter = self.getIter()

        return train_iter, test_iter


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
        a = torch.ones(len(freq))
        b = torch.ones(len(freq))
        
        for jm in range(len(vel)-1):
            a1 = den[jm]
            a2 = den[jm+1]
            alfa = (a1*vel[jm]*(1+1j*damp[jm])) / (a2*vel[jm+1]*(1+1j*damp[jm+1]))
            ksH = 2 * torch.pi * freq * thi[jm] / (vel[jm] + damp[jm]*1j*vel[jm])
            A = 0.5 * a * (1 + alfa) * torch.exp(1j*ksH) + 0.5 * b * (1 - alfa) * torch.exp(-1j*ksH)
            B = 0.5 * a * (1 - alfa) * torch.exp(1j*ksH) + 0.5 * b * (1 + alfa) * torch.exp(-1j*ksH)
            
            a = A
            b = B
        
        hvsr = torch.abs(1 / A)
        
        return hvsr

    def den(self, Vs):
        Den = torch.zeros(len(Vs))
        for i, vs in enumerate(Vs):
            Den[i] = 0.85 * vs**0.14
        return Den

    def damp(self, Vs):
        Damp = torch.zeros(len(Vs))
        for i, vs in enumerate(Vs):
            if vs < 1000:
                Damp[i] = 1/(2 * 0.06 * vs)
            elif vs < 2000:
                Damp[i] = 1/(2 * 0.04 * vs)   
            else:
                Damp[i] = 1/(2 * 0.16 * vs)
        return Damp

    def getHVSR(self, v1, v2, v3, num_h1, num_h2, dx, depth_end, sampleNum, HVSR, VVs, freqs_end):
        for h1 in range(0, num_h1,):
            for h2 in range(0, num_h2,):
                H1 = h1 * dx + 1
                H2 = h2 * dx + 1
                H3 = depth_end - H1 - H2
                H = [H1, H2, H3]
                Vs = [v1, v2, v3]
                Den = self.den(Vs)
                Damp = self.damp(Vs)

                depthmax = sum(H) + sum(H)*10./100

                #model_x, model_y = self.set_array(H, Vs, depthmax)

                freqs = torch.linspace(0, freqs_end, sampleNum)
                ## main function to get HVSR
                HVSR[int((v1-200)/50)][int((v2-400)/50)][int((v3-600)/50)][h1][h2] = self.calc_hvsr(Vs, H, Den, Damp, freqs)

                VVs[int((v1-200)/50)][int((v2-400)/50)][int((v3-600)/50)][h1][h2][0:h1*5] =  Vs[0]
                VVs[int((v1-200)/50)][int((v2-400)/50)][int((v3-600)/50)][h1][h2][h1*5:(h1+h2)*5] =  Vs[1]
                VVs[int((v1-200)/50)][int((v2-400)/50)][int((v3-600)/50)][h1][h2][(h1+h2)*5:] =  Vs[2]

        return HVSR, VVs

    def getIter(self):
        num_h1 = self.num_h1
        num_h2 = self.num_h2
        sampleNum = self.sampleNum
        dx = self.dx
        depth_end = self.depth_end
        HVSR = self.HVSR
        VVs = self.VVs
        freqs_end = self.freqs_end

        for v1 in range(200, 400, 50):
            for v2 in range(400, 800, 50):
                for v3 in range(600, 700, 50):
                    HVSR, VVs = self.getHVSR(v1, v2, v3, num_h1, num_h2, dx, depth_end, sampleNum, HVSR, VVs, freqs_end)

                    

        VVs = VVs.reshape(-1, sampleNum); HVSR = HVSR.reshape(-1, sampleNum)
        VVs = VVs.reshape(-1, 1, 1, sampleNum); HVSR = HVSR.reshape(-1, 1, 1, sampleNum)

        # 将HVSR制成训练集， VVs制成标签
        VVs = VVs / 100.

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(HVSR, VVs, test_size=0.3, random_state=42)
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)  
        test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

        return train_iter, test_iter