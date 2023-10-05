## 计算瑞雷波/勒夫波前三个阶次的相速度频散曲线(根据输入模型计算理论频散曲线)
import numpy as np
from disba import PhaseDispersion           #计算相速度频散
#from disba import GroupDispersion          #计算群速度频散
from disba._helpers import resample 

def get_velocity_model(H1, H2, H3, v1, v2, v3):
    velocity_model = np.array([
    [H1, 2*v1, v1, 2.00],
    [H2, 2*v2, v2, 2.00],
    [H3, 2*v3, v3, 2.00]
    ])

    ## 0.1.对原始速度模型进行重采样
    dz = 0.005

    velocity_model_thickness=velocity_model.T[0]
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


depth_end = 0.20

vv1 = np.arange(0.2, 0.4, 0.05)
vv2 = np.arange(0.4, 0.6, 0.05)
vv3 = np.arange(0.6, 0.7, 0.05)
HH1 = np.arange(0.03, 0.09, 0.01)
HH2 = np.arange(0.03, 0.09, 0.01)

dx = int(512/depth_end)

disp = np.zeros((len(HH1), len(HH2), len(vv1), len(vv2), len(vv3), 512))
v = np.zeros((len(HH1), len(HH2), len(vv1), len(vv2), len(vv3), 512))
for ii, h1 in enumerate(HH1):
    for jj, h2 in enumerate(HH2):
        for i, v1 in enumerate(vv1):
            for j, v2 in enumerate(vv2):
                for k, v3 in enumerate(vv3):
                    H3 = depth_end - h1 - h2
                    # thickness, Vp, Vs, density
                    # km, km/s, km/s, g/cm3

                    try:
                        disp[ii][jj][i][j][k] = get_velocity_model(round(h1,2), round(h2,2), round(H3,2), \
                                            round(v1,2), round(v2,2), round(v3,2))[1]
                    except :
                        print("***", h1, h2, H3, v1, v2, v3)
                        continue
                    v[ii][jj][i][j][k][:int(h1*dx)] = v1
                    v[ii][jj][i][j][k][int(h1*dx):int((h1+h2)*dx)] = v2
                    v[ii][jj][i][j][k][int((h1+h2)*dx):] = v3

v = v.reshape(-1, 512); disp = disp.reshape(-1, 512)
np.savez('disp.npz', disp=disp, v=v)