# Use ODE solver to solve Hodgkin-Huxley model for Morris-Lecar model
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def M_inf(V, V1, V2):
    return (1/2) * (1 + np.tanh((V - V1) / V2))

def N_inf(V, V3, V4):
    return (1/2) * (1 + np.tanh((V - V3) / V4))

def lambda_M(V, lambda_M_max, V1, V2):
    return lambda_M_max * np.cosh((V - V1) / (2 * V2))

def lambda_N(V, lambda_N_max, V3, V4):
    return lambda_N_max * np.cosh((V - V3) / (2 * V4))

def hodgkin_huxley(V, M, N, t, params):
    C, I, gCa, gK, gL, VCa, VK, VL, V1, V2, V3, V4, lambda_N_max, lambda_M_max = params.get_params()
    dVdt = (I- gL*(V - VL)-gCa*M*(V-VCa)-gK*N*(V-VK))/C
    dMdt = lambda_M(V, lambda_M_max, V1, V2) * (M_inf(V, V1, V2) - M)
    dNdt = lambda_N(V, lambda_N_max, V3, V4) * (N_inf(V, V3, V4) - N)
    return dVdt, dMdt, dNdt


# Parameters as a class to easily update and pass to the ODE solver

class Parameters:
    def __init__(self, C=20, I=40, gCa=1, gK=8, gL=2, VCa=50, VK=-70, VL=-50, V1=-1.2, V2=18, V3=-1, V4=14.5, lambda_N_max=1, lambda_M_max=1):
        self.C, self.I = C, I
        self.gCa, self.gK, self.gL = gCa, gK, gL
        self.VCa, self.VK, self.VL = VCa, VK, VL
        self.V1, self.V2 = V1, V2
        self.V3, self.V4 = V3, V4
        self.lambda_N_max, self.lambda_M_max = lambda_N_max, lambda_M_max
    def update(self, C=None, I=None, gCa=None, gK=None, gL=None, VCa=None, VK=None, VL=None, V1=None, V2=None, V3=None, V4=None, lambda_N_max=None, lambda_M_max=None):
        if C is not None:
            self.C = C
        if I is not None:
            self.I = I
        if gCa is not None:
            self.gCa = gCa
        if gK is not None:
            self.gK = gK
        if gL is not None:
            self.gL = gL
        if VCa is not None:
            self.VCa = VCa
        if VK is not None:
            self.VK = VK
        if VL is not None:
            self.VL = VL
        if V1 is not None:
            self.V1 = V1
        if V2 is not None:
            self.V2 = V2
        if V3 is not None:
            self.V3 = V3
        if V4 is not None:
            self.V4 = V4
        if lambda_N_max is not None:
            self.lambda_N_max = lambda_N_max
        if lambda_M_max is not None:
            self.lambda_M_max = lambda_M_max
    def get_params(self):
        return self.C, self.I, self.gCa, self.gK, self.gL, self.VCa, self.VK, self.VL, self.V1, self.V2, self.V3, self.V4, self.lambda_N_max, self.lambda_M_max


if __name__ == "__main__":
    params = Parameters(gL = 2, VL = - 50, VCa = 100, VK = -70, lambda_M_max = 1, lambda_N_max = 0.1, V1 = 0, V2 = 15, V3 = 10, V4 = 10, C = 20, gCa = 4, gK = 8)
    # Time span
    t_span = (0, 200)
    t_eval = np.linspace(*t_span, 1000)
    # Initial conditions
    V0 = -50
    M0 = M_inf(V0, params.V1, params.V2)
    N0 = N_inf(V0, params.V3, params.V4)
    # Solve ODE
    fig, ax = plt.subplots(1, 1, sharex=True)
    for I in [25, 100, 400]:
        params.update(I=I)
        sol = solve_ivp(lambda t, y: hodgkin_huxley(y[0], y[1], y[2], t, params), t_span, [V0, M0, N0], t_eval=t_eval)
    # Plot results


        ax.plot(sol.t, sol.y[0], label=f'V (I={I})')
        #ax[1].plot(sol.t, sol.y[1], label=f'M (I={I})')
        #ax[2].plot(sol.t, sol.y[2], label=f'N (I={I})')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (mV)')
    #ax[1].set_ylabel('M')
    #ax[2].set_ylabel('N')
    ax.legend(loc= 'upper right')
    #ax[1].legend(loc= 'upper right')
    #ax[2].legend(loc= 'upper right')
    plt.show()
