# Morris-Lecar model implementation
from ODE_hodgkinhuxley import Parameters, M_inf, N_inf
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def dVdt(V, N, params):
    """Calculate V'."""
    gCa, gK, gL = params.gCa, params.gK, params.gL
    VCa, VK, VL = params.VCa, params.VK, params.VL
    I, C = params.I, params.C
    return (gCa * M_inf(V, params.V1, params.V2) * (VCa - V) + gK * N * (VK - V) + gL * (VL - V) + I) / C

def dNdt(V, N, params):
    """Calculate N'."""
    lambda_N_max = params.lambda_N_max
    V3, V4 = params.V3, params.V4
    return lambda_N_max * (N_inf(V, V3, V4) - N)

def morris_lecar(t, y, params):
    V, N = y
    dV = dVdt(V, N, params)
    dN = dNdt(V, N, params)
    return [dV, dN]

if __name__ == "__main__":
    params = Parameters( gL = 2, VL = - 50, VCa = 100, VK = -70, lambda_M_max = 1, lambda_N_max = 0.1, V1 = 0, V2 = 15, V3 = 10, V4 = 10, C = 20, gCa = 4, gK = 8)
    t_span = (0, 200)
    V0, N0 = -60, 0.5
    for I in [25, 100, 400]:
        params.I = I
        sol = solve_ivp(morris_lecar, t_span, [V0, N0], args=(params,), t_eval=np.linspace(*t_span, 1000))
        plt.plot(sol.t, sol.y[0], label=f'I={I}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.title('Morris-Lecar Model')
    plt.legend()
    plt.show()
