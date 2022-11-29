import numpy as np
import matplotlib.pyplot as plt

def getSubarray(dims, new_dims, offset=[0, 0, 0], spacing=1):

    idx_column = np.arange(0, dims[0], spacing).reshape(-1,1)
    idx_row = np.arange(0, dims[1], spacing)
    idx_freq = np.arange(offset[2], offset[2] + new_dims[2], 1)

    if (len(idx_column) < new_dims[0]) or (len(idx_row) < new_dims[1]):
        print('Problem in finding the subarray')
        exit()
    else:
        idx_column = idx_column[offset[0]:offset[0] + new_dims[0]]

    idx_array = np.zeros((new_dims[0] * new_dims[1], 1), dtype=int)

    for il2 in range(new_dims[1]):
        idx_array[il2 * new_dims[0]:(il2 + 1) * new_dims[0]] = idx_column + dims[0] * (il2 + offset[1]) * spacing

    return idx_array.reshape(-1), idx_freq
    
def delay_response_vector_naive(az, tau, r, f, wave):
    a = np.zeros(r.shape[1], dtype=complex)
    e = np.matrix([np.cos(az), np.sin(az)])

    for i in range(len(a)):
        a[i] = np.exp(2j * np.pi * wave * np.dot(e, r[:, i]))
    
    b = np.zeros(len(f), dtype=complex)
    for i in range(len(b)):
        b[i] = np.exp(-2j * np.pi * f[i] * tau)

    return np.kron(b, a)

def delay_response_vector(az, tau, r, f, wave):
    # Angle
    e = np.matrix([np.cos(az), np.sin(az)])
    a = np.exp(2j * np.pi * (1 / wave) * e @ r).T
    
    # Time delay
    b = np.exp(-2j * np.pi * f * tau)
    
    return np.kron(b, a)

def get_noise(x, SNR_dB=5):
    L, N = x.shape
    SNR = 10.0**(SNR_dB / 10.0) # Desired linear SNR
    xVar = x.var() # Power of signal
    nVar = xVar / SNR # Desired power of noise
    n = np.random.normal(0, np.sqrt(nVar*2.0)/2.0, size=(L, 2*N)).view(complex)
    
    #print('varX = ', xVar, 'varY = ', nVar)
    #print(10*np.log10(xVar / nVar), SNR_dB)
    
    return n

def spatialSmoothing(X, L, Ls, method=None):
    P1, P2, P3 = L - Ls + 1
    RF = np.zeros((np.prod(Ls), np.prod(Ls)), dtype=complex)

    for p1 in range(P1):
        for p2 in range(P2):
            for p3 in range(P3):
                
                idx_array, idx_n = getSubarray(L, Ls, offset=[p1, p2, p3])
                  
                x_sm = X[idx_array][:, idx_n].flatten(order='F')

                x_sm = x_sm.reshape(-1, 1)

                RF += x_sm @ x_sm.conj().T

    RF = RF / (P1 * P2 * P3)

    if method is None:
        return RF
    else:
        J = np.flipud(np.eye(np.prod(Ls)))
        RFB = 1/2 * (RF + J @ RF.conj().T @ J)
        return RFB

def plotter(az_search, tau_search, P, data, title, fname=None):
    # Plot spectrogram
    fig, ax = plt.subplots()
    im = ax.pcolormesh(np.rad2deg(az_search), tau_search, 10*np.log10(np.abs(P)))
    # Plot ground truth
    AoA = (data['smc_param'][0][0][1])*180/np.pi
    AoA[AoA < 0] = AoA[AoA < 0] + 360
    TDoA = (data['smc_param'][0][0][2])*(1/3e8) + np.abs(data['tau'][0])
    ax.scatter(AoA, TDoA, color='r', marker='o', facecolors='none', alpha=1/2)
    ax.set_xlabel('Azimuth of arival [$^\circ$]')
    ax.set_ylabel('Time-delay [s]')
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    if fname is not None:
        fig.savefig(fname, dpi=100)