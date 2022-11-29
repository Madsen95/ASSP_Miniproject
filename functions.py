import numpy as np
import matplotlib.pyplot as plt

def getSubarray(dims, new_dims, offset=[0, 0, 0], spacing=1):
    """
    Generate a sub-array, from original array to new dimensions. An offset
    and spacing can be set to change the positions and spacing between array
    elements. 

    Args:
        dims (list): Original dimension [Nrow, Ncol, Nsamples]
        new_dims (list): New dimension [L1, L2, L3]
        offset (list): Position of first element [x, y, sample]
        spacing (int): Spacing between selected elements
    Returns:
        idx_array (np.array): Indexation array for sub-array elements
        idx_n (np.array): Indexation array for sub-samples
    """
    idx_column = np.arange(0, dims[0], spacing).reshape(-1,1)
    idx_row = np.arange(0, dims[1], spacing)
    idx_n = np.arange(offset[2], offset[2] + new_dims[2], 1)

    if (len(idx_column) < new_dims[0]) or (len(idx_row) < new_dims[1]):
        print('Problem in finding the subarray')
        exit()
    else:
        idx_column = idx_column[offset[0]:offset[0] + new_dims[0]]

    idx_array = np.zeros((new_dims[0] * new_dims[1], 1), dtype=int)

    for il2 in range(new_dims[1]):
        idx_array[il2 * new_dims[0]:(il2 + 1) * new_dims[0]] = idx_column + dims[0] * (il2 + offset[1]) * spacing

    return idx_array.reshape(-1), idx_n
    
def delay_response_vector_naive(az, tau, r, f, wave):
    """
    Compute delay response vector mu, as a function of azimuth of
    arrival and time-delay. This function is implemented in a naive
    manner using for-loops.

    Args:
        az (float): Azimuth search direction
        tau (float): Time-delay search direction
        r (np.array): Array element positions
        f (np.array): Frequency response
        wave (float): Wavelength
    Returns:
        mu (np.array): Delay response vector
    """
    a = np.zeros(r.shape[1], dtype=complex)
    e = np.matrix([np.cos(az), np.sin(az)])

    for i in range(len(a)):
        a[i] = np.exp(2j * np.pi * wave * np.dot(e, r[:, i]))
    
    b = np.zeros(len(f), dtype=complex)
    for i in range(len(b)):
        b[i] = np.exp(-2j * np.pi * f[i] * tau)

    return np.kron(b, a)

def delay_response_vector(az, tau, r, f, wave):
    """
    Compute delay response vector mu, as a function of azimuth of
    arrival and time-delay. This function is implemented in a
    vectorized manner using numpy.

    Args:
        az (float): Azimuth search direction
        tau (float): Time-delay search direction
        r (np.array): Array element positions
        f (np.array): Frequency response
        wave (float): Wavelength
    Returns:
        mu (np.array): Delay response vector
    """
    # Angle
    e = np.matrix([np.cos(az), np.sin(az)])
    a = np.exp(2j * np.pi * (1 / wave) * e @ r).T
    
    # Time delay
    b = np.exp(-2j * np.pi * f * tau)
    
    return np.kron(b, a)

def get_noise(x, SNR_dB=-10):
    """
    Generate samples of noise for a desired SNR in dB, where only
    the noise samples will be returned. Thus the output of this
    function must be added to the desired signal.

    Args:
        x (np.array): Input data
        SNR_dB (float): Desired amount of SNR in dB
    Returns:
        n (np.array): Noise array
    """
    L, N = x.shape
    SNR = 10.0**(SNR_dB / 10.0) # Desired linear SNR
    xVar = x.var() # Power of signal
    nVar = xVar / SNR # Desired power of noise
    n = np.random.normal(0, np.sqrt(nVar*2.0)/2.0, size=(L, 2*N)).view(complex)
    
    #print('varX = ', xVar, 'varY = ', nVar)
    #print(10*np.log10(xVar / nVar), SNR_dB)
    
    return n

def spatialSmoothing(X, L, Ls, method=None):
    """
    Given some data, perform spatial smoothing to obtain a full-rank
    covariance matrix for beamformer calculations. 

    Args:
        X (float): Input data
        L (np.array): Sub-array dimensions [L1, L2, L3]
        Ls (np.array): Smoothing array dimensions [Ls1, Ls2, Ls3]
        method (string): If not none, do forward-backward spatial smoothing
    Returns:
        R (np.array): Spatial smoothed covariance matrix
    """
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
    """
    Simple plotter function to illustrate results.

    Args:
        az_search (np.array): Azimuth search direction
        tau_search (np.array): Time-delay search direction
        P (np.array): Beamforming spectrum
        data (struct): Data provided for the miniproject
        title (string): Plot title
        fname (string): If set, a file with this name will be saved
    """
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