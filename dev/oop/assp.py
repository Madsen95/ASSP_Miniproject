#!/usr/bin/env python

"""
These classes are designed to implement Array Sensor Signal Processing Algorithms easily. 

First a signal model is needed to be implemented with its corresponding parameters. 
Then the beamforming method should be used. 
"""

import numpy as np
import matplotlib.pyplot as plt


class Signal1D():
    """
    1D signal model.
    """
    def __init__(self, s_var:list[float], thetas:list[float], N:int, delay = None, c=3e8, f0=2.4e9):
        """
        1D signal model.
        params:
            power (float) : variance of the random signal
            thetas (list[float]) : list of azimut angles in rad
            N (int) : number of observations (samples)
            c (float) : velocity of propagation
            f0 : carrier frequency
        """
        self.c = c
        self.f0 = f0
        self.wavelength = c / f0
        self.N = N
        self.thetas = thetas
        self.var = np.array([s_var]).T #* np.identity(len(s_var))
        self.M = len(self.thetas)
        self.fi = np.random.uniform(0, np.pi * 2, (self.M, N))
        self.s = np.sqrt(self.var) * (np.exp(1j * self.fi))
        if not delay == None:
            for i, d in enumerate(delay):
                self.s[i][:d] = 0
    
    def log(self):
        print("Generated signal:")
        print(f"\tshape: {self.s.shape}")
        print(f"\tsamples: {self.N}")
        print(f"\tazimut angles: {self.thetas}")

class Signal2D():
    """
    2D Signal model
    """
    def __init__(self, s_var:float, thetas:list[float], fis:list[float], N:int, c=3e8, f0=2.4e9):
        """
        2D Signal model
        params:
            s_var (float) : variance of the signal
            thetas (list[float]) : list of azimut angles in rad
            fis (list[float]) : list of elevation angles in rad
            N (int) : number of observations (samples)
            c (float) : velocty of propagation
            f0 : carrier frequency
        """
        self.c = c
        self.f0 = f0
        self.wavelength = c / f0
        self.N = N
        self.thetas = thetas
        self.fis = fis
        self.var = s_var
        self.M = min(len(self.thetas), len(self.fis))
        self.phi_rand = np.random.uniform(0, np.pi * 2, (self.M, N))
        self.s = np.sqrt(s_var) * np.exp(1j * self.phi_rand)

class ULA():
    """
    Universal Linear Array
    """

    def __init__(self, d, L, snr, signal : Signal1D):
        """
        Initializes the Universal Linear Array
        params:
            d (float): distance between the sensors
            L (int): dimension of the array (number of sensors)
            s (Signal1D): signal
        """
        self.d = d
        self.L = L
        self.A = None
        self.signal = signal
        self.s = signal.s
        self.snr = snr
        self.get_steering(self.signal)

    def log(self):
        print("Generated ULA:")
        print(f"\tsignal shape: {self.s.shape}")
        print(f"\tarray dimension: {self.L}")
        print(f"\tdistance between sensors: {self.d}")
        print(f"\tsteering matrix shape: {self.A.shape}")

    
    def get_steering(self, s : Signal1D):
        self.A = np.array([[np.exp(1j * 2 * np.pi * self.d / s.wavelength * l * np.cos(theta)) for l in range(self.L)] for theta in s.thetas]).T

    def clean_response(self):
        """
        Returns a non-noisy response of the array for a signal s. 
        x = A @ s, where A is the response matrix, and s is the signal 
        params: 
        returns:
            x : dim(L,1)
        """

        self.x = self.A @ self.s
        return self.x

    def noisy_response(self):
        """
        Returns a nosiy response of the array for a signal s. The noise is a Gaussian white noise. 
        params: 
            snr : Signal-to-noise-ratio in dB
        """
        self.get_steering(self.signal)
        snr_lin = 10.0 ** (self.snr / 10.0)
        self.x = self.A @ self.s
        x_var = self.x.var()
        n_var = x_var / snr_lin
        n = np.random.normal(0, np.sqrt(n_var*2.0)/2.0, size=(self.L, 2*self.signal.N)).view(complex)
        return self.x + n

class BarlettBeamformer():
    """
    Barlett beamformer. 
    """
    def __init__(self, M, resolution, ula: ULA ):
        """
        Barlett beamformer.
        params:
            M: number of expected inpinging waves
            resolution: resolution of the beamforming
            ula: the ULA we want to use the Barlett beamformer with
            
        """
        self.M = M
        self.ula = ula
        self.P = np.zeros(resolution, dtype=np.complex128)
        self.thetas = np.linspace(-np.pi, np.pi, resolution, endpoint=True)
        self.x = ula.noisy_response()
        R = self.x @ self.x.conj().T 
        a = np.zeros((ula.L, 1), dtype=np.complex128)
        for i, theta in enumerate(self.thetas):
            a[:,0] = np.array([np.exp(1j * 2 * np.pi * l * (ula.d / ula.signal.wavelength * np.cos(theta))) for l in range(ula.L)])
            self.P[i] = (a.conjugate().T @ R @ a) / np.linalg.norm(a) ** 4
        #self.P /= self.P.max()

class CaponBeamformer():
    """
    Capon beamformer
    """
    def __init__(self, M, resolution, ula: ULA):
        """
        Capon beamformer
        params: 
            M: number of expected impinging waves
            resolution: resolution of the beamforming
            ula: ULA
        """
        self.M = M
        self.ula = ula
        self.P = np.zeros(resolution, dtype=np.complex128)
        self.thetas = np.linspace(-np.pi, np.pi, resolution, endpoint=True)
        self.x = ula.noisy_response()
        R = self.x @ self.x.conj().T
        a = np.zeros((ula.L, 1), dtype=np.complex128)
        for i, theta in enumerate(self.thetas):
            a[:,0] = np.array([np.exp(1j * 2 * np.pi * l * (ula.d / ula.signal.wavelength * np.cos(theta))) for l in range(ula.L)])
            self.P[i] = 1 / (a.conjugate().T @ np.linalg.pinv(R) @ a)
        self.P /= self.P.max()

class MUSIC_1D():
    """
    1D Spectral MUSIC algorithm
    """
    def __init__(self, M, resolution, ula: ULA):
        """
        MUSIC algorithm
        params:
            M (int) : number of expected impinging waves
            resolution (int) : resolutino of the beamforming
            ula (ULA) : ULA we want to use the beamforming with
        """
        self.M = M
        self.ula = ula
        self.P = np.zeros(resolution, dtype=np.complex128)
        self.thetas = np.linspace(-np.pi, np.pi, resolution, endpoint=True)
        self.x = ula.noisy_response()
        R = self.x @ self.x.conj().T / ula.signal.N
        a = np.zeros((ula.L, 1), dtype=np.complex128)
        w, v = np.linalg.eig(R)
        i = np.argsort(w)
        v = v[:, i]
        U = v[:, 0:(ula.L - self.M)]
        for i, theta in enumerate(self.thetas):
            a[:,0] = np.array([np.exp(1j * 2 * np.pi * l * (ula.d / ula.signal.wavelength * np.cos(theta))) for l in range(ula.L)])
            self.P[i] = 1 / (a.conjugate().T @ U @ U.conjugate().T @ a)
        self.P /= self.P.max()

class URA():
    """
    Universal Rectangular Array
    """
    def __init__(self, dx, dy, Lx, Ly, snr, signal:Signal2D):
        """
        Universal Planar array
        params:
            dx (float) : distance between the sensors along x-axis (hint: wavelength / 2) 
            dy (float) : distnace between the sensors along y-axis (hint: wavelength / 2)
            Lx (int) : number of sensors (dimension) along x-axis
            Ly (int) : number of sensors (dimension) along y-axis
            snr (float) : signal to noise ratio
            signal (Signal) : impinging signal
        """
        self.dx = dx
        self.dy = dy
        self.Lx = Lx
        self.Ly = Ly
        self.L = self.Lx * self.Ly
        self.snr = snr
        self.signal = signal
        self.s = signal.s
        self.rs = np.zeros((3, self.Lx, self.Ly))
        self.get_steering()

    def get_steering(self):
        """
        Returns the steering matrix for 2D case
        """
        self.es = np.array([np.array([np.sin(theta) * np.cos(fi), np.sin(theta) * np.sin(fi), np.cos(theta)]) for theta, fi in zip(self.signal.thetas, self.signal.fis)])
        self.rs = self.get_sensor_positions().T
        self.A = np.array([[np.exp(1j * 2 * np.pi / self.signal.wavelength * e.T @ r) for r in self.rs] for e in self.es]).T


    def get_sensor_positions(self):
        for lx in range(self.Lx):
            for ly in range(self.Ly):
                self.rs[0, lx, ly] = self.dx * lx # x dimension of r_xy
                self.rs[1, lx, ly] = self.dy * ly # y dimension of r_xy
        self.rs = self.rs.reshape(3, self.Lx * self.Ly)
        return self.rs


    def get_noisy_response(self):
        self.x = self.A @ self.s
        self.x = self.x.reshape(self.L, self.signal.N)
        snr_lin = 10.0 ** (self.snr / 10.0)
        x_var = self.x.var()
        n_var = x_var / snr_lin
        n = np.random.normal(0, np.sqrt(n_var*2.0)/2.0, size=(self.Lx * self.Ly, 2*self.signal.N)).view(complex)
        n = n.reshape(100, self.signal.N)
        print(f'x shape:{self.x.shape}')
        self.x = self.x + n
        return self.x

class MUSIC_2D():
    """
    2D Spectral MUSIC algorithm
    """
    def __init__(self, M, resolution, ura : URA):
        """
        2D Spectral MUSIC algorithm
        params:
            M (int) : number of expected impinging waves
            resolution : resolution of the beamforming 
            ura : URA we want to use the beamforming with
        """
        self.M = M
        self.upa = ura
        self.resolution = resolution
        self.P = np.zeros(resolution, dtype=np.complex128)
        self.thetas = np.linspace(-np.pi, np.pi, resolution, endpoint=True)
        self.fis = np.linspace(-np.pi, np.pi, resolution, endpoint=True)
        self.x = ura.get_noisy_response()
        R = self.x @ self.x.conj().T / ura.signal.N
        a = np.zeros((ura.Lx, ura.Ly), dtype=np.complex128)
        w, v = np.linalg.eig(R)
        i = np.argsort(w)
        v = v[:, i]
        U = v[:, 0:(ura.L - self.M)]
        #self.A = np.array([[np.exp(1j * 2 * np.pi / upa.signal.wavelength * e.T @ r) for r in rs] for e in es])
        #print(f'U shape: {U.shape}')
        #print(f'A shape {self.A.shape}')
        #self.P = np.array([1 / (a.conjugate().T @ U @ U.conjugate().T @ a) for a in self.A])
        #print(f'P shape: {self.P.shape}')
        self.P = np.zeros((self.thetas.shape[0], self.fis.shape[0]), dtype=np.complex128)
        self.rs = ura.rs.T
        for i, theta in enumerate(self.thetas):
            for j, fi in enumerate(self.fis):
                e = np.array([np.sin(theta) * np.cos(fi), np.sin(theta) * np.sin(fi), np.cos(theta)])
                print(f'e shape: {e.shape}')
                print(f'r shape: {self.rs.shape}')
                a = np.exp(2j * np.pi * ura.signal.wavelength * e.T @ self.rs).T
                print(f'a shape: {a.shape}')
                self.P[i, j] = 1 / (a.conj().T @ U @ U.conj().T @ a)
                print(self.P[i,j].shape)


