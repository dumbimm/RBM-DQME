import torch
import numpy as np


def ReadHamilton() :
    hbar = 0.658211928
    nsgn = np.loadtxt("res_corr.data", max_rows=1, dtype = int) 
    nspin = np.loadtxt("res_corr.data", skiprows=1, max_rows=1, dtype = int)
    nvar2 = np.loadtxt("res_corr.data", skiprows=2, max_rows=1, dtype = int)
    nalf = np.loadtxt("res_corr.data", skiprows=3, max_rows=1, dtype = int)
    ncor = np.loadtxt("res_corr.data", skiprows=4, max_rows=1, dtype = int)

    nrho = np.loadtxt("ham_sys.data", max_rows=1, dtype = int)
    Hsys = np.loadtxt("ham_sys.data", skiprows=1, dtype = np.float64).reshape(nrho,nrho)/hbar

    c = np.loadtxt("c.data", dtype = np.float64).reshape(nvar2*nspin*nrho,nrho)

    cgama_real = np.loadtxt("res_corr.data", skiprows=5, usecols=10, dtype = np.float64)
    cgama_img = np.loadtxt("res_corr.data", skiprows=5, usecols=11, dtype = np.float64)
    cgama = np.zeros((cgama_real.shape), dtype=np.complex128)
    cgama = cgama_real + (1.j)*cgama_img
    gamma = cgama

    cb_real = np.loadtxt("res_corr.data", skiprows=5, usecols=6, dtype = np.float64)
    cb_img = np.loadtxt("res_corr.data", skiprows=5, usecols=7, dtype = np.float64)
    cb = np.zeros((cb_real.shape), dtype=np.complex128)
    cb = cb_real + (1.j)*cb_img
    eta = np.transpose(np.reshape(cb,(nsgn,-1)))

    return (Hsys,c,gamma,eta,nvar2,nspin,ncor,nalf,nsgn)
