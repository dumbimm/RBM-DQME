import torch
import numpy as np
from NADO_32 import NADO32
from NADO_64 import NADO
from hamsys64 import hsys
from read import ReadHamilton
from chainsampler import Chainsampler
import time


input_str = np.loadtxt("input",max_rows=1,dtype=np.str_)
case = input_str[0]
input_device = input_str[1]
device = torch.device(str(input_device))
print(f'device:{device}')
sys = hsys(ReadHamilton(),device)
input_para = np.loadtxt("input",skiprows=1,max_rows=3,dtype=np.int32)
input_t = np.loadtxt("input",skiprows=4,dtype=np.float64)
nhidden = input_para[0,0]
nauxillary = input_para[0,1]
nonmccut = input_para[1,0]
allcut = input_para[1,1]
chains_number = input_para[2,0]
N_start = input_para[2,1]
N_end = input_para[2,2]

nstate = 2*sys.nvar*sys.nspin + sys.nsgn*sys.nvar*sys.nspin*sys.ncor*sys.nalf
Ns = sys.nvar * sys.nspin
Nd = sys.nsgn * Ns * sys.nalf * sys.ncor
statescols = [i for i in range(3,3+nstate)]
table = np.loadtxt("table_cut3.data",usecols=(statescols),dtype=np.int8)
condition = np.sum(table[:,:Nd],axis=1)==0
table0 = table[condition][:,Nd:]

rho = NADO(sys.nvar,sys.nspin,sys.ncor,sys.nalf,sys.nsgn,nhidden,nauxillary,device,table0).to(device)
rho32 = NADO32(sys.nvar,sys.nspin,sys.ncor,sys.nalf,sys.nsgn,nhidden,nauxillary,device,table0).to(device)
print('rbm_vec_len: ',rho.rbm_to_vec().shape,'\n')

chainsampler_exact = Chainsampler(rule='exact',nstate=nstate,filename="table_cut"+str(nonmccut+1)+".data")
states = chainsampler_exact.sampling(rho)
print('states_exact',':',states.shape,states.dtype,flush=True)

sys.set_states_need(rho,states,cut=allcut)
print('states_exact_need',':',sys.states_need['states_need'].shape,'\n',flush=True)

if nonmccut < allcut:
    import os
    if not os.path.exists('state0_mc') :
        statescols = [i for i in range(3,3+nstate)]
        table = torch.tensor(np.loadtxt("table_cut"+str(allcut+1)+".data",usecols=statescols,dtype=np.int8),device=device)
        state0 = torch.zeros((chains_number,rho.nstate),dtype=torch.int8,device=device)
        for i in range(chains_number) :
            j = 0
            while j>=0 :
                k = np.random.randint(0,table.shape[0])
                if torch.sum(table[k][:rho.Nd])==3 :
                    state0[i] = table[k]
                    break
                j += 1
        torch.save(state0,'state0_mc')
    lamda = -3.
    t1 = time.time()
    chainsampler = Chainsampler(rule='mhmc',N_start=N_start,N_end=N_end,cut1=nonmccut,cut2=allcut,
        lmbda=lamda,choices=chains_number,nstate=nstate)
    states_mc = chainsampler.sampling(rho)
    t2 = time.time()
    print(f'mc_sampling: {t2-t1:.2f}')
    print('states_mc',':',states_mc.shape,flush=True)
    # torch.save(states_mc,'sampling_results_'+str(N_end-N_start)+'_mccut'+str(nonmccut))
    sys.set_states_need_mc(rho,states_mc,cut=allcut)
    print('states_mc_need',':',sys.states_need_mc[0]['states_need'].shape,'\n',flush=True)


