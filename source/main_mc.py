import torch
import numpy as np
import sys as sys_sys
import time

from minresQLP_torch import MinresQLP_torch_64
from minresQLP import MinresQLP
from operator_print import operator_print
from init import sys,rho,rho32,device,input_t,states,states_mc,chainsampler,allcut,case

from scipy.special import comb
from scipy.integrate import RK45
import os


count = int(input_t[1])
rho.load_state_dict(torch.load('para'+str(count),map_location=device)) 
rho32.load_state_dict(torch.load('para'+str(count),map_location=device)) 

m = rho.nstate
m1 = rho.Ns
n = rho.nparameters


def get_last_line(inputfile):
    filesize = os.path.getsize(inputfile)
    blocksize = 200
    dat_file = open(inputfile, 'rb')
    last_line = ""
    if filesize > blocksize:
        maxseekpoint = (filesize // blocksize)
        dat_file.seek((maxseekpoint - 1) * blocksize)
    elif filesize:
        dat_file.seek(0, 0)
    lines = dat_file.readlines()
    if lines:
        last_line = lines[-1].strip()
    dat_file.close()
    return last_line

class fun() :
    def __init__(self,states,states_mc,cut1,cut2,lamda) :
        self.count_in_fun = 0
        self.time_list = [0]
        self.states = states
        self.states_mc = states_mc
        self.choices_mc = states_mc.shape[0]
        self.mc_cut = cut1
        self.all_cut = cut2
        self.lamda = lamda

    def coefficient(self,rho,rho32,sys,lmbda=0.) :
        states = self.states
        y = states[:,0:rho.Nd]
        weight = torch.exp(lmbda*torch.sum(y,dim=1)).view(-1,1)
        t1 = time.time()
        gradients = rho32.Gradients_t(states).to(torch.complex128)
        t11 = time.time()
        with torch.no_grad():
            rho_states = rho.States(states).view(-1,1)
            t12 = time.time()
            Llocals = sys.Lforward_nonmc(rho,loss=False).view(-1,1)
        t2 = time.time()
        F = torch.sum(torch.real(torch.conj(gradients)*Llocals)/weight,dim=0)
        L2 = torch.sum(torch.real(torch.conj(Llocals)*Llocals)/weight)
        Z = torch.sum(torch.real(torch.conj(rho_states)*rho_states)/weight)

        t3 = time.time()
        S = torch.real(torch.matmul(torch.conj(gradients).T,gradients))

        rho_0 = rho.rho_0().detach() 
        Z0 = torch.real(torch.trace(rho_0))**2
        t4 = time.time()

        print(f'time_nonmc: grad:{t11-t1:.2e} Lloc:{t2-t12:.2e} S:{t4-t3:.2e}',flush=True)
        print(f'L2_nonmc:{L2/Z:.5e}, Z_nonmc:{Z:.5e}',flush=True)
        return S, F, L2, Z, Z0, rho_0

    def coefficient_mc(self,rho,rho32,sys,lmbda=0.) :
        F = torch.Tensor(self.choices_mc,rho.nparameters).to(torch.float64).to(rho.bs.device)
        S = torch.zeros(self.choices_mc,rho.nparameters,rho.nparameters).to(torch.float64).to(rho.bs.device)
        L2 = torch.Tensor(self.choices_mc).to(torch.float64).to(rho.bs.device)
        Z = torch.Tensor(self.choices_mc).to(torch.float64).to(rho.bs.device)
        
        for j in range(0,self.choices_mc):
            states = self.states_mc[j]
            N = states.shape[0]
            y = states[:,0:rho.Nd]
            weight = torch.exp(lmbda*torch.sum(y,dim=1)).view(-1,1)
            t1 = time.time()
            gradients = rho32.Gradients_t(states).to(torch.complex128)
            t11 = time.time()
            with torch.no_grad():
                rho_states = rho.States(states).view(-1,1)
                t12 = time.time()
                Llocals = sys.Lforward_mc(rho,loss=False,choices=j).view(-1,1)
            t2 = time.time()

            F[j] = torch.sum(torch.real(torch.conj(gradients)*Llocals)/weight,dim=0)
            L2[j] = torch.sum(torch.real(torch.conj(Llocals)*Llocals)/weight)
            Z[j] = torch.sum(torch.real(torch.conj(rho_states)*rho_states)/weight)
            L2[j] = L2[j]/N
            Z[j] = Z[j]/N
            F[j] = F[j]/N

            t3 = time.time()
            S[j] = torch.real(torch.matmul(torch.conj(gradients).T,gradients/weight))
            S[j] = S[j]/N
            t4 = time.time()
        
        F = torch.mean(F,dim=0)
        S = torch.mean(S,dim=0)
        L2_std = torch.std(L2)
        L2 = torch.mean(L2)
        Z_std = torch.std(Z)
        Z = torch.mean(Z)

        nchain = self.choices_mc
        print(f'time_mc: grad:{(t11-t1)*nchain:.2e} Lloc:{(t2-t12)*nchain:.2e} S:{(t4-t3)*nchain:.2e}',flush=True)
        print(f'L2_mc: {L2:.5e}, Z_mc:{Z:.5e}; L2_std:{L2_std:.2e}, Z_std:{Z_std:.2e}')
        return S, F, L2, Z

    def coefficient_mix(self,rho,rho32,sys,lmbda=-3.) :
        S_0,F_0,L2_0,Z_0,Z0,rho_0_torch = self.coefficient(rho,rho32,sys,lmbda=0.)
        S_mc,F_mc,L2_mc,Z_mc = self.coefficient_mc(rho,rho32,sys,lmbda=lmbda)

        a = np.exp(lmbda)
        sum_weight = 0
        for i in range(self.mc_cut+1,self.all_cut+1):
            sum_weight += 50*rho.Nv**3*rho.Nb*rho.M**3*a**i
        S = S_0 + S_mc*sum_weight
        F = F_0 + F_mc*sum_weight
        L2 = L2_0 + L2_mc*sum_weight
        Z = Z_0 + Z_mc*sum_weight
        print(f'L2: {L2/Z0},  Z:{Z}',flush=True)
        return S/Z0, F/Z0, L2/Z0, Z, Z0, rho_0_torch

    def fun(self,t,parameters_numpy) : 
        rho.vec_to_rbm(torch.from_numpy(parameters_numpy).to(device))
        rho32.vec_to_rbm(torch.Tensor(parameters_numpy).to(device))

        t1 = time.time()
        S,F,L2,Z,Z0,rho_0_torch = self.coefficient_mix(rho,rho32,sys,lmbda=self.lamda)

        t2 = time.time()
        if F.is_cuda :
            g = MinresQLP_torch_64(S,F,50*pow(10,-13),50000)[0]
            g0 = g.to('cpu').numpy()
        else :
            S0,F0 = S.to('cpu').numpy(),F.to('cpu').numpy()
            g0 = MinresQLP(S0,F0,80*pow(10,-13),30000)[0]
            g = torch.tensor(g0,dtype=torch.float64,device=device)
        t3 = time.time()

        gsg = g.matmul(S.matmul(g)-2*F)
        ldaggerl_unif = L2
        n_up, n_down = operator_print.occupation(rho_0_torch)
        trace = torch.real(torch.trace(rho_0_torch))

        self.time_list.append(time.time())

        print(f'{self.count_in_fun:d}')
        print(f'{t:8f}')
        print(f' trace: {trace:.5e}')
        print(f'deltas: {gsg+ldaggerl_unif: .6e}')
        print(f'ldaggl: {ldaggerl_unif: .6e}')
        print(f'  del_: {gsg: .6e}')
        print(f'  norm:  Z: {Z:.6e}   trace**2:{Z0:.6e}')
        print(f'occupy:  up: {np.real(n_up):.5e}  down: {np.real(n_down):.5e}')
        print(f'time:  coefficient: {t2-t1:.3f} , g: {t3-t2:.3f} , all: {self.time_list[-1]-self.time_list[-2]:.3e}')
        print(' ', flush=True)

        f0 = open('t-n_all','a')
        f0.write(f'{t:.6f}   {n_up:.5e}  {n_down:.5e}   {trace:.3e}   {(gsg+ldaggerl_unif):.5e}  {ldaggerl_unif:.5e}\n')
        f0.flush()
        f0.close()
        
        self.count_in_fun += 1
        return g0


fun = fun(states,states_mc,cut1=chainsampler.nonmccut,cut2=chainsampler.allcut,lamda=chainsampler.lmbda)
print(f'nonmc, cut={chainsampler.nonmccut}')
print(f'mc, cut={chainsampler.allcut}\n')

t_lower = input_t[0]
t_upper = 50.0
y0 = rho.rbm_to_vec().to('cpu').numpy()
operator_print = operator_print(sys,states)
result_rk = RK45(fun.fun, t0=t_lower, y0=y0, t_bound=t_upper, rtol=1e-13, atol=1e-5)

while result_rk.t < t_upper :
    result_rk.step()
    count += 1

    print('here print the real step of t')
    t0 = time.time()
    rho_0 = rho.rho_0().detach()
    trace_rho0 = torch.real(torch.trace(rho_0))

    f1 = open('t-step','a')
    I = operator_print.current_general(rho,rho_0)
    Ib = torch.sum(I,dim=0)
    I_tot = torch.sum(Ib).reshape(1)
    print(I)
    f1.write(f'{get_last_line("t-n_all").decode()}')
    f1.write('      ')
    if rho.Nb == 1 : 
        f1.write(f'{torch.real(I[0][0]).item(): .5e}  {torch.real(I[1][0]).item(): .5e}  {torch.real(I_tot[0]).item(): .5e}')
    else : 
        I_tot = torch.cat((Ib,I_tot))
        for i in range(rho.Nb+1) : 
            f1.write(f'{torch.real(I_tot[i]).item(): .5e}  ')
    if rho.Nv>1 :
        S12, Sx2, Sy2, Sz2 = operator_print.spin_ddot(rho_0)
        f1.write('      ')
        f1.write(f'{S12: .5e}   {Sx2:.5e}   {Sy2:.5e}   {Sz2:.5e}')
    f1.write(f'    {fun.count_in_fun-1:5d}\n')
    f1.flush()
    f1.close()
    t1 = time.time()
    print(f'timeI: {t1-t0:.3e}')

    if case=='case2' :
        rho0 = rho_0/torch.trace(rho_0)
        f2 = open('rho0_elements','a')
        f2.write(f'{result_rk.t:.6f}  ')
        for j in range(rho_0.shape[0]):
            f2.write(f'{torch.real(rho0[j][j]):.6e}  ')
        f2.write(f'  {torch.real(rho0[6][9]):.6e}  {torch.real(rho0[9][6]):.6e}  ')
        f2.write(f'  {torch.imag(rho0[6][9]):.6e}  {torch.imag(rho0[9][6]):.6e}  ')
        f2.write(f'{torch.real(torch.trace(rho_0)):.3e}\n')
        f2.flush()
        f2.close()

    if count%1==0 and count!=int(input_t[1]) :
        t2 = time.time()
        states_mc = chainsampler.sampling(rho)
        sys.set_states_need_mc(rho,states_mc,cut=allcut)
        fun.states_mc = states_mc
        t3 = time.time()
        print(f'mc_time: {t3-t2:.2f}')
    if count%20==0 and count!=int(input_t[1]) :
        torch.save(rho.state_dict(),'para'+str(count))
        print(f'here generate para{count:d}')
    print('\n')
