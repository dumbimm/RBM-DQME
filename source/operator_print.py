import torch
import numpy as np
import time
from hamsys64 import SubscriptTrans0_torch
import scipy.linalg 



class operator_print() :
    def __init__(self,sys,states) :
        self.states = states
        nsgn,No,Nv,Nb,M = sys.nsgn,sys.nspin,sys.nvar,sys.nalf,sys.ncor
        Nd = nsgn * No * Nv * Nb * M
        Ns = No * Nv
        nrho = sys.nrho
        self.device = states.device
        c_vo = torch.tensor(sys.c,dtype=torch.complex128,device=self.device)
        self.H_S = torch.tensor(sys.Hsys.toarray(),dtype=torch.complex128,device=self.device)


        self.occu_up = torch.zeros((nrho,nrho),dtype=torch.complex128,device=self.device)
        self.occu_down = torch.zeros((nrho,nrho),dtype=torch.complex128,device=self.device)
        for i in range(Nv) :
            self.occu_up += c_vo[2*i*nrho:(2*i+1)*nrho,:].T.matmul(c_vo[2*i*nrho:(2*i+1)*nrho,:])
            self.occu_down += c_vo[(2*i+1)*nrho:2*(i+1)*nrho,:].T.matmul(c_vo[(2*i+1)*nrho:2*(i+1)*nrho,:])
        
        self.coe_curr = 1.602*pow(10,5)
        states_n = self.states[:,0:Nd]
        sum_states_n = torch.sum(states_n,dim=1)
        self.ado1_posi1 = []
        for i in range(sum_states_n.shape[0]) :
            if sum_states_n[i]==1 :
                self.ado1_posi1.append(i)
        self.states_ado1 = self.states[self.ado1_posi1]
        y_,x1_,x2_ = torch.split(self.states_ado1,[Nd, Ns, Ns], dim=1)
        self.ado1_posi2 = torch.zeros((3,len(self.ado1_posi1)),dtype=torch.int)
        self.ado1_posi2[0] = torch.nonzero(y_)[:,1]
        self.ado1_posi2[1] = SubscriptTrans0_torch(x1_)
        self.ado1_posi2[2] = SubscriptTrans0_torch(x2_)
        self.c_ov_I = torch.zeros((Nd,nrho,nrho),dtype=torch.complex128,device=self.device)
        self.c_ov_E = torch.zeros((Nd,nrho,nrho),dtype=torch.complex128,device=self.device)
        for i in range(Nd//2) : 
            count_ov = (i%(Nd//2))//(Nb*M)
            count_vo = No*(count_ov%Nv)+count_ov//Nv
            c_v = c_vo[count_vo*nrho:(count_vo+1)*nrho,:]
            self.c_ov_I[i+Nd//2,:,:] = c_ov
            self.c_ov_I[i,:,:] = -c_ov.T
            self.c_ov_E[i+Nd//2,:,:] = c_ov
            self.c_ov_E[i,:,:] = c_ov.T

        self.pauli = torch.zeros((3,2,2),dtype=torch.complex128,device=self.device)
        self.pauli[0] = torch.tensor([[0,0.5],[0.5,0]])
        self.pauli[1] = torch.tensor([[0,0.5j],[-0.5j,0]])
        self.pauli[2] = torch.tensor([[-0.5,0],[0,0.5]])
        self.S_xyz = torch.zeros((3,nrho,nrho),dtype=torch.complex128,device=self.device)
        self.S_v = torch.zeros((Nv,3,nrho,nrho),dtype=torch.complex128,device=self.device)
        for i in range(3) :
            for v in range(Nv) :
                for s1 in range(No) :
                    for s2 in range(No) :
                        count_vo1 = v*No+s1
                        count_vo2 = v*No+s2
                        self.S_xyz[i] += self.pauli[i,s1,s2]*(c_vo[count_vo1*nrho:(count_vo1+1)*nrho,:].T).matmul(c_vo[count_vo2*nrho:(count_vo2+1)*nrho,:])
                        self.S_v[v,i] += self.pauli[i,s1,s2]*(c_vo[count_vo1*nrho:(count_vo1+1)*nrho,:].T).matmul(c_vo[count_vo2*nrho:(count_vo2+1)*nrho,:])

    def occupation(self,rho_0) :
        rho_0 = rho_0/torch.trace(rho_0)
        up = torch.trace(self.occu_up.matmul(rho_0))
        down = torch.trace(self.occu_down.matmul(rho_0))
        return torch.real(up.detach()), torch.real(down.detach())
        
    def current_general(self,rho,rho_0) :
        trace_rho0 = torch.trace(rho_0)
        ado1 = torch.zeros((rho.Nd,2**rho.Ns,2**rho.Ns),dtype=torch.complex128,device=self.device)
        rho_states_ado1 = rho.States(self.states_ado1)
        posi2 = self.ado1_posi2
        for i in range(len(self.ado1_posi1)) : 
            ado1[posi2[0,i],posi2[1,i],posi2[2,i]] = rho_states_ado1[i]
        I_pre = self.c_ov_I.matmul(ado1)
        I = torch.zeros((rho.No,rho.Nb),dtype=torch.complex128,device=self.device)
        for i in range(rho.No) : 
            for j in range(rho.Nb) : 
                for k1 in range(rho.nsgn) : 
                    for k2 in range(rho.Nv) : 
                        count_I = k1*(rho.Nd//2)+i*(rho.Nv*rho.Nb*rho.M)+k2*(rho.Nb*rho.M)+j*rho.M
                        I[i][j] += torch.trace(torch.sum(I_pre[count_I:count_I+rho.M,:,:],dim=0))
        I = (1.j*I/trace_rho0)*self.coe_curr
        return I

    def spin_ddot(self,rho_0) :
        rho_0 = rho_0/torch.trace(rho_0)
        S12 = torch.trace(torch.sum(self.S_v[0].matmul(self.S_v[1]),dim=0).matmul(rho_0)).detach()
        Sx2 = torch.trace(((self.S_xyz[0]).matmul(self.S_xyz[0])).matmul(rho_0)).detach()
        Sy2 = torch.trace(((self.S_xyz[1]).matmul(self.S_xyz[1])).matmul(rho_0)).detach()
        Sz2 = torch.trace(((self.S_xyz[2]).matmul(self.S_xyz[2])).matmul(rho_0)).detach()
        return torch.real(S12), torch.real(Sx2), torch.real(Sy2), torch.real(Sz2)

    def S_vN(self,rho_0) :
        rho_0 = rho_0/torch.trace(rho_0)
        log_rho0 = torch.tensor(scipy.linalg.logm(rho_0.to('cpu').numpy()),device=self.device)
        return -torch.real(torch.trace(torch.matmul(rho_0,log_rho0)))

    def E_SE(self,rho,rho_0) :
        ado1 = torch.zeros((rho.Nd,2**rho.Ns,2**rho.Ns),dtype=torch.complex128,device=self.device)
        rho_states_ado1 = rho.States(self.states_ado1)
        posi2 = self.ado1_posi2
        for i in range(len(self.ado1_posi1)) : 
            ado1[posi2[0,i],posi2[1,i],posi2[2,i]] = rho_states_ado1[i]

        c_ado1 = self.c_ov_E.matmul(ado1)
        E_SB = torch.zeros(1,dtype=torch.complex128,device=self.device)
        for i in range(c_ado1.shape[0]) : 
            E_SB += torch.trace(c_ado1[i,:,:])
        return E_SB/torch.trace(rho_0)

    def E_S(self,rho_0) :
        return torch.trace(self.H_S.matmul(rho_0))/torch.trace(rho_0)

