import numpy as np
import scipy.sparse as sp
import time
import sys as sys_sys
import torch


class hsys():

    def __init__(self,read,device) :
        self.device = device
        self.Hsys = sp.csc_array(read[0])
        self.nrho = read[0].shape[0]
        self.c = read[1]
        self.c_c = sp.csc_array(read[1])
        self.c_r = sp.csr_array(read[1])
        self.gamma = read[2]
        self.eta = read[3]
        self.nsgn = read[8]
        self.nspin = read[5]
        self.nvar = read[4]
        self.nalf = read[7]
        self.ncor = read[6]

        hbar = 0.658211928
        chem_potential = True
        if chem_potential :
            chem_potentials = np.loadtxt("chem_potentials.data")
            chem_potentials = chem_potentials.reshape((self.nalf,self.nspin))
            mu = np.zeros((self.nsgn,self.nspin,self.nvar,self.nalf,self.ncor), dtype=np.complex128)
            for a1 in range(self.nsgn) :
                for a2 in range(self.nspin) :
                    for a3 in range(self.nvar) :
                        for a4 in range(self.nalf) :
                            for a5 in range(self.ncor) :
                                mu[a1,a2,a3,a4,a5] = 1.j*(-1)**(a1+1)*chem_potentials[a4,a2]
            self.gamma = self.gamma + mu.flatten()/hbar

    def Llocal(self,rho,state) :
        global num_energy
        Llocal = np.zeros(1,dtype=np.complex128)
        N = rho.Nd
        M = rho.Ns
        num_energy = M
        env_num = rho.Nb*rho.M
        nminus = state[0:N//2]
        Nminus = np.sum(nminus)
        npositive = state[N//2:N]
        Npositive = np.sum(npositive)
        m_0 = state[N:N+M]
        m_1 = state[N+M:N+M*2]
        Subleft = SubscriptTrans0(m_0)
        Subright = SubscriptTrans0(m_1)

        Llocal = Llocal + (-1.0j)*(leftdot(self.Hsys[Subleft,:],rho,nminus,npositive,m_1)
            -rightdot(self.Hsys[:,Subright].T,rho,nminus,npositive,m_0))\
            + self.gamma.dot(state[0:N])*rho.State(state)
        i = Nminus
        A = nminus
        v = 0
        for k in range(0,N//2) :
            i = i - nminus[k]
            if nminus[k] == 0 :
                A[k] = 1
                v = Jtov(k,env_num,rho.No,rho.Nv) 
                Llocal = Llocal + (-1.0j)*np.power(-1,i)*(
                    leftdot(np.conj(self.c_r[v*self.nrho:(v+1)*self.nrho,Subleft]).T,rho,A,npositive,m_1) \
                    - np.power(-1,Nminus+Npositive)*\
                    rightdot(np.conj(self.c_c[v*self.nrho+Subright,:]),rho,A,npositive,m_0)
                )
                A[k] = 0
        i = 0
        A = npositive
        v = 0
        for k in range(0,N//2) :
            if k > 0 :
                i = i + npositive[k-1]
            if npositive[k] == 0 :
                A[k] = 1
                v = Jtov(k,env_num,rho.No,rho.Nv) 
                Llocal = Llocal + (-1.0j)*np.power(-1,i)*(np.power(-1,Nminus+Npositive)* \
                    leftdot(self.c_c[v*self.nrho+Subleft,:],rho,nminus,A,m_1) - \
                    rightdot(self.c_r[v*self.nrho:(v+1)*self.nrho,Subright].T,rho,nminus,A,m_0)
                )
                A[k] = 0
        i = 0
        A = nminus
        B = np.zeros(1,dtype=np.complex128)
        v = 0
        for k in range(0,N//2) :
            if k > 0 :
                i = i + nminus[k-1]
            if nminus[k] == 1:
                A[k] = 0
                v = Jtov(k,env_num,rho.No,rho.Nv)
                B = np.power(-1,Nminus-1)*self.eta[k,0]* \
                    leftdot(self.c_c[v*self.nrho+Subleft,:],rho,A,npositive,m_1) - np.power(-1,Npositive)*\
                    rightdot(np.conj(self.eta[k,1])*self.c_r[v*self.nrho:(v+1)*self.nrho,Subright].T,rho,A,npositive,m_0)
                Llocal = Llocal + (-1.j)*np.power(-1,i)*B
                A[k] = 1
        i = Npositive
        A = npositive
        for k in range(0,N//2) :
            i = i - npositive[k]
            if npositive[k] == 1 :
                A[k] = 0
                v = Jtov(k,env_num,rho.No,rho.Nv)
                B = np.power(-1,Nminus)*self.eta[k,1]* \
                    leftdot(np.conj(self.c_r[v*self.nrho:(v+1)*self.nrho,Subleft]).T,rho,nminus,A,m_1) - \
                    np.power(-1,Npositive-1)*\
                    rightdot(np.conj(self.eta[k,0])*np.conj(self.c_c[v*self.nrho+Subright,:]),rho,nminus,A,m_0)
                Llocal = Llocal + (-1.j)*np.power(-1,i)*B
                A[k] = 1
        Llocal = Llocal
        return Llocal

    def Llocal_pre(self,rho,state) :
        global num_energy
        N = rho.Nd
        N_half = N//2
        M = rho.Ns
        num_energy = M
        env_num = rho.Nb*rho.M
        nminus = state[0:N_half]
        Nminus = np.sum(nminus)
        npositive = state[N_half:N]
        Npositive = np.sum(npositive)
        m_0 = state[N:N+M]
        m_1 = state[N+M:N+M*2]
        Subleft = int(SubscriptTrans0(m_0))
        Subright = int(SubscriptTrans0(m_1))
        count = 0
        L_pre = np.zeros(2*rho.Nd+1,dtype=np.complex128)
        states_need = np.zeros((2*rho.Nd+1,rho.nstate),dtype=np.int8)
        
        states_need[count] = state
        L_pre[count] = (-1.0j)*(self.Hsys[Subleft,Subleft]-self.Hsys[Subright,Subright])+self.gamma.dot(state[0:N])
        count += 1

        H_nonzero_left = self.Hsys[[Subleft],:].nonzero()[1]
        for i in range(H_nonzero_left.shape[0]) :
            if H_nonzero_left[i]!=Subleft :
                states_need[count] = np.hstack((nminus,npositive,SubscriptTrans1(H_nonzero_left[i],M),m_1))
                L_pre[count] = (-1.0j)*self.Hsys[Subleft,H_nonzero_left[i]]
                count += 1
        H_nonzero_right = self.Hsys[:,[Subright]].nonzero()[0]
        for i in range(H_nonzero_right.shape[0]) :
            if H_nonzero_right[i]!=Subright :
                states_need[count] = np.hstack((nminus,npositive,m_0,SubscriptTrans1(H_nonzero_right[i],M)))
                L_pre[count] = (1.0j)*self.Hsys[H_nonzero_right[i],Subright]
                count += 1

        i = Nminus
        A = nminus
        v = 0
        for k in range(0,N_half) :
            i = i - nminus[k]
            if nminus[k] == 0 :
                nminus[k] = 1
                v = Jtov(k,env_num,rho.No,rho.Nv) 
                w = rho.Ns-v-1
                if m_0[w] == 1 :
                    m_0[w] = 0
                    states_need[count] = np.hstack((nminus,npositive,m_0,m_1))
                    L_pre[count] = (-1.0j)*pow(-1,i)*self.c_r[v*self.nrho+int(SubscriptTrans0(m_0)),Subleft].conjugate()
                    count += 1
                    m_0[w] = 1
                if m_1[w] == 0 :
                    m_1[w] = 1
                    states_need[count] = np.hstack((nminus,npositive,m_0,m_1))
                    L_pre[count] = -(-1.0j)*pow(-1,i)*pow(-1,Nminus+Npositive)*self.c_c[v*self.nrho+Subright,int(SubscriptTrans0(m_1))].conjugate()
                    count += 1
                    m_1[w] = 0
                nminus[k] = 0
        i = 0
        A = npositive
        v = 0
        for k in range(0,N_half) :
            if k > 0 :
                i = i + npositive[k-1]
            if npositive[k] == 0 :
                npositive[k] = 1
                v = Jtov(k,env_num,rho.No,rho.Nv) 
                w = rho.Ns-v-1
                if m_0[w] == 0 :
                    m_0[w] = 1
                    states_need[count] = np.hstack((nminus,npositive,m_0,m_1))
                    L_pre[count] =(-1.0j)*pow(-1,i)*pow(-1,Nminus+Npositive)*self.c_c[v*self.nrho+Subleft,int(SubscriptTrans0(m_0))]
                    count += 1
                    m_0[w] = 0
                if m_1[w] == 1 :
                    m_1[w] = 0
                    states_need[count] = np.hstack((nminus,npositive,m_0,m_1))
                    L_pre[count] = -(-1.0j)*pow(-1,i)*self.c_r[v*self.nrho+int(SubscriptTrans0(m_1)),Subright]
                    count += 1
                    m_1[w] = 1
                npositive[k] = 0
        i = 0
        A = nminus
        B = np.zeros(1,dtype=np.complex128)
        v = 0
        for k in range(0,N_half) :
            if k > 0 :
                i = i + nminus[k-1]
            if nminus[k] == 1:
                nminus[k] = 0
                v = Jtov(k,env_num,rho.No,rho.Nv)
                w = rho.Ns-v-1
                if m_0[w] == 0 :
                    m_0[w] = 1
                    states_need[count] = np.hstack((nminus,npositive,m_0,m_1))
                    L_pre[count] = (-1.j)*pow(-1,i)*pow(-1,Nminus-1)*self.eta[k,0]*self.c_c[v*self.nrho+Subleft,int(SubscriptTrans0(m_0))]
                    count += 1
                    m_0[w] = 0
                if m_1[w] == 1 :
                    m_1[w] = 0
                    states_need[count] = np.hstack((nminus,npositive,m_0,m_1))
                    L_pre[count] = -(-1.j)*pow(-1,i)*pow(-1,Npositive)*(self.eta[k,1]).conj()*self.c_r[v*self.nrho+int(SubscriptTrans0(m_1)),Subright]
                    count += 1
                    m_1[w] = 1
                nminus[k] = 1
        i = Npositive
        A = npositive
        for k in range(0,N_half) :
            i = i - npositive[k]
            if npositive[k] == 1 :
                npositive[k] = 0
                v = Jtov(k,env_num,rho.No,rho.Nv)
                w = rho.Ns-v-1
                if m_0[w] == 1 :
                    m_0[w] = 0
                    states_need[count] = np.hstack((nminus,npositive,m_0,m_1))
                    L_pre[count] =(-1.j)*pow(-1,i)*pow(-1,Nminus)*self.eta[k,1]*(self.c_r[v*self.nrho+int(SubscriptTrans0(m_0)),Subleft]).conjugate()
                    count += 1
                    m_0[w] = 1
                if m_1[w] == 0 :
                    m_1[w] = 1
                    states_need[count] = np.hstack((nminus,npositive,m_0,m_1))
                    L_pre[count] = (self.eta[k,0]).conj()*self.c_c[v*self.nrho+Subright,int(SubscriptTrans0(m_1))].conjugate()
                    L_pre[count] =-(-1.j)*pow(-1,i)*pow(-1,Npositive-1)*(self.eta[k,0]).conj()*self.c_c[v*self.nrho+Subright,int(SubscriptTrans0(m_1))].conjugate()
                    count += 1
                    m_1[w] = 0
                npositive[k] = 1
        return count, L_pre[0:count], states_need[0:count]

    def Llocals_pre(self,rho,states) :
        torch.set_default_dtype(torch.float64)
        states = states.to('cpu').numpy()
        n = states.shape[0]
        N = (2*rho.Nd+1)*n
        count = np.zeros(n,dtype=np.int)
        L_pre = np.zeros(N,dtype=np.complex128)
        states_need = np.zeros((N,rho.nstate),dtype=np.int8)
        for i in range(0,n):
            count_0, L_pre_0, states_need_0 = self.Llocal_pre(rho,states[i])
            if i == 0:
                count[i] = count_0
                L_pre[0:count_0] = L_pre_0
                states_need[0:count_0] = states_need_0
            else:
                count[i] = count[i-1] + count_0
                L_pre[count[i-1]:count[i]] = L_pre_0
                states_need[count[i-1]:count[i]] = states_need_0
        return torch.tensor(count),torch.tensor(L_pre[0:count[-1]],device=self.device),torch.tensor(states_need[0:count[-1]],device=self.device)

    def set_states_need(self,rho,states,cut=3):
        t1 = time.time()
        count_pre, L_pre, states_need = self.Llocals_pre(rho,states)
        states_cut = self.states_cut(rho,states_need,cut=cut)
        Llocals_pre = {
            'states':states,
            'count_pre':count_pre, 
            'L_pre':L_pre, 
            'states_need':states_need,
            'states_cut':states_cut
            }
        self.states_need = Llocals_pre
        t2 = time.time()
        print(f'time_exact-L_pre: {t2-t1:.3f}', flush=True)

    def set_states_need_mc(self,rho,states,cut=3):
        self.states_choices_mc = states.shape[0]
        self.states_need_mc = []
        t1 = time.time()
        for i in range(0,self.states_choices_mc):
            count_pre, L_pre, states_need = self.Llocals_pre(rho,states[i])
            states_cut = self.states_cut(rho,states_need,cut=cut)
            Llocals_pre = {
                'states':states[i],
                'count_pre':count_pre, 
                'L_pre':L_pre, 
                'states_need':states_need,
                'states_cut':states_cut
                }
            self.states_need_mc.append(Llocals_pre)
        t2 = time.time()
        print(f'time_mc-L_pre: {t2-t1:.3f}', flush=True)

    def Lforward_nonmc(self,rho,lmbda=0.,loss=True) :
        count = self.states_need['count_pre']
        n = self.states_need['states'].shape[0]
        Lforward = torch.zeros(n,dtype=torch.complex128,device=self.device)
        rho_need = rho.States(self.states_need['states_need'])
        Lforward_pre = torch.multiply(self.states_need['L_pre']*self.states_need['states_cut'],rho_need)
        for i in range(0,n):
           if i == 0:
               Lforward[0] = torch.sum(Lforward_pre[0:count[0]])
           else:
               Lforward[i] = torch.sum(Lforward_pre[count[i-1]:count[i]])
        if loss:
            y = self.states_need['states'][:,0:rho.Nd]
            N = torch.exp(lmbda*torch.sum(y,dim=1))
            return torch.real(torch.multiply(Lforward,Lforward.conj()))/N
        else:
            return Lforward.detach()
        
    def Lforward_mc(self,rho,lmbda=0.,loss=True,choices=0) :
        j = choices
        count = self.states_need_mc[j]['count_pre']
        n = self.states_need_mc[j]['states'].shape[0]
        Lforward = torch.zeros(n,dtype=torch.complex128,device=self.device)
        rho_need = rho.States(self.states_need_mc[j]['states_need'])
        Lforward_pre = torch.multiply(self.states_need_mc[j]['L_pre']*self.states_need_mc[j]['states_cut'],rho_need)
        for i in range(0,n):
            if i == 0:
                Lforward[0] = torch.sum(Lforward_pre[0:count[0]])
            else:
                Lforward[i] = torch.sum(Lforward_pre[count[i-1]:count[i]])
        if loss:
            y = self.states_need_mc[j]['states'][:,0:rho.Nd]
            N = torch.exp(lmbda*torch.sum(y,dim=1))
            return torch.real(torch.multiply(Lforward,Lforward.conj()))/N
        else:
            return Lforward.detach()

    def states_cut(self,rho,states,cut=2) :
        N = rho.Nd//2
        states = states.to(torch.int8)
        state_cut = torch.sum(states[:,0:rho.Nd],dim=1)
        ifcut = (-torch.sign(state_cut-cut-0.1)+1)//2
        return (ifcut).to(torch.complex128)



def AcceptProbability(Statenow,Statenext,rho,lamda) :
    if rho.f0(Statenext):
        N_now = np.sum(Statenow[0:rho.Nd])
        N_next = np.sum(Statenext[0:rho.Nd])
        a = np.exp(lamda*(N_next-N_now))
        return np.min(np.array([1.,a]))
    return 0.

def SubscriptTrans0(m) :
    j = np.zeros(1,dtype=int)
    a = np.zeros(1,dtype=int)
    N = m.size
    for i in m :
        a = a + i*np.power(2,N-j-1)
        j = j + 1
    return a

def SubscriptTrans0_torch(m) :
    shape = m.size()[-1]
    a = torch.zeros(shape,dtype=torch.int8,device=m.device)
    for i in range(shape) : 
        a[i] = torch.pow(2,torch.tensor([shape-i-1]))
    return torch.sum(m*a,dim=-1)

def SubscriptTrans1(a,N) :
    m = np.zeros(N,dtype=int)
    i = 0
    while a > 0 :
        m[N-i-1] = np.mod(a,2)
        a = int(np.floor_divide(a,2))
        i += 1
    return m

def leftdot(l,rho,nminus,npositive,m_1):
    cloumn = l.nonzero()[1]
    dot = np.zeros(1,dtype=np.complex128)
    for i in cloumn:
        dot = dot + l[0,i]*rho.State(np.hstack((nminus,npositive,
                SubscriptTrans1(i,m_1.size),m_1)))
    return dot

def rightdot(l,rho,nminus,npositive,m_0):
    cloumn = l.nonzero()[1]
    dot = np.zeros(1,dtype=np.complex128)
    for i in cloumn:
        dot = dot + l[0,i]*rho.State(np.hstack((nminus,npositive,
                m_0,SubscriptTrans1(i,m_0.size))))
    return dot

def leftdot_d(l,rho,nminus,npositive,m_1):
    cloumn = l.nonzero()[1]
    dot = np.zeros(rho.nparameters,dtype=np.complex128)
    for i in cloumn:
        state = np.hstack((nminus,npositive,SubscriptTrans1(i,m_1.size),m_1))
        dot = dot + l[0,i]*rho.Gradient(state)*rho.State(state)
    return dot

def rightdot_d(l,rho,nminus,npositive,m_0):
    cloumn = l.nonzero()[1]
    dot = np.zeros(rho.nparameters,dtype=np.complex128)
    for i in cloumn:
        state = np.hstack((nminus,npositive,m_0,SubscriptTrans1(i,m_0.size)))
        dot = dot + l[0,i]*rho.Gradient(state)*rho.State(state)
    return dot

def LeftRho(rho,nminus,npositive,m_1) :
    global num_energy
    N = np.power(2,num_energy)
    Rho = np.zeros(N,dtype=np.complex128)
    for i in range(0,N) :
        Rho[i] = rho.State(np.hstack((nminus,npositive,SubscriptTrans1(i,num_energy),m_1)))
    return Rho

def RightRho(rho,nminus,npositive,m_0) :
    global num_energy
    Rho = np.zeros(2**num_energy,dtype=np.complex128)
    for i in range(0,2**num_energy) :
        Rho[i] = rho.State(np.hstack((nminus,npositive,m_0,SubscriptTrans1(i,num_energy))))
    return Rho

def Jtov(k,env_num,No,Nv) :
    k = k%(env_num*No*Nv)//env_num
    return No*(k%Nv)+k//Nv
