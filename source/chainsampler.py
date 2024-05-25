import torch
import numpy as np


class Chainsampler():
    
    def __init__(self,rule='exact',N_start=2000,N_end=10000,lmbda=0.,choices=5,cut1=2,cut2=3,nstate=0,filename='table1.data'):
        self.rule = rule
        if rule == 'exact' :
            statescols = [i for i in range(3,3+nstate)]
            self.table = torch.tensor(np.loadtxt(filename,usecols=statescols,dtype=np.int8))
        elif rule == 'mhmc' :
            self.start = N_start
            self.end = N_end
            self.lmbda = lmbda #exp(-lambda*n)
            self.choices = choices
            self.nonmccut = cut1
            self.allcut = cut2

    def sampling(self,rho):
        device = rho.device
        if self.rule == 'exact' :
            self.table = self.table.to(device)
            return self.table

        elif self.rule == 'mhmc' :
            device = 'cpu'
            m = rho.nstate
            # Statenow = torch.zeros(m,dtype=torch.int8,device=device)
            # Statenow = torch.load('state0_mc',map_location=device)
            chain = torch.zeros((self.choices,self.end-self.start,m),dtype=torch.int8,device=device)
            for j in range(0,self.choices) :
                Statenow = torch.load('state0_mc',map_location=device)[j]
                for i in range(0,self.end) :
                    Statenow = self.flip(Statenow,rho)
                    if i >= self.start :
                        chain[j,i-self.start] = Statenow
            return chain.to(rho.device)

    def flip(self,statenow,rho) :
            n = rho.Nd
            m = statenow.shape[0]
            statenext = torch.zeros(m,device=statenow.device)
            N = torch.div(rho.Nd,2,rounding_mode='trunc')
            M = rho.M*rho.Nb
            count = 0
            k = 0
            while k==0 or torch.sum(statenow[0:rho.Nd]) <= self.nonmccut or torch.sum(statenow[0:rho.Nd]) > self.allcut:
                k += 1
                for j in range(0,5):
                    statenext = statenow.clone()
                    u = torch.rand((1,1))
                    if u < 1/rho.M :
                        state_flip = torch.randint(0,rho.Ns,(1,1))[0,0]
                        statenext[rho.Nd+state_flip] =  (statenext[rho.Nd+state_flip]+1)%2
                        statenext[rho.Nd+rho.Ns+state_flip] = (statenext[rho.Nd+rho.Ns+state_flip]+1)%2
                    else :
                        state_flip = torch.randint(0,rho.Nd,(1,1))[0,0]
                        statenext[state_flip] =  (statenext[state_flip]+1)%2
                        if state_flip < N:
                            i = torch.div(state_flip,M,rounding_mode='trunc')
                            i = rho.Ns-i-1
                            i = (i%rho.No)*rho.Nv+i//rho.No
                            i = rho.Ns + i
                            statenext[rho.Nd+i] = (statenext[rho.Nd+i]+1)%2
                            # statenext[rho.Nd+rho.Ns+rho.Ns-i-1] = (statenext[rho.Nd+rho.Ns+rho.Ns-i-1]+1)%2
                        else:
                            i = torch.div(state_flip-N,M,rounding_mode='trunc')
                            i = rho.Ns-i-1
                            i = (i%rho.No)*rho.Nv+i//rho.No
                            i = i
                            statenext[rho.Nd+i] = (statenext[rho.Nd+i]+1)%2
                    # if torch.sum(statenext[0:rho.Nd]) <= self.nonmccut:
                    #     # j = j-1
                    #     continue
                    v = AcceptProbability(statenow,statenext,rho,self.lmbda)

                    u = torch.rand((1,1))
                    if u < v :
                        count += 1
                        statenow = statenext.clone()
            return statenow

def AcceptProbability(Statenow,Statenext,rho,lmbda) :
    if rho.f0(Statenext):
        N_now = torch.sum(Statenow[0:rho.Nd])
        N_next = torch.sum(Statenext[0:rho.Nd])
        a = torch.exp(lmbda*(N_next-N_now))
        return torch.min(torch.tensor([1.,a]))
    return 0.

      
