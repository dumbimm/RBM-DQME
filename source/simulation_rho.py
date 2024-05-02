import torch
import numpy as np
import sys as sys_sys
import time

from hamsys64 import SubscriptTrans0_torch
from operator_print import operator_print
from init import sys,rho,device,states,nonmccut,allcut,case
import scipy.optimize as spop


rho_input = torch.tensor(np.loadtxt("table_cut"+str(allcut+1)+".data",usecols=(0,1),dtype=np.float64),device=device)
target = (rho_input[:,0] + 1.j*rho_input[:,1])

N_rho = torch.sum(states[:,:rho.Nd],axis=1)
coef = 1**torch.sum(states[:,:rho.Nd],dim=1,dtype=torch.float64)
abs_target = torch.abs(target)

N_M = torch.zeros((states.shape[0],rho.M),dtype=torch.int8,device=device)
for i in range(rho.M):
    N_M[:,i] = torch.sum(states[:,[j*rho.M+i for j in range(rho.Nd//rho.M)]],dim=1)
coef_para = np.sum(np.abs(sys.eta.T.reshape(-1,rho.M))+np.abs(sys.gamma.reshape(-1,rho.M)),axis=0)/(rho.Nd//rho.M)
coef = 1+torch.sum(N_M*torch.tensor(coef_para,dtype=torch.float64,device=device),axis=1)/4
if case=='case1':
    for i in range(2**rho.Ns):
        coef[i] = torch.floor((torch.floor(16/torch.abs(target[i])))/100)+1
elif case=='case2':    
    for i in range(2**rho.Ns+2):
        coef[i] = torch.floor((torch.floor(16/torch.abs(target[i])))/100)+1
        coef[7],coef[10] = 50,50
else :
    print('error, The first line in input needs to specify parameters "case1" or "case2".')


m = rho.nstate
m1 = rho.Ns
n = rho.nparameters

count = 0
step_count = 0
step_count_back = 0
t_hop = [time.time()]
@torch.no_grad()
def loss(vec_of_rho) :
    rho.vec_to_rbm(torch.tensor(vec_of_rho,dtype=torch.float64,device=device))
    rho_states = rho.States(states)
    delta = (rho_states-target)*coef
    loss2 = torch.real(torch.dot(delta,torch.conj(delta)))
    global count
    global count_0
    global step_count_back
    global t3
    global t4
    if count%count_0==0 or step_count-step_count_back==1 :
        if count==0:
            t3 = time.time()            
        t4 = time.time()
        print(f'count:{count:5d}    loss_{meth}:{loss2:5e}   time:{t4-t3}',flush=True)
        step_count_back = step_count
        t3 = time.time()
    count += 1
    if count%1000==0 and meth=='BFGS':
        torch.save(rho.state_dict(),f'rho_BFGS_{count}')
        result_print(loss2,ifresult=False)
    if count%40000==0 and meth=='TNC' and tol<=1e-16:
        torch.save(rho.state_dict(),f'rho_TNC_{count}')
        result_print(loss2,ifresult=False)
    return loss2.to('cpu').numpy()

def loss_d(vec_of_rho) :
    rho.vec_to_rbm(torch.tensor(vec_of_rho,dtype=torch.float64,device=device))
    rho.zero_grad()
    rho_states = rho.States(states)
    delta = (rho_states-target)*coef
    loss = torch.real(torch.dot(delta,torch.conj(delta)))
    loss.backward()
    lossd = rho.rbmd_to_vec()
    rho.zero_grad()
    return lossd.to('cpu').numpy()

@torch.no_grad()
def ldaggl(vec_of_rho):
    rho.vec_to_rbm(torch.tensor(vec_of_rho,dtype=torch.float64,device=device))
    L2 = torch.sum(sys.Lforward_nonmc(rho))
    trace = torch.trace(rho.rho_0())
    Z0 = torch.real(trace)**2
    return L2/Z0

class MyTakeStep:
    def __init__(self,stepsize):
        self.step = stepsize
        self.rng = np.random.default_rng()
    def __call__(self, x):

        if np.abs(np.average(x)) > 10:
            print('overflow,then restart\n')
            x = self.rng.uniform(-4,0,x.shape)
        x_next = self.unirandom(x)
        while(np.average(x_next) > 5):
            print(f'average over 5:{np.average(x_next)}\n')
            x_next = self.unirandom(x)
        x = x_next
        rho.vec_to_rbm(torch.tensor(x,dtype=torch.float64,device=device))
        global count
        global meth
        torch.save(rho.state_dict(),f'rho_simulate_{meth}_{step_count+1}_0')
        count = 0
        return x
    def unirandom(self,x) :
        s = self.step
        return x + self.rng.uniform(-s,s,x.shape)

def print_fun(x,f,accepted):
    global step_count
    global meth
    step_count += 1
    t_hop.append(time.time())
    print("step:%2d, time_step = %.2f, accepted %d, loss at minimum %.5e" % (step_count,t_hop[-1]-t_hop[-2],int(accepted),f))
    rho.vec_to_rbm(torch.tensor(x,dtype=torch.float64,device=device))

    rho_0 = rho.rho_0().detach()
    n_up, n_down = operator_print.occupation(rho_0)
    print(f'up:{n_up:6e} ,down:{n_down:6e} ,trace: {torch.real(torch.trace(rho_0)):.5e}')

    print('ldaggl = %.5e\n' % (ldaggl(x)))

    if isinstance(f,float) :
        if accepted==1 and np.isnan(f)==False :
            rho.vec_to_rbm(torch.tensor(x,dtype=torch.float64,device=device))
            torch.save(rho.state_dict(),f'rho_simulate_{meth}_{step_count}')
    print('\n')
    return 0

def result_print(loss,ifresult=True):
    global count
    if ifresult:
        print("n_it: %d" % result.nit)
        print("total evaluations: %d" % result.nfev,flush=True)
        print(f'terminate because:{result.message}')
        rho.vec_to_rbm(torch.tensor(result.x,dtype=torch.float64,device=device))

    rho_0 = rho.rho_0().detach()
    n_up, n_down = operator_print.occupation(rho_0)
    trace = torch.real(torch.trace(rho_0))
    ll = ldaggl(rho.rbm_to_vec().numpy())
    if ifresult:
        print(f'up:{n_up:6e}  ;down:{n_down:6e}')
        print(f'trace: {trace:.3e}')
        print('ldaggl_at_end')
        print(f'ldaggl:{ll:.5e}')
        print('\n')
    else:
        I = operator_print.current_general(rho,rho_0)
        Ib = torch.sum(I,dim=0)
        I_tot = torch.sum(Ib).reshape(1)
        print(f'{count:5d}   {n_up:.5e}  {n_down:.5e}   {trace:.3e}   {loss:.5e}  {ll:.5e}')
        if rho.Nb == 1 : 
            print(f'    {torch.real(I[0][0]).item(): .5e}    {torch.real(I[1][0]).item(): .5e}    {torch.real(I_tot[0]).item(): .5e}')
        else : 
            I_tot = torch.cat((Ib,I_tot))
            for i in range(rho.Nb+1) : 
                print(f'    {torch.real(I_tot[i]).item(): .5e}',end='')
            print('\n')
        if rho.Nv>1 :
            S12, Sx2, Sy2, Sz2 = operator_print.spin_ddot(rho_0)
            print(f'       {S12: .5e}   {Sx2:.5e}  {Sy2:.5e}  {Sz2:.5e}')

operator_print = operator_print(sys,states)

rho_0_target = torch.zeros(4**m1,dtype=torch.complex128,device=device)
for i in range(2*2**m1) :
    if torch.sum(states[i,0:rho.Nd])==0 and (states[i,rho.Nd:rho.Nd+rho.Ns]==states[i,rho.Nd+rho.Ns:]).all() :
        position = SubscriptTrans0_torch(states[i])
        rho_0_target[position] = target[i]
rho_0_target = rho_0_target.detach().reshape(2**m1,2**m1)
n_up_target, n_down_target = operator_print.occupation(rho_0_target)
print(f'up_target:{n_up_target:6e}  ;down_target:{n_down_target:6e}\n')
rho_0 = rho.rho_0().detach()
trace_origin = torch.trace(rho_0)
print(f'origin_trace: {torch.real(trace_origin): .3e}\n',flush=True)
print(f'ldaggl_init:{ldaggl(rho.rbm_to_vec().numpy()):8e}\n',flush=True)


torch.save(rho.state_dict(),f'rho_simulate_TNC_1_0')
print(f'datatype:{rho.bh.dtype}')
step = 0.32
mytakestep = MyTakeStep(step)


t0 = time.time()
tol = 1e-10
meth='TNC'
count_0 = 1000
maxiter = 16000
print(f'tol:{tol}')
print(f'maxfun:{maxiter}')
print(f'method:{meth}')
result = spop.basinhopping(loss,rho.rbm_to_vec(),niter=3,T=0.011,
    take_step=mytakestep,callback=print_fun,
    minimizer_kwargs={'method':meth,'jac':loss_d,'tol':tol,'options':{'maxfun':maxiter}})
result_print(loss(result.x))
torch.save(rho.state_dict(),'rho_basinhopping_TNC')

tol = 1e-16
meth='TNC'
print(f'tol:{tol}')
print(f'method:{meth}')
maxiter = 10*len(rho.rbm_to_vec())
count_0 = 1000
result = spop.minimize(loss,rho.rbm_to_vec(),method=meth,jac=loss_d,tol=tol,options={'maxfun':maxiter})
result_print(loss(result.x))


if ldaggl(rho.rbm_to_vec().numpy())<2e-7 and case=='case1':
    torch.save(rho.state_dict(),'rho_TNC')
elif ldaggl(rho.rbm_to_vec().numpy())<6e-6 and case=='case2':
    torch.save(rho.state_dict(),'rho_TNC')
else :
    print('Simulation not successful, please simulate again.')
