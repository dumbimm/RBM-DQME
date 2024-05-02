import torch.nn as nn
import torch
# from hamsys64 import SubscriptTrans0_torch


def Complex1_s(a) :
    m_real = torch.Tensor(a).uniform_(-0.5,0.3).type(torch.complex64)
    m_img = torch.Tensor(a).normal_(0,0.01).type(torch.complex64)
    m = m_real + (1.j)*m_img
    return m

def Complex2_sd(a,b) :
    m_real = torch.Tensor(a,b).uniform_(-0.1,0.05).type(torch.complex64)
    m_img = torch.Tensor(a,b).normal_(0,0.2).type(torch.complex64)
    m = m_real + (1.j)*m_img
    return m

def Complex1_h(a) :
    m_real = torch.Tensor(a).uniform_(-0.01,0.01).type(torch.complex64)
    m_img = torch.Tensor(a).normal_(0,0.02).type(torch.complex64)
    m = m_real + (1.j)*m_img
    return m

def Complex2_sh(a,b) :
    m_real = torch.Tensor(a,b).uniform_(-0.03,0.05).type(torch.complex64)
    m_img = torch.Tensor(a,b).normal_(0,0.02).type(torch.complex64)
    m = m_real + (1.j)*m_img
    return m

def Complex2_dh(a,b) :
    m_real = torch.Tensor(a,b).uniform_(-0.03,0.05).type(torch.complex64)
    m_img = torch.Tensor(a,b).normal_(0,0.1).type(torch.complex64)
    m = m_real + (1.j)*m_img
    return m

def Complex2_sa(a,b) :
    m_real = torch.Tensor(a,b).uniform_(-0.03,0.05).type(torch.complex64)
    m_img = torch.Tensor(a,b).normal_(0,0.02).type(torch.complex64)
    m = m_real + (1.j)*m_img
    return m


def Complex1_M1(a) :
    m_real = torch.Tensor(a).uniform_(-0.00,0.05).type(torch.complex64)
    m_img = torch.Tensor(a).normal_(0,0.2).type(torch.complex64)
    m = m_real + (1.j)*m_img
    return m

def Complex1_M2(a) :
    m_real = torch.Tensor(a).uniform_(-0.00,0.15).type(torch.complex64)
    m_img = torch.Tensor(a).normal_(0,0.2).type(torch.complex64)
    m = m_real + (1.j)*m_img
    return m


class NADO32(nn.Module):
    def __init__(
        self,
        system_level,
        spin_degree,
        environment_model,
        bath_number,
        nsgn,
        hidden_number,
        ancillary_number,
        device,
        table0
        ):
        super().__init__()
        self.Nv = system_level
        self.No = spin_degree
        self.M = environment_model
        self.Nb = bath_number
        self.nsgn = nsgn
        self.Nh = hidden_number
        self.Na = ancillary_number
        self.device = device
        self.table0 = torch.tensor(table0).to(self.device)

        self.Ns = self.No * self.Nv 
        self.Nd = nsgn * self.No * self.Nv * self.Nb * self.M 
        self.Nbs = self.No * self.Nv * self.Nb * self.M

        self.nstate = self.Ns*2 + self.Nd
        self.nparameters = 2*self.Ns+2*self.Nh+1*self.Nd+self.Na+2*(1*2*self.Nd*self.Ns+self.Ns*self.Nh+
                self.Ns*self.Na+2*self.Nh*self.Nd)+1*self.Na*self.Nd+0*self.Ns*self.Ns
        
        self.bs = nn.Parameter(Complex1_s(self.Ns))
        self.bd = nn.Parameter(torch.Tensor(self.Nd).type(torch.float32).uniform_(-1.2,-0.6))
        self.bsd0 = nn.Parameter(Complex2_sd(self.Ns, self.Nd))
        self.bsd1 = nn.Parameter(Complex2_sd(self.Ns, self.Nd))
        self.w_sh = nn.Parameter(Complex2_sh(self.Ns, self.Nh))
        self.bh = nn.Parameter(Complex1_h(self.Nh))
        self.w_sa = nn.Parameter(Complex2_sa(self.Ns, self.Na))
        self.w_dh1 = nn.Parameter(Complex2_dh(self.Nd, self.Nh))
        self.w_dh2 = nn.Parameter(Complex2_dh(self.Nd, self.Nh))
        self.w_da = nn.Parameter(torch.Tensor(self.Nd, self.Na).type(torch.float32).uniform_(-0.01,0.01))
        self.ba = nn.Parameter(torch.Tensor(self.Na).type(torch.float32).uniform_(-0.01,0.01))

        sparsity_rho0 = self.table0[:,0:self.Ns] - self.table0[:,self.Ns:]
        self.sparsity = torch.unique(sparsity_rho0,dim=0)
        
        self.count = 0


    def States(self, x) :
        self.count += x.shape[0]
        y,x1,x2 = torch.split(x,[self.Nd, self.Ns, self.Ns],dim=1)
        coef = torch.floor(torch.sum(y[:,:self.Nd//2], dim=1)/2)+torch.floor(torch.sum(y[:,self.Nd//2:], dim=1)/2)
        y,x1,x2 = y.type(torch.complex64), x1.type(torch.complex64), x2.type(torch.complex64)
        r1 = torch.exp(y.matmul(2*self.bd.type(torch.complex64)) + 
            x1.matmul(self.bs) + torch.conj(x2.matmul(self.bs)) + 
            torch.sum(x1.matmul(self.bsd0).multiply(y) + x2.matmul(self.bsd1).multiply(y),dim=1))
        r2 = torch.prod((1 + torch.exp(self.f3(x1, x2, y))),dim=1)*0.5**self.Na
        r3_l = torch.prod((1 + torch.exp(self.f1(x1, y))),dim=1)*0.5**self.Nh
        r3_r = torch.prod((1 + torch.exp(self.f2(x2, y))),dim=1)*0.5**self.Nh
        r0 = r1 * r2 * r3_l * r3_r
        y_c = y.clone()
        y_c[:,0:self.Nbs] = y[:,self.Nbs:self.Nd]
        y_c[:,self.Nbs:self.Nd] = y[:,0:self.Nbs]
        x1_c, x2_c = x2, x1
        r1_c = torch.exp(y_c.matmul(2*self.bd.type(torch.complex64)) + 
                x1_c.matmul(self.bs) + torch.conj(x2_c.matmul(self.bs)) + 
                torch.sum(x1_c.matmul(self.bsd0).multiply(y_c) + x2_c.matmul(self.bsd1).multiply(y_c),dim=1))
        r2_c = torch.prod((1 + torch.exp(self.f3(x1_c, x2_c, y_c))),dim=1)*0.5**self.Na
        r3_l_c = torch.prod((1 + torch.exp(self.f1(x1_c, y_c))),dim=1)*0.5**self.Nh
        r3_r_c = torch.prod((1 + torch.exp(self.f2(x2_c, y_c))),dim=1)*0.5**self.Nh
        r0_conj = torch.conj(r1_c * r2_c * r3_l_c * r3_r_c)
        rho_x = r0 + (-1)**(coef) * r0_conj
        return rho_x

    def Gradients_t(self,x):
        n = x.shape[0]
        gradients = torch.zeros((n,self.nparameters),dtype=torch.complex64,device=x.device)
        n_part = 50
        quot, remai = n//n_part, n%n_part
        left = 0
        for i in range(n_part) :
            count = 0
            if i<remai :  right = left + quot + 1
            else :  right = left + quot
            y = self.States(x[left:right])
            eye = torch.eye(y.shape[0], device=x.device)
            for name,p in self.named_parameters() :
                n_p = p.numel()
                p_grad_real = torch.autograd.grad(y.real, p, (eye,), retain_graph=True, is_grads_batched=True)[0].flatten(start_dim=1, end_dim=-1)
                self.zero_grad()
                p_grad_imag = torch.autograd.grad(y.imag, p, (eye,), retain_graph=True, is_grads_batched=True)[0].flatten(start_dim=1, end_dim=-1)
                self.zero_grad()
                if p.is_complex() :
                    gradients[left:right,count:count+n_p] = torch.real(p_grad_real)+1.j*torch.real(p_grad_imag)
                    gradients[left:right,count+n_p:count+2*n_p] = torch.imag(p_grad_real)+1.j*torch.imag(p_grad_imag)
                    count = count + 2*n_p
                else:
                    gradients[left:right,count:count+n_p] = p_grad_real+1.j*p_grad_imag
                    count = count + n_p
            left = right
        return gradients

    def State(self, x):    
        self.count += 1
        if not self.f0(x) :
            print(x)
            return 0.+0.j
        y,x1,x2 = torch.split(x,[self.Nd, self.Ns, self.Ns])
        coef = torch.floor(torch.sum(y[:self.Nd//2])/2)+torch.floor(torch.sum(y[self.Nd//2:])/2)
        y = y.type(torch.complex64)
        x1 = x1.type(torch.complex64)
        x2 = x2.type(torch.complex64)
        r1 = torch.exp(y.matmul((self.bd + self.bd).type(torch.complex64)) + 
                    x1.matmul(self.bs) + torch.conj(x2.matmul(self.bs)) + 
                    x1.matmul((self.bsd0).matmul(y)) + x2.matmul((self.bsd1).matmul(y)))
        r2 = torch.prod((1 + torch.exp(self.f3(x1, x2, y))))
        r3_l = torch.prod((1 + torch.exp(self.f1(x1, y))))
        r3_r = torch.prod((1 + torch.exp(self.f2(x2, y))))
        r0 = r1 * r2 * r3_l * r3_r
        y_c = y.clone()
        y_c[0:self.Nbs] = y[self.Nbs:self.Nd]
        y_c[self.Nbs:self.Nd] = y[0:self.Nbs]
        x2_c = x1
        x1_c = x2
        r1_c = torch.exp(y_c.matmul((self.bd+self.bd).type(torch.complex64)) + 
                    x1_c.matmul(self.bs) + torch.conj(x2_c.matmul(self.bs)) + 
                    x1_c.matmul(self.bsd0.matmul(y_c)) + x2_c.matmul(self.bsd1.matmul(y_c)))
        r2_c = torch.prod((1 + torch.exp(self.f3(x1_c, x2_c, y_c))))
        r3_l_c = torch.prod((1 + torch.exp(self.f1(x1_c, y_c))))
        r3_r_c = torch.prod((1 + torch.exp(self.f2(x2_c, y_c))))
        r0_conj = torch.conj(r1_c * r2_c * r3_l_c * r3_r_c)
        r = r0 + (-1)**coef * r0_conj
        return r * 0.5**(2*self.Nh+self.Na)

    def f1(self, x1, y):
        return  x1.matmul(self.w_sh) + y.matmul(self.w_dh1) + self.bh

    def f2(self, x2 ,y):
        return torch.conj(x2.matmul(self.w_sh) + y.matmul(self.w_dh2)) + torch.conj(self.bh)

    def f3(self, x1, x2, y):
        a = (2*self.ba.type(torch.complex64) + 
            x1.matmul(self.w_sa) + torch.conj(x2.matmul(self.w_sa)) + 
            y.matmul((self.w_da + self.w_da).type(torch.complex64)))
        return a

    def sigmoid(self,x):  
        return 1/(1+torch.exp(-x))

    def f0(self,x):
        Ns = self.Ns
        Nd = self.Nd
        No = self.No
        N = torch.div(Nd, 2, rounding_mode='floor')
        n_minus = torch.sum(x[0:N].reshape(Ns,-1),dim=(1)).flatten()
        n_plus = torch.sum(x[N:Nd].reshape(Ns,-1),dim=(1)).flatten()
        m_0 = x[Nd:Nd+Ns]
        m_1 = x[Nd+Ns:Nd+2*Ns]
        n_minus = n_minus.flip(0).reshape(No,-1).T.flatten()
        n_plus = n_plus.flip(0).reshape(No,-1).T.flatten()
        sparsity = self.sparsity.to(x.device)
        for i in range(0,self.sparsity.shape[0]):
            if all(n_minus - n_plus + m_0 - m_1 == sparsity[i]):
                return True
        return False

    def rbm_to_vec(self):
        vec = torch.zeros(self.nparameters,dtype=torch.float32) 
        i = 0
        j = 0
        for name,p in self.named_parameters():
            i = j
            n = p.numel()
            j = i + n
            if p.is_complex():
                vec_complex = p.detach().flatten()
                vec[i:j] = torch.real(vec_complex)
                i = j
                j = j + n
                vec[i:j] = torch.imag(vec_complex)
            else:
                vec[i:j] = p.detach().flatten()
        return vec

    def vec_to_rbm(self,vec):
        dict = self.state_dict()
        i = 0
        j = 0
        for para in dict:
            i = j
            n = dict[para].numel()
            j = i + n
            if dict[para].is_complex():
                a = vec[i:j].reshape(dict[para].shape)
                i = j
                j = j + n
                b = vec[i:j].reshape(dict[para].shape)
                dict[para] = a+1.0j*b
            else:
                dict[para] = vec[i:j].reshape(dict[para].shape)
            self.load_state_dict(dict)
        return self

    def rbmd_to_vec(self):
        gradient = torch.zeros(self.nparameters,dtype=torch.float32) 
        i = 0
        j = 0
        for name,p in self.named_parameters() :
            i = j
            n = p.numel()
            j = i + n
            if p.is_complex() :
                gradient_complex = p.grad.flatten()
                gradient[i:j] = torch.real(gradient_complex)
                i = j
                j = j + n
                gradient[i:j] = torch.imag(gradient_complex)
            else:
                gradient[i:j] = p.grad.flatten()
        return gradient

    def vec_to_rbmd(self,gradient):
        i = 0
        j = 0
        for name,p in self.named_parameters():
            i = j
            n = p.numel()
            j = i + n
            if p.is_complex():
                a = gradient[i:j].reshape(p.shape)
                i = j
                j = j + n
                b = gradient[i:j].reshape(p.shape)
                p.grad = a+1.0j*b
            else:
                p.grad = gradient[i:j].reshape(p.shape)
        return gradient
