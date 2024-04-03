import torch
import torch.nn as nn
import numpy

class LNDS(nn.Module):

    def __init__(self, n_states=3, n_input=5, n_obs=3, nonlinearity=None,
                 A0=None, B0=None, C0=None, x0=None, d0=None, batch=1,
                 A_sigma=0.1, B_sigma=0.1, C_sigma=0.1, x0_sigma=1, d_sigma=0.1,
                 device='cpu', tau=1 ):
        '''
        Simple linear dynamical system with potentially non-linear observations
        Implements the following generative process:

        dX/dt = A X(t-1) + B U(t)
        Y(t) = f( C X(t) + d )

        where A is the dynamics state space, U is input timeseries,
        A and B are intrinsic and input weights
        Y is the observable, C the observation matrix and f the potential observation nonlinearity

        
        '''
        super().__init__()
        self.n_states = n_states
        self.n_input = n_input
        self.n_obs = n_obs
        self.device = device
        self.tau = tau
        
        self.nonlinearity = self.set_nonlinearity(nonlinearity)

        # Parameters:
        self.x0 = torch.nn.Parameter( torch.Tensor(size=(1,batch,self.n_states)).to(device=device) )
        self.A = torch.nn.Parameter( torch.Tensor(size=(n_states, n_states)).to(device=device) )
        self.B = torch.nn.Parameter( torch.Tensor(size=(n_input, n_states)).to(device=device)  )
        self.C = torch.nn.Parameter( torch.Tensor(size=(n_states, n_obs)).to(device=device)  )
        self.d = torch.nn.Parameter( torch.Tensor(size=(1, n_obs)).to(device=device)  )

        # Initialise:
        torch.random.seed()
        with torch.no_grad():
            if A0 is None:
                self.A.copy_(torch.normal(mean=0,std=A_sigma,size=self.A.shape,device=device)-torch.eye(n=n_states, device=device))
                
            else:
                self.A.copy_( A0 )
            if B0 is None:
                self.B.normal_(std = B_sigma)
            else:
                self.B.copy_( B0 )
            if C0 is None:
                self.C.normal_(std = C_sigma)
            else:
                self.C.copy_( C0 )
            if x0 is None:
                self.x0.normal_(std = x0_sigma)
            else:
                self.x0.copy_( x0 )
            if d0 is None:
                self.d.normal_(std = d_sigma)
            else:
                self.d.copy_( d0 )
        
    def set_nonlinearity(self, nonlinearity ):
        if nonlinearity=='tanh':
            return nn.functional.tanh
        elif nonlinearity=='relu':
            return nn.functional.relu
        elif nonlinearity=='logsigmoid': 
            return nn.functional.logsigmoid
        elif nonlinearity is None:
            return noNonlin
        

    def fstep( self, ui, xi, dt ):
        xi2 = xi+ dt*( xi @ self.A + ui @ self.B)               # latent linear dynamics
        yi2 = self.nonlinearity( xi2 @ self.C + self.d )        # observation - linear transformation+nonlinearity 
        return xi2, yi2


    def forward(self, input, x0=None, dt=0.1 ):
        tm, batch, _ = input.shape

        xt = torch.Tensor(size=(tm,batch,self.n_states)).to(device=self.device)
        yt = torch.Tensor(size=(tm,batch,self.n_obs)).to(device=self.device)

        xi2 = torch.randn(size=(1,batch,self.n_states)).to(device=self.device) #self.x0.copy_()
        xi2[0,:,:] = self.x0
        for jj in range(tm):
            xi2, yi2 = self.fstep( input[jj] , xi2, dt )
            xt[jj] = xi2
            yt[jj] = yi2

        return xt, yt
    


def noNonlin( x ):
    ''' No nonlinearity - identity transfer function '''
    return x