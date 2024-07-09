"""class for fixed point analysis"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class FixedPoint_Joint(object):
    def __init__(self, model, device, gamma=0.01, lambda0=5,
                 speed_tor=1e-06, grad_tor=0.1, max_epochs=200000,
                 lr_decay_epoch=10000, nonlinearity='relu'):
        self.model = model
        self.device = device
        self.gamma = gamma
        self.lambda0 = lambda0
        self.speed_tor = speed_tor
        self.grad_tor = grad_tor
        self.max_epochs = max_epochs
        self.lr_decay_epoch = lr_decay_epoch
        if nonlinearity=='tanh':
            self.nonlinearity = F.tanh
        elif nonlinearity=='relu':
            self.nonlinearity = F.relu

        self.model.eval()

    def calc_speed(self, input_ff, target_fb, hiddenx, position ):
        # assumed a clamped / constant input
        # hidden x is the underlying voltage signal that is evolving
        # speed is calculated in observed rate variable change 
        # (thus subthreshold changes are not counted)
        
        #_,x1 = self.model(input_ff, input_fb, hiddenx )
        #hiddenr = self.nonlinearity(hiddenr)
        # these are all batch(=1) x dim tensors

        hiddenr = self.nonlinearity(hiddenx)
        dxdt = (- hiddenx + self.model.bias
                + hiddenr @ self.model.w_hh.weight.T 
                + input_ff @ self.model.w_in.weight.T 
                + (position-target_fb) @ self.model.w_fb.weight.T  )
        dpdt = hiddenr @ self.model.w_out.weight.T

        speed = torch.norm( dxdt ) + self.lambda0 * torch.norm(dpdt)    # joint system speed
        #print( 'dxdt = '+ np.str_(torch.norm(dxdt).item()) + ' dpdt = '+ np.str_(torch.norm(dpdt).item()) )


        return speed
    
    def calc_dxdt(self, input_ff, target_fb, hiddenx, position ):
        # assumed a clamped / constant input
        # hidden x is the underlying voltage signal that is evolving
        # speed is calculated in observed rate variable change 
        # (thus subthreshold changes are not counted)
        
        #_,x1 = self.model(input_ff, input_fb, hiddenx )
        #hiddenr = self.nonlinearity(hiddenr)
        # these are all batch(=1) x dim tensors

        hiddenr = self.nonlinearity(hiddenx)
        dxdt = (- hiddenx + self.model.bias
                + hiddenr @ self.model.w_hh.weight.T 
                + input_ff @ self.model.w_in.weight.T 
                + (position-target_fb) @ self.model.w_fb.weight.T  )
        

        return dxdt


    def find_fixed_point(self, init_hiddenx, init_hiddenp, const_ff, const_tgt, view=False):
        new_x1 = init_hiddenx.clone()
        new_p1 = init_hiddenp.clone()

        gamma = self.gamma
        result_ok = True
        ctr = 0
        const_ff.requires_grad=False
        const_tgt.requires_grad=False
        hiddenx = Variable(new_x1).to(self.device)
        hiddenx.requires_grad = True
        hiddenp = Variable(new_p1).to(self.device)
        hiddenp.requires_grad = True

        while True:    
            # ensure gradient computation turned on. speed gradient is on because it is a function of hiddenx and hiddenp
            #hiddenx = Variable(new_x1).to(self.device)
            #hiddenx.requires_grad = True
            #hiddenp = Variable(new_p1).to(self.device)
            #hiddenp.requires_grad = True
            
            # zero gradients
            #if ctr>0:
            #    hiddenx.grad.zero_()
            #    hiddenp.grad.zero_()

            hiddenx.requires_grad = True
            hiddenp.requires_grad = True
            speed = self.calc_speed(const_ff, const_tgt, hiddenx, hiddenp)
            speed.backward()
            dx = hiddenx.grad
            dp = hiddenp.grad

            #print( 'gradx = '+ np.str_(torch.norm(dx).item()) + ' gradp = '+ np.str_(torch.norm(dp).item()) )

            if view and ctr % 5000 == 0:
                print(f'epoch: {ctr}, speed={speed.item()}')
            if speed.item() < self.speed_tor:
                print(f'epoch: {ctr}, speed={speed.item()}')
                break
            elif torch.norm(dx) < self.grad_tor:
                print('stuck in plateau: jumping around') # speed hasn't decreased enough
                print(torch.norm(dx))
                gamma=self.gamma
                hiddenx = hiddenx + torch.randn(hiddenx.shape).to(self.device)*0.5
                hiddenp = hiddenp + torch.randn(hiddenp.shape).to(self.device)*0.5
            if ctr % self.lr_decay_epoch == 0 and ctr > 0:
                gamma *= 0.5
            if ctr == self.max_epochs:
                print(f'forcibly finished. speed={speed.item()}')
                result_ok = False
                break
            ctr += 1

            with torch.no_grad():
                hiddenx = hiddenx - gamma * dx
                hiddenp = hiddenp - gamma * dp
    
        fixed_point = hiddenx[0].detach() #
        fixed_pos = hiddenp[0].detach()
        return fixed_point, fixed_pos, result_ok

    def calc_input_jacobian(self, fixed_point_0, fixed_point_pos, const_ff, const_fb ):
        # F( x, u ) = dx/dt, where F describes voltage dynamics as a fn of state x and input u
        # At fixed point FP=(x0,u0), F(x=x0, u=u0) = 0
        # the input jacobian is [ dF/du]
        # we are evaluating this at the given state u=u0 (and x=x0)
        # this can be used to see which neurons go up and which go down.... 
        
        fixed_point = torch.unsqueeze( fixed_point_0, dim=0)
        position = torch.unsqueeze( fixed_point_pos, dim=0)
        input_ff = torch.unsqueeze( const_ff, dim=1 )
        input_ff = Variable(input_ff).to(self.device)
        input_ff.requires_grad = True 

        dxdt = self.calc_dxdt(input_ff.T, const_fb, fixed_point, position ) 
        
        #r0 = self.model.nonlinearity( x0 )
        #dxdt = self.model.alpha* (- fixed_point)
        #dxdt = (- x0 + self.model.W_rec.T @ r0 + self.model.W_in.T @ input  )

        #input_jacobian = torch.zeros( (self.model.n_in, self.model.n_hid) )
        input_jacobian = torch.zeros((self.model.n_in, self.model.n_hid)).to(self.device)
        # torch autograd is returning J.T * v where v is some vector, rather than directly giving us J.T
        # here, we use g(F)=F_i, s.t. v = dg/dF  = [0,..,0,1,0,..] 
        # by using these vectors v, we get one column (i) of the Jacobian at a time = [dF_i/dx_1, ... dF_i/dx_n]
        for i in range(self.model.n_hid):
            output = torch.zeros_like(dxdt).to(self.device)
            output[0,i] = 1. # vec = [0,.0,....1, ...] so that we get dl/dx_i
            input_jacobian[:, i:i+1] = torch.autograd.grad(dxdt, input_ff, grad_outputs=output, retain_graph=True)[0]

        input_jacobian = input_jacobian.numpy().T

        return input_jacobian
    


    def calc_hiddenx_jacobian(self, fixed_point_0, fixed_point_pos, const_ff, const_fb):
        # Unsqueeze to add batch dimension
        fixed_point = torch.unsqueeze(fixed_point_0, dim=0)
        position = torch.unsqueeze(fixed_point_pos, dim=0)
        
        hiddenx = Variable(fixed_point).to(self.device)
        hiddenx.requires_grad = True

        dxdt = self.calc_dxdt(const_ff, const_fb, hiddenx, position)

        # Initialize the hiddenx Jacobian with the correct shape
        hiddenx_jacobian = torch.zeros((self.model.n_hid, self.model.n_hid)).to(self.device)

        # Compute the Jacobian
        for i in range(self.model.n_hid):
            output = torch.zeros_like(dxdt).to(self.device)
            output[0, i] = 1.0  # Set the i-th element of the output gradient to 1
            
            # Compute the gradient of dxdt with respect to hiddenx
            hiddenx_jacobian[:, i]  = torch.autograd.grad(dxdt, hiddenx, grad_outputs=output, retain_graph=True, create_graph=True)[0]
            

        # Convert the hiddenx Jacobian to numpy
        hiddenx_jacobian = hiddenx_jacobian.T.cpu().detach().numpy()

        return hiddenx_jacobian


    '''
    def calc_jacobian(self, fixed_point_0, const_ff, const_fb):
        # F( x, u ) = dx/dt, where F describes voltage dynamics as a fn of state x and input u
        # At fixed point FP=(x0,u0), F(x=x0, u=u0) = 0
        # then the jacobian is [ dF/dx ]
        # for fixed point stability analysis, F(x=FP)=0 , jacobian is used to estimate stability
        
        fixed_point = torch.unsqueeze(fixed_point_0, dim=0)
        fixed_point = Variable(fixed_point).to(self.device)
        fixed_point.requires_grad = True # calculate gradients around fixed point

        # time * batch * neurons
        #input_signal = torch.unsqueeze(const_signal_tensor[0,0,:],dim=1) # use 1 time step
        
        #r0 = self.model.nonlinearity( fixed_point )
        #dxdt = self.model.alpha* (- fixed_point)
        dxdt = self.calc_speed(const_ff, const_fb, fixed_point)     # should be 1 timepoint

        jacobian = torch.zeros( (self.model.n_hid, self.model.n_hid) )
        # torch autograd is returning J.T * v where v is some vector, rather than directly giving us J.T
        # here, we use g(F)=F_i, s.t. v = dg/dF  = [0,..,0,1,0,..] 
        # by using these vectors v, we get one column (i) of the Jacobian at a time = [dF_i/dx_1, ... dF_i/dx_n]
        for i in range(self.model.n_hid):
            output = torch.zeros( (self.model.n_hid, 1) ).to(self.device)
            output[i] = 1. # vec = [0,.0,....1, ...] so that we get dl/dx_i
            jacobian[:, i:i+1] = torch.autograd.grad(dxdt, fixed_point, grad_outputs=output, retain_graph=True)[
                0]

        jacobian = jacobian.numpy().T

        return jacobian
    
    
    def calc_input_jacobian(self, fixed_point_0, const_ff, const_fb ):
        # F( x, u ) = dx/dt, where F describes voltage dynamics as a fn of state x and input u
        # At fixed point FP=(x0,u0), F(x=x0, u=u0) = 0
        # the input jacobian is [ dF/du]
        # we are evaluating this at the given state u=u0 (and x=x0)
        # this can be used to see which neurons go up and which go down.... 
        
        fixed_point = torch.unsqueeze( fixed_point_0, dim=1)
        input_ff = torch.unsqueeze( const_ff, dim=1 )
        input_ff = Variable(input).to(self.device)
        input_ff.requires_grad = True 

        dxdt = self.calc_speed(const_ff, const_fb, fixed_point) 
        
        #r0 = self.model.nonlinearity( x0 )
        #dxdt = self.model.alpha* (- fixed_point)
        #dxdt = (- x0 + self.model.W_rec.T @ r0 + self.model.W_in.T @ input  )

        input_jacobian = torch.zeros( (self.model.n_in, self.model.n_hid) )
        # torch autograd is returning J.T * v where v is some vector, rather than directly giving us J.T
        # here, we use g(F)=F_i, s.t. v = dg/dF  = [0,..,0,1,0,..] 
        # by using these vectors v, we get one column (i) of the Jacobian at a time = [dF_i/dx_1, ... dF_i/dx_n]
        for i in range(self.model.n_hid):
            output = torch.zeros( (self.model.n_hid, 1) ).to(self.device)
            output[i] = 1. # vec = [0,.0,....1, ...] so that we get dl/dx_i
            input_jacobian[:, i:i+1] = torch.autograd.grad(dxdt, input_ff, grad_outputs=output, retain_graph=True)[
                0]

        input_jacobian = input_jacobian.numpy().T

        return input_jacobian
    '''

    def flow_field( self, hiddenx, inputs ):
        # hiddenx is batch * neurons
        # inputs is batch * n_inputs

        hiddenr = self.nonlinearity(hiddenx)
        dxdt = (- hiddenx + hiddenr @ self.model.W_rec + inputs @ self.model.W_in  )

        return dxdt
    