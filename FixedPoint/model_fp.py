"""define recurrent neural networks"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN_simple(nn.Module):
    def __init__(self, n_in, n_out, n_hid, device,
                 activation='relu', sigma=0.05, alpha=0.2, use_bias=True):
        super(RNN_simple, self).__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_error = n_out
        self.alpha = alpha
        self.w_in = nn.Linear(self.n_in, self.n_hid, bias=False)
        self.w_in.weight.requires_grad = False
        self.w_hh = nn.Linear(self.n_hid, self.n_hid, bias=False) # bias added separately
        self.w_hh.weight.requires_grad = False
        self.w_out = nn.Linear(self.n_hid, self.n_out, bias=False)
        self.w_out.weight.requires_grad = False
        self.w_fb = nn.Linear(self.n_error, self.n_hid, bias=False)
        self.w_fb.weight.requires_grad = False
        self.bias = nn.Parameter(torch.zeros((1,n_hid)))
        self.bias.requires_grad = False

        self.activation = activation
        self.sigma = sigma
        self.device = device
        self.nonlinearity = torch.relu

    def forward(self, ff_input, fbk_input, hidden, position=None):
        # 1-step forward
        ## Input is of size B x N_inputs
        # error is of size B x N_fbk
        # hidden is of size B x N_hid
        num_batch = ff_input.size(0)
        output_list = torch.zeros(num_batch, self.n_out).type_as(ff_input.data)

        input_ff = ff_input[:,:self.n_in]
        #target = ff_input[:,self.n_in:self.n_in+self.n_out]
        #output_prev = ff_input[:,self.n_in+self.n_out:]
        #error = output_prev - target

        rate = self.nonlinearity( hidden )
        pre_activates = - hidden + self.bias+ self.w_in(input_ff) + self.w_hh(rate) + self.w_fb(fbk_input)
        hidden = hidden+ self.alpha*pre_activates

        rate = self.nonlinearity( hidden )
        output_list = self.w_out(rate) * 1/1000
        if position is not None:
            position = position+output_list
            return output_list, hidden, rate, position
        else:
            return output_list, hidden, rate#, position
    
    def load_parameters( self, params ):

        dic = self.state_dict()
        dic['w_in.weight'] = torch.Tensor(params['W_in_0'].T).to(device=self.device)
        dic['w_hh.weight'] = torch.Tensor(params['W_rec_0'].T).to(device=self.device)
        dic['w_fb.weight'] = torch.Tensor(params['W_fbk_0'].T).to(device=self.device)
        dic['w_out.weight'] = torch.Tensor(params['W_out_0'].T).to(device=self.device)
        dic['bias'] = torch.Tensor(params['bias_n']).to(device=self.device)
        '''
        self.w_in.weight.copy_(torch.Tensor(params['W_in_0'].T).to(device=self.device))
        self.w_hh.weight.copy_(torch.Tensor(params['W_rec_0'].T).to(device=self.device))
        self.w_fb.weight.copy_(torch.Tensor(params['W_fbk_0'].T).to(device=self.device))
        self.w_out.weight.copy_(torch.Tensor(params['W_out_0'].T).to(device=self.device))
        '''
        self.load_state_dict( dic )

