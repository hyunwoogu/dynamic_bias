import torch
import torch.nn as nn
from ..hyper import *

__all__ = ['Model']

class Model(nn.Module):
    def __init__(self, hp=hp):
        super(Model, self).__init__()
        self._initialize_variable(hp)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=hp['learning_rate'])

    def forward(self, trial_info, hp):
        y, loss = self._train_oneiter(trial_info['u_rho'], trial_info['u_the'],
                                      trial_info['desired_decision'], trial_info['desired_estim'],
                                      trial_info['mask_decision'], trial_info['mask_estim'], hp)
        return y, loss

    def rnn_model(self, input_data1, input_data2, hp):
        _h1 = torch.zeros((input_data1.shape[1], hp['w_rnn11'].shape[0]))
        _h2 = torch.zeros((input_data1.shape[1], hp['w_rnn22'].shape[0]))
        _n1 = _h1.clone().zero_()
        _n2 = _h2.clone().zero_()
        h1_stack, h2_stack, y_dm_stack, y_em_stack = [], [], [], []

        for i, rnn_input1 in enumerate(input_data1):
            rnn_input2 = input_data2[i]
            _n1, _n2, _h1, _h2 = self._rnn_cell(_n1, _n2, _h1, _h2, rnn_input1, rnn_input2, hp)
            h1_stack.append(_h1)
            h2_stack.append(_h2)

            if hp['w_out_dm_fix']:
                y_dm_matmul = (_h1 + _h2) @ hp['w_out_dm']
            else:
                y_dm_matmul = (_h1 + _h2) @ self.var_dict['w_out_dm']
            y_dm_stack.append(y_dm_matmul)

            if hp['w_out_em_fix']:
                y_em_matmul = (_h1 + _h2) @ hp['w_out_em']
            else:
                y_em_matmul = (_h1 + _h2) @ self.var_dict['w_out_em']
            y_em_stack.append(y_em_matmul)

        return torch.stack(y_dm_stack), torch.stack(y_em_stack), torch.stack(h1_stack), torch.stack(h2_stack)

    def _train_oneiter(self, input_data1, input_data2, target_data_dm, target_data_em, mask_dm, mask_em, hp):
        self.optimizer.zero_grad()
        _Ydm, _Yem, _H1, _H2 = self.rnn_model(input_data1, input_data2, hp)
        loss_dm = self._calc_loss(_Ydm, target_data_dm, mask_dm, hp)
        loss_em = self._calc_loss(_Yem, target_data_em, mask_em, hp)
        loss = hp['lam_decision'] * loss_dm + hp['lam_estim'] * loss_em
        loss.backward()

        # Gradient capping and clipping
        for param in self.parameters():
            if param.grad is not None:
                torch.nn.utils.clip_grad_norm_(param, hp['grad_max'])

        self.optimizer.step()
        return {'dm': _Ydm, 'em': _Yem}, {'loss': loss, 'loss_dm': loss_dm, 'loss_em': loss_em}

    def _initialize_variable(self, hp):
        self.var_dict = nn.ParameterDict()
        for k, v in hp.items():
            if k[-1] == '0':
                name = k[:-1]
                self.var_dict[name] = nn.Parameter(torch.tensor(hp[k], dtype=torch.float32))

    def _calc_loss(self, y, target, mask, hp):
        loss = torch.mean(mask * (-target * torch.nn.functional.log_softmax(y, dim=-1)))
        return loss
    
    def _blockwise_transpose(self, m):
        mT = m*0
        mT[:24,:24] = m[:24,:24].T
        mT[:24,24:] = m[:24,24:].T
        mT[24:,:24] = m[24:,:24].T
        mT[24:,24:] = m[24:,24:].T
        
        return mT

    def _rnn_cell(self, _n1, _n2, _h1, _h2, rnn_input1, rnn_input2, hp):
        _w_rnn11 = self.var_dict['w_rnn11']
        _w_rnn12 = self.var_dict['w_rnn12']
        _w_rnn21 = self.var_dict['w_rnn21']
        _w_rnn22 = self.var_dict['w_rnn22']

        if hp['w_in_dm_fix']:
            _w_in1 = hp['w_in1']
        else:
            _w_in1 = self.var_dict['w_in1']

        if hp['w_in_em_fix']:
            _w_in2 = hp['w_in2']
        else:
            _w_in2 = self.var_dict['w_in2']

        if hp['DtoE_off']: _w_rnn12 = _w_rnn12 * 0
        if hp['EtoD_off']: _w_rnn21 = _w_rnn21 * 0

        # Ornstein-Uhlenbeck noise model
        _n1 = _n1*torch.exp(-hp['alpha_noise']) + \
              torch.sqrt(1.-torch.exp(-2.*hp['alpha_noise']))*torch.normal(0,hp['noise_rnn_sd'],_h1.shape)
        _n2 = _n2*torch.exp(-hp['alpha_noise']) + \
              torch.sqrt(1.-torch.exp(-2.*hp['alpha_noise']))*torch.normal(0,hp['noise_rnn_sd'],_h2.shape)

        _h1 = _h1*(1. - hp['alpha_neuron']) + hp['alpha_neuron']*torch.sigmoid(
                rnn_input1 @ _w_in1 + _h1 @ _w_rnn11 + _h2 @ _w_rnn21 + _n1)

        _h2 = _h2*(1. - hp['alpha_neuron']) + hp['alpha_neuron']*torch.sigmoid(
                rnn_input2 @ _w_in2 + _h1 @ _w_rnn12 + _h2 @ _w_rnn22 + _n2)

        return _n1, _n2, _h1, _h2