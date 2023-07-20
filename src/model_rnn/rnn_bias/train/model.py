import tensorflow as tf
from .hyper import *

__all__ = ['Model']

class Model(tf.Module):
	def __init__(self, hp=hp):
		super(Model, self).__init__()
		self._initialize_variable(hp)
		self.optimizer = tf.optimizers.Adam(hp['learning_rate'])

	@tf.function
	def __call__(self, trial_info, hp):
		y, loss = self._train_oneiter(trial_info['neural_input1'], trial_info['neural_input2'],
									  trial_info['desired_decision'], trial_info['desired_estim'],
									  trial_info['mask_decision'], trial_info['mask_estim'], hp)
		return y, loss

	@tf.function
	def rnn_model(self, input_data1, input_data2, hp):
		_h1 = tf.zeros((hp['batch_size_v'].shape[0], hp['w_rnn11'].shape[0]), tf.float32)
		_h2 = tf.zeros((hp['batch_size_v'].shape[0], hp['w_rnn22'].shape[0]), tf.float32)
		_n1 = _h1*0
		_n2 = _h2*0
		h1_stack    = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
		h2_stack    = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
		y_dm_stack  = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
		y_em_stack  = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
		i = 0
		for rnn_input1 in input_data1:
			rnn_input2  = input_data2[i]
			_n1, _n2, _h1, _h2 = self._rnn_cell(_n1, _n2, _h1, _h2, rnn_input1, rnn_input2, hp)
			h1_stack    = h1_stack.write(i, tf.cast(_h1, tf.float32))
			h2_stack    = h2_stack.write(i, tf.cast(_h2, tf.float32))

			if hp['w_out_dm_fix']: 
				y_dm_matmul = tf.cast(_h1+_h2, tf.float32) @ (hp['w_out_dm'] + self.var_dict['w_out_dm']*0)
			else: 
				y_dm_matmul = tf.cast(_h1+_h2, tf.float32) @ self.var_dict['w_out_dm']
			y_dm_stack  = y_dm_stack.write(i, y_dm_matmul)

			if hp['w_out_em_fix']: 
				y_em_matmul = tf.cast(_h1+_h2, tf.float32) @ (hp['w_out_em'] + self.var_dict['w_out_em']*0)
			else: 
				y_em_matmul = tf.cast(_h1+_h2, tf.float32) @ self.var_dict['w_out_em']
			y_em_stack  = y_em_stack.write(i, y_em_matmul)
			i += 1
		h1_stack   = h1_stack.stack()
		h2_stack   = h2_stack.stack()
		y_dm_stack = y_dm_stack.stack()
		y_em_stack = y_em_stack.stack()
		return y_dm_stack, y_em_stack, h1_stack, h2_stack

	def _train_oneiter(self, input_data1, input_data2, target_data_dm, target_data_em,  mask_dm, mask_em, hp):
		with tf.GradientTape() as t:
			_Ydm, _Yem, _H1, _H2 = self.rnn_model(input_data1, input_data2, hp)  # capitalized since they are stacked
			loss_dm = self._calc_loss(tf.cast(_Ydm,tf.float32), tf.cast(target_data_dm,tf.float32),tf.cast(mask_dm,tf.float32), hp)
			loss_em = self._calc_loss(tf.cast(_Yem,tf.float32), tf.cast(target_data_em,tf.float32),tf.cast(mask_em,tf.float32), hp)
			loss = hp['lam_decision'] * loss_dm + hp['lam_estim'] * loss_em

		vars_and_grads = t.gradient(loss, self.var_dict)
		capped_gvs = [] # gradient capping and clipping
		for var, grad in vars_and_grads.items():
			if hp['DtoE_off']:
				if var in ['w_rnn12']: grad *= 0

			if hp['EtoD_off']:
				if var in ['w_rnn21']: grad *= 0

			if grad is None:
				capped_gvs.append((grad, self.var_dict[var]))
			else:
				capped_gvs.append((tf.clip_by_norm(grad, hp['grad_max']), self.var_dict[var]))
		self.optimizer.apply_gradients(grads_and_vars=capped_gvs)
		return {'dm':_Ydm, 'em':_Yem}, {'loss':loss, 'loss_dm': loss_dm, 'loss_em': loss_em}

	def _initialize_variable(self,hp):
		_var_dict = {}
		for k, v in hp.items():
			if k[-1] == '0':
				name = k[:-1]
				_var_dict[name] = tf.Variable(hp[k], name=name, dtype='float32')
		self.var_dict = _var_dict

	def _calc_loss(self, y, target, mask, hp):
		loss = tf.reduce_mean(mask*(-target * tf.nn.log_softmax(y)))
		return loss

	def _rnn_cell(self, _n1, _n2, _h1, _h2, rnn_input1, rnn_input2, hp):
		_w_rnn11 = self.var_dict['w_rnn11']
		_w_rnn12 = self.var_dict['w_rnn12']
		_w_rnn21 = self.var_dict['w_rnn21']
		_w_rnn22 = self.var_dict['w_rnn22']

		if hp['w_in_dm_fix']: 
			_w_in1 = (hp['w_in1'] + self.var_dict['w_in1']*0)
		else:
			_w_in1 = self.var_dict['w_in1']
		
		if hp['w_in_em_fix']: 
			_w_in2 = (hp['w_in2'] + self.var_dict['w_in2']*0)
		else:
			_w_in2 = self.var_dict['w_in2']

		if hp['DtoE_off']: _w_rnn12 = _w_rnn12 * 0
		if hp['EtoD_off']: _w_rnn21 = _w_rnn21 * 0
            
        # Ornstein-Uhlenbeck noise model
		_n1 = _n1*tf.exp(-hp['alpha_noise']) + \
                tf.sqrt(1.-tf.exp(-2.*hp['alpha_noise']))*tf.random.normal(_h1.shape,0,hp['noise_rnn_sd'])
		_n2 = _n2*tf.exp(-hp['alpha_noise']) + \
                tf.sqrt(1.-tf.exp(-2.*hp['alpha_noise']))*tf.random.normal(_h2.shape,0,hp['noise_rnn_sd'])
	
		_h1 = _h1*(1. - hp['alpha_neuron']) + hp['alpha_neuron']*tf.nn.sigmoid(
                rnn_input1@_w_in1 + _h1@_w_rnn11 + _h2@_w_rnn21 + _n1)
                        
		_h2 = _h2*(1. - hp['alpha_neuron']) + hp['alpha_neuron']*tf.nn.sigmoid(
                rnn_input2@_w_in2 + _h1@_w_rnn12 + _h2@_w_rnn22 + _n2)
                        
		return _n1, _n2, _h1, _h2