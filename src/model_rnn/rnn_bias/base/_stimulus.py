import numpy as np
from ._parameters import par
import randomgen.generator as random

__all__ = ['Stimulus']

class Stimulus(object):
    """generates the combined DM-EM task stimuli
    """
    def __init__(self, par=par):
        self.set_params(par) # Equip the stimulus class with parameters
        self._generate_tuning() # Generate tuning/input config

    def set_params(self, par):
        for k, v in par.items():
            setattr(self, k, v)

    def generate_trial(self):
        stimulus                     = self._gen_stimseq()
        neural_input1, neural_input2 = self._gen_stims(stimulus)
        desired_output               = self._gen_output(stimulus)
        mask                         = self._gen_mask()
        return {'neural_input1'    : neural_input1.astype(np.float32),
                'neural_input2'    : neural_input2.astype(np.float32),
                'stimulus_ori'     : stimulus['stimulus_ori'],
                'reference_ori'    : stimulus['reference_ori'],
                'desired_decision' : desired_output['decision'].astype(np.float32),
                'desired_estim'    : desired_output['estim'].astype(np.float32),
                'mask_decision'    : mask['decision'].astype(np.float32),
                'mask_estim'       : mask['estim'].astype(np.float32)}

    def _gen_stimseq(self):
        stimulus_ori  = np.random.choice(np.arange(self.n_ori), p=self.stim_p, size=self.batch_size)
        reference_ori = np.random.choice(self.reference, p=self.ref_p, size=self.batch_size)
        return {'stimulus_ori': stimulus_ori, 'reference_ori': reference_ori}

    def _gen_stims(self, stimulus):
        neural_input1  = random.standard_normal(size=(self.n_timesteps, self.batch_size, self.n_input))*self.noise_sd*np.sqrt(2*self.tau/self.dt)
        neural_input2  = random.standard_normal(size=(self.n_timesteps, self.batch_size, self.n_input))*self.noise_sd*np.sqrt(2*self.tau/self.dt)

        for t in range(self.batch_size):
            neural_input2[self.design_rg['stim'],t] += self._gen_stim_variable(stimulus['stimulus_ori'][t]).reshape((1,-1))
            neural_input1[self.design_rg['decision'],t,(stimulus['stimulus_ori'][t]+stimulus['reference_ori'][t])%self.n_ori] += self.strength_ref

        return neural_input1, neural_input2

    def _gen_output(self, stimulus):
        desired_decision = np.zeros((self.n_timesteps,self.batch_size,self.n_output_dm), dtype=np.float32)
        desired_estim    = np.zeros((self.n_timesteps,self.batch_size,self.n_output_em), dtype=np.float32)
        for t in range(self.batch_size):
            desired_decision[self.dm_output_rg, t, int(0 < stimulus['reference_ori'][t])] += self.strength_decision
            desired_estim[self.em_output_rg, t] = self.tuning_output[:, stimulus['stimulus_ori'][t]].reshape((1, -1))
        return {'decision' : desired_decision, 'estim' : desired_estim}

    def _gen_mask(self):
        mask_decision = np.zeros((self.n_timesteps, self.batch_size, self.n_output_dm), dtype=np.float32)
        mask_estim    = np.zeros((self.n_timesteps, self.batch_size, self.n_output_em), dtype=np.float32)

        # set "specific" period
        for step in ['iti','stim','delay','decision','estim']:
            mask_decision[self.design_rg[step]] = self.mask_dm[step]
            mask_estim[self.design_rg[step]] = self.mask_em[step]
        return {'decision' : mask_decision, 'estim' : mask_estim}

    def _generate_tuning(self):
        _tuning_output = np.zeros((self.n_tuned_output, self.n_ori))
        stim_dirs = np.float32(np.arange(0,180,180/self.n_ori))
        pref_dirs = np.float32(np.arange(0,180,180/(self.n_ori)))
        for n in range(self.n_tuned_input):
            for i in range(self.n_ori):
                d = np.cos((stim_dirs[i] - pref_dirs[n])/90*np.pi)
                _tuning_output[n,i] = self.strength_output*np.exp(self.kappa*d)/np.exp(self.kappa)
        self.tuning_output = _tuning_output

    def _gen_stim_variable(self, stimulus_ori):
        stim_dirs = np.float32(np.arange(0,180,180/self.n_ori))
        stim_dir  = stim_dirs[stimulus_ori]
        stim_dir += random.standard_normal() * (self.noise_center[stimulus_ori])

        d = np.cos((stim_dirs - stim_dir)/90*np.pi)
        v = self.strength_input*np.exp(self.kappa*d)/np.exp(self.kappa)
        return v