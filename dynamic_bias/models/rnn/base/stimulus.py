import numpy as np
from .parameters import par

__all__ = ['Stimulus']

class Stimulus(object):
    """generates the task stimulus
    """
    def __init__(self, par=par):
        self.set_params(par)    # Equip the stimulus class with parameters
        self._generate_tuning()

    def set_params(self, par):
        for k, v in par.items():
            setattr(self, k, v)

    def generate_trial(self):
        stimulus        = self._generate_stimseq()
        u_rho, u_the    = self._generate_stim(stimulus)
        desired_output  = self._generate_output(stimulus)
        mask            = self._generate_mask()
        return {'u_rho'    : u_rho.astype(np.float32),
                'u_the'    : u_the.astype(np.float32),
                'stimulus_ori'     : stimulus['stimulus_ori'],
                'reference_ori'    : stimulus['reference_ori'],
                'desired_decision' : desired_output['decision'].astype(np.float32),
                'desired_estim'    : desired_output['estim'].astype(np.float32),
                'mask_decision'    : mask['decision'].astype(np.float32),
                'mask_estim'       : mask['estim'].astype(np.float32)}

    def _generate_stimseq(self):
        stimulus_ori  = np.random.choice(np.arange(self.n_ori), p=self.stim_p, size=self.batch_size)
        reference_ori = np.random.choice(self.reference, p=self.ref_p, size=self.batch_size)
        return {'stimulus_ori': stimulus_ori, 'reference_ori': reference_ori}

    def _generate_stim(self, stimulus):
        """generate RNN inputs
            u_ρ : reference inputs
            u_θ : stimulus inputs
        """
        u_rho = np.random.normal(size=(self.n_timesteps, self.batch_size, self.n_input))*self.noise_sd*np.sqrt(2*self.tau/self.dt)
        u_the = np.random.normal(size=(self.n_timesteps, self.batch_size, self.n_input))*self.noise_sd*np.sqrt(2*self.tau/self.dt)

        for t in range(self.batch_size):
            u_the[self.design_rg['stim'],t] += self._gen_stim_variable(stimulus['stimulus_ori'][t]).reshape((1,-1))
            u_rho[self.design_rg['decision'],t,(stimulus['stimulus_ori'][t]+stimulus['reference_ori'][t])%self.n_ori] += self.gamma_ref
            
        return u_rho, u_the

    def _generate_output(self, stimulus):
        """generate desired RNN outputs
        """
        desired_decision = np.zeros((self.n_timesteps,self.batch_size,self.n_output_dm), dtype=np.float32)
        desired_estim    = np.zeros((self.n_timesteps,self.batch_size,self.n_output_em), dtype=np.float32)
        for t in range(self.batch_size):
            desired_decision[self.dm_output_rg, t, int(0 < stimulus['reference_ori'][t])] += 1.
            desired_estim[self.em_output_rg, t] = self.tuning_output[:, stimulus['stimulus_ori'][t]].reshape((1, -1))
        return {'decision' : desired_decision, 'estim' : desired_estim}

    def _generate_mask(self):
        mask_decision = np.zeros((self.n_timesteps, self.batch_size, self.n_output_dm), dtype=np.float32)
        mask_estim    = np.zeros((self.n_timesteps, self.batch_size, self.n_output_em), dtype=np.float32)

        # set "specific" period
        for step in ['iti','stim','delay','decision','estim']:
            mask_decision[self.design_rg[step]] = self.mask_dm[step]
            mask_estim[self.design_rg[step]] = self.mask_em[step]
        return {'decision' : mask_decision, 'estim' : mask_estim}

    def _generate_tuning(self):
        """generate tuning/input config"""
        _tuning_output = np.zeros((self.n_tuned_output, self.n_ori))
        stim_dirs = np.float32(np.arange(0,180,180/self.n_ori))
        pref_dirs = np.float32(np.arange(0,180,180/(self.n_ori)))
        for n in range(self.n_tuned_input):
            for i in range(self.n_ori):
                d = np.cos((stim_dirs[i] - pref_dirs[n])/90*np.pi)
                _tuning_output[n,i] = self.gamma_output*np.exp(self.kappa*d)/np.exp(self.kappa)
        self.tuning_output = _tuning_output

    def _gen_stim_variable(self, stimulus_ori):
        stim_dirs = np.float32(np.arange(0,180,180/self.n_ori))
        stim_dir  = stim_dirs[stimulus_ori]
        stim_dir += np.random.normal() * (self.noise_center[stimulus_ori])

        d = np.cos((stim_dirs - stim_dir)/90*np.pi)
        v = self.gamma_input*np.exp(self.kappa*d)/np.exp(self.kappa)
        return v