install:
	pip install -e .

# preprocess
preprocess:
# [1] preprocess behavior data : combine behavior info from openneuro dataset
	python dynamic_bias/analyses/behavior/workflow/preprocess.py

# [2] preprocess fmri data : load fmriprep data, extract ROI voxels, and save the responses
	python dynamic_bias/analyses/fmri/workflow/preprocess.py

analysis-behavior:
# [1] estimate stimulus-specific biases from behavior data
	python dynamic_bias/analyses/behavior/workflow/stimulus_specific_bias.py

# [2] estimate decision-consistent biases from behavior data
	python dynamic_bias/analyses/behavior/workflow/decision_consistent_bias.py

analysis-ddm:
# [1] fit drift-diffusion models (DDMs) to behavior data
	python dynamic_bias/analyses/ddm/workflow/fit.py

# [2] analyze data from DDM behaviors
	python dynamic_bias/analyses/ddm/workflow/analyze_data.py

# [3] estimate decision-consistent biases from DDM behaviors
	python dynamic_bias/analyses/ddm/workflow/decision_consistent_bias.py

analysis-fmri:
# [1] decode BOLD data
	python dynamic_bias/analyses/fmri/workflow/decode.py

# [2] analyze data from decoded BOLD data
	python dynamic_bias/analyses/fmri/workflow/analyze_data.py

# [3] estimate hemodynamic model parameters from decoded BOLD data
	python dynamic_bias/analyses/fmri/workflow/hemodynamic_model.py

# [4] estimate decision-consistent biases from decoded BOLD data
	python dynamic_bias/analyses/fmri/workflow/decision_consistent_bias.py

# [5] compute correspondences with drift-diffusion models
	python dynamic_bias/analyses/fmri/workflow/drift_diffusion_model.py

analysis-rnn:
# [1] train RNNs
	@MODEL_TYPES="heterogeneous homogeneous heterogeneous_emonly heterogeneous_d2e_ablation"; \
	for model in $$MODEL_TYPES; do \
		python dynamic_bias/analyses/rnn/workflow/train.py --model_type $$model; \
	done

# [2] run RNNs
	@MODEL_TYPES="heterogeneous heterogeneous_emonly heterogeneous_d2e_ablation"; \
	for model in $$MODEL_TYPES; do \
		python dynamic_bias/analyses/rnn/workflow/run.py --model_type $$model; \
	done
	
# [3] analyze data from RNN predictions
	python dynamic_bias/analyses/rnn/workflow/analyze_data.py

# [4] estimate decision-consistent biases from RNN data
	python dynamic_bias/analyses/rnn/workflow/decision_consistent_bias.py

# [5] analyze RNN states
	python dynamic_bias/analyses/rnn/workflow/analyze_states.py