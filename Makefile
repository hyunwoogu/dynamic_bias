# add path
export PYTHONPATH := $(PYTHONPATH):$(abspath $(CURDIR))

# initialization
init:
	pip install -r requirements.txt --no-cache-dir

# preprocessing

# 
analysis-behavior:
	python src/analysis_behavior/psychometric_curve.py
	python src/analysis_behavior/stimulus_specific_bias_pse.py
	python src/analysis_behavior/stimulus_specific_bias.py
	python src/analysis_behavior/stimulus_specific_bias_weight.py

analysis-fmri:
	python src/analysis_fmri/decode.py
	python src/analysis_fmri/analysis_bias_bold.py
	python src/analysis_fmri/visual_drive.py
	python src/analysis_fmri/decision_consistent_bias.py
	python src/analysis_fmri/correspondence_score.py

model-ddm:
	python src/model_ddm/fit_models.py
	python src/model_ddm/near_reference_variability.py
	python src/model_ddm/post_decision_bias.py
	python src/model_ddm/decision_consistent_bias.py
	python src/model_ddm/analyze_bias_ddm.py

model-rnn:
	python src/model_rnn/train_models.py
	python src/model_rnn/run_models.py
	python src/model_rnn/sum_models.py
	python src/model_rnn/analysis_models.py
