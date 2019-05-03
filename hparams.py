# Audio:
num_mels = 80
num_freq = 1025
sample_rate = 20000
frame_length_ms = 50
frame_shift_ms = 12.5
preemphasis = 0.97
min_level_db = -100
ref_level_db = 20
griffin_lim_iters = 60
power = 1.5
signal_normalization = True
use_lws = False

# Text
text_cleaners = ['english_cleaners']

# Train
batch_size = 6
epochs = 10000
dataset_path = "dataset"
learning_rate = 1e-3
weight_decay = 1e-6
checkpoint_path = "./model_new"
grad_clip_thresh = 1.0
decay_step = [100000, 200000, 300000]
save_step = 5
log_step = 5
clear_Time = 20

# Model-Tacotron
hidden_size = 128
embedding_size = 256
teacher_forcing_ratio = 1.0
max_iters = 200
outputs_per_step = 5
