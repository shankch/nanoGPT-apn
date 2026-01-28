# train a miniature character-level shakespeare model with APN
# APN (Attractor Patch Network) replaces the standard MLP FFN
# mirrors train_shakespeare_char.py but with APN enabled

out_dir = 'out-shakespeare-char-apn'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt-apn'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# APN configuration - replaces MLP FFN with Attractor Patch Network
use_apn = True
apn_K = 256      # number of prototypes/patches
apn_k = 4        # number of active patches per token (top-k)
apn_r = 32       # code dimension (rank of patch updates)
apn_tau = 0.07   # routing temperature
apn_gamma = 1.0  # residual scaling factor

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
