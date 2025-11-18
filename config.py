# =========================
# Training Config
# =========================
n_step = 3000000
scheduler_checkpoint_step = 100000
# scheduler_checkpoint_step = 1
log_checkpoint_step = 1000
gradient_accumulate_every = 1
lr = 5e-5
decay = 0.8
minf = 0.2
ema_decay = 0.99
optimizer = "adam"  # adamw or adam
ema_step = 5
ema_start_step = 2000

# load config
load_model = False
# load_model = True
# load_step = 0

# diffusion config
loss_type = "l1"
iteration_step = 1000

context_dim_factor = 1
transform_dim_factor = 1
init_num_of_frame = 8
pred_mode = "noise"
clip_noise = True
transform_mode = "residual"
val_num_of_batch = 1
backbone = "resnet"
aux_loss = False
additional_note = ""

data_config = {
    "dataset_name": "satellite",
    "data_path": "satellite_data",
    "sequence_length": 24,
    "img_size": 256,
    "img_channel": 1,
    "add_noise": False,
    "img_hz_flip": False,
    'mode': 'train'
}

if data_config["img_size"] == 64:
    embed_dim = 48
    transform_dim_mults = (1, 2, 2, 4)
    dim_mults = (1, 2, 4, 8)
    batch_size = 24
elif data_config["img_size"] in [128, 256]:
    embed_dim = 64
    transform_dim_mults = (1, 2, 3, 4)
    dim_mults = (1, 1, 2, 2, 4, 4)
    batch_size = 1
else:
    raise NotImplementedError

model_name = f"{backbone}-{optimizer}-{pred_mode}-{loss_type}-{data_config['dataset_name']}-d{embed_dim}-t{iteration_step}-{transform_mode}-al{aux_loss}{additional_note}"

result_root = "./results"
tensorboard_root = result_root + '/tensorboard'