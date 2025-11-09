# training config
N_CONTEXT = 8
N_BATCH = 200  # 要超过之前的58条
BATCH_SIZE = 1

# n_step = 1000000
# scheduler_checkpoint_step = 100000
# log_checkpoint_step = 4000
# gradient_accumulate_every = 1
# lr = 5e-5
# decay = 0.8
# ema_decay = 0.99
optimizer = "adam"  # adamw or adam
# ema_step = 10
# ema_start_step = 2000

# diffusion config
loss_type = "l1"
iteration_step = 1000
context_dim_factor = 1
transform_dim_factor = 1
init_num_of_frame = 4  # for sampling initial condition
pred_modes = ["noise"]  # pred_prev or noise or pred_true
clip_noise = True
transform_modes = ["residual"]  # transform residual flow none ll_transform
val_num_of_batch = 1
backbone = "resnet"
aux_loss = False

additional_note = ""

# data config
data_configs = [
{
    "dataset_name": "satellite",
    "data_path": "satellite_data",
    "sequence_length": 24,
    "img_size": 256,
    "img_channel": 1,
    "add_noise": False,
    "img_hz_flip": False,
},
# {
#     "dataset_name": "city",
#     "data_path": "/extra/ucibdl0/shared/data",
#     "sequence_length": 20,
#     "img_size": 256,
#     "img_channel": 3,
#     "add_noise": False,
#     "img_hz_flip": False,
# },
]

result_root = "./results-mse-10-retrain-nature"
tensorboard_root = "./results-mse-10-retrain-nature"
