# =========================
# Test Config
# =========================
N_CONTEXT = 8
N_BATCH = 200          # Number of batches for validation
BATCH_SIZE = 1

optimizer = "adam"     # Options: "adam", "adamw"

# =========================
# Diffusion Config
# =========================
loss_type = "l1"
iteration_step = 1000
context_dim_factor = 1
transform_dim_factor = 1
init_num_of_frame = 4  # Number of initial frames for sampling
pred_modes = ["noise"] # Options: "noise", "pred_prev", "pred_true"
clip_noise = True
transform_modes = ["residual"]  # Options: "transform", "residual", "flow", "none", "ll_transform"
val_num_of_batch = 1
backbone = "resnet"
aux_loss = False
additional_note = ""

# =========================
# Data Configs
# =========================
data_configs = [
    {
        "dataset_name": "satellite",
        "data_path": "satellite_data",
        "sequence_length": 24,
        "img_size": 256,
        "img_channel": 1,
        "add_noise": False,
        "img_hz_flip": False,
        'mode': 'test'
    },
]

# =========================
# Paths
# =========================
result_root = "./results"
tensorboard_root = result_root + '/tensorboard'