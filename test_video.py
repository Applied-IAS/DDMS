import os
import argparse
import collections
import torch
from joblib import Parallel, delayed
from torchvision.utils import save_image
from torchvision.io import write_video

from data import load_data
import config_test as config
from modules.denoising_diffusion import GaussianDiffusion
from modules.unet import Unet
from modules.temporal_models import HistoryNet, CondNet

parser = argparse.ArgumentParser(description="values from bash script")
parser.add_argument("--device", type=int, required=True, help="cuda device")
args = parser.parse_args()


def get_dim(data_config):
    return 48 if data_config["img_size"] == 64 else 64


def get_transform_mults(data_config):
    return (1, 2, 3, 4) if data_config["img_size"] in [128, 256] else (1, 2, 2, 4)


def get_main_mults(data_config):
    return (1, 1, 2, 2, 4, 4) if data_config["img_size"] in [128, 256] else (1, 2, 4, 8)


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def save_images_and_video(batch, base_path, prefix=""):
    """Save a sequence of images and a video from a batch tensor"""
    batch = batch.cpu()
    Parallel(n_jobs=4)(
        delayed(save_image)(frame, os.path.join(base_path, f"{prefix}{j}.png"))
        for j, frame in enumerate(batch)
    )
    write_video(
        os.path.join(base_path, f"{prefix}video.mp4"),
        torch.round(255 * batch.permute(0, 2, 3, 1)).expand(-1, -1, -1, 3),
        fps=4,
    )


for data_config in config.data_configs:
    train_data, val_data = load_data(
        data_config, config.BATCH_SIZE, pin_memory=False, num_workers=8, distributed=False
    )

    for pred_mode in config.pred_modes:
        for transform_mode in config.transform_modes:
            model_name = f"{config.backbone}-{config.optimizer}-{pred_mode}-l1-{data_config['dataset_name']}-d{get_dim(data_config)}-t{config.iteration_step}-{transform_mode}-al{config.aux_loss}{config.additional_note}"
            results_folder = os.path.join(config.result_root, model_name)

            loaded_param = torch.load(
                os.path.join(results_folder, f"{model_name}_9.pt"),
                map_location=lambda storage, loc: storage,
            )

            denoise_model = Unet(
                dim=get_dim(data_config),
                context_dim_factor=config.context_dim_factor,
                channels=data_config["img_channel"],
                dim_mults=get_main_mults(data_config),
            )

            context_model = CondNet(
                dim=int(config.context_dim_factor * get_dim(data_config)),
                channels=data_config["img_channel"],
                backbone=config.backbone,
                dim_mults=get_main_mults(data_config),
            )

            transform_model = None
            if transform_mode in ["residual", "transform", "flow", "ll_transform"]:
                transform_model = HistoryNet(
                    dim=int(config.transform_dim_factor * get_dim(data_config)),
                    channels=data_config["img_channel"],
                    context_mode=transform_mode,
                    backbone=config.backbone,
                    dim_mults=get_transform_mults(data_config),
                )

            model = GaussianDiffusion(
                denoise_fn=denoise_model,
                history_fn=context_model,
                transform_fn=transform_model,
                pred_mode=pred_mode,
                clip_noise=config.clip_noise,
                timesteps=config.iteration_step,
                loss_type=config.loss_type,
                aux_loss=config.aux_loss
            )

            N_SAMPLED = 12 if 'kth' in data_config["dataset_name"] else 16

            # Remove 'module.' prefix if present
            new_state_dict = collections.OrderedDict()
            for k, v in loaded_param["model"].items():
                new_state_dict[k[7:] if k.startswith("module.") else k] = v
            model.load_state_dict(new_state_dict)

            print(f"Loaded model: {model_name}")
            model.eval()
            model.to(args.device)

            count = 0
            for batch_idx, batch in enumerate(val_data):
                if batch_idx >= config.N_BATCH:
                    break

                # =========================
                # Save ground truth
                # =========================
                truth_base = os.path.join("../results/evaluate/truth", model_name)
                ensure_dirs(truth_base)

                for seq_idx, seq in enumerate(batch[config.N_CONTEXT:config.N_CONTEXT + N_SAMPLED].transpose(0, 1)):
                    prefix = f"{batch_idx}-{seq_idx}-"
                    save_images_and_video(seq, truth_base, prefix=prefix)

                # =========================
                # Prepare batch for sampling
                # =========================
                batch = ((batch - 0.5) * 2.0).to(args.device)
                sampled = model.sample(init_frames=batch[:config.N_CONTEXT], num_of_frames=N_SAMPLED)[0].transpose(0, 1)
                sampled = ((sampled + 1.0) / 2.0)[:, config.N_CONTEXT:, :, :, :]
                sampled = sampled.clamp(0, 1)

                # =========================
                # Save generated results
                # =========================
                gen_base = os.path.join("../results/evaluate/generated", model_name)
                for seq in sampled:
                    pred_path = os.path.join(gen_base, "pred", str(count))
                    video_path = os.path.join(gen_base, "video", str(count))
                    ensure_dirs(pred_path, video_path)

                    save_images_and_video(seq, pred_path)
                    write_video(
                        os.path.join(video_path, "video.mp4"),
                        torch.round(255 * seq.permute(0, 2, 3, 1)).expand(-1, -1, -1, 3),
                        fps=4
                    )

                    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                    print(f"[INFO] Finished generating sample #{count}")
                    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

                    count += 1

