from data import load_data
import config
import argparse
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from modules.denoising_diffusion import GaussianDiffusion
from modules.unet import Unet
from modules.trainer import Trainer
from modules.temporal_models import HistoryNet, CondNet
from torch.nn.parallel import DistributedDataParallel as DDP
import torch


parser = argparse.ArgumentParser(description="values from bash script")
parser.add_argument("--ndevice", type=int, required=True, help="number of cuda device")
parser.add_argument("--output_len", type=int, required=True, help="output length of one prediction")
args = parser.parse_args()

def schedule_func(ep):
    return max(config.decay ** ep, config.minf)

def main(rank, world_size):
    dist.init_process_group("gloo", init_method='env://', rank=rank, world_size=world_size)
    # dist.init_process_group(backend='nccl', init_method='env://', rank = torch.cuda.device_count(), world_size = 0)

    train_data, val_data = load_data(
        config.data_config,
        config.batch_size,
        pin_memory=True,
        num_workers=4,
        distributed=True
    ) # add for multi-cuda training

    denoise_model = Unet(
        dim=config.embed_dim,
        context_dim_factor=config.context_dim_factor,
        channels=config.data_config["img_channel"],
        dim_mults=config.dim_mults,
    )

    context_model = CondNet(
        dim=int(config.context_dim_factor * config.embed_dim),
        channels=config.data_config["img_channel"],
        backbone=config.backbone,
        dim_mults=config.dim_mults,
    )

    transform_model = (
        HistoryNet(
            dim=int(config.transform_dim_factor * config.embed_dim),
            channels=config.data_config["img_channel"],
            context_mode=config.transform_mode,
            dim_mults=config.transform_dim_mults,
            backbone=config.backbone
        )
        if config.transform_mode
        in ["residual"]
        else None
    )


    diffusion = GaussianDiffusion(
        denoise_fn=denoise_model,
        history_fn=context_model,
        transform_fn=transform_model,
        pred_mode=config.pred_mode,
        clip_noise=config.clip_noise,
        timesteps=config.iteration_step,
        loss_type=config.loss_type,
        aux_loss=False,
    ).to(rank)

    diffusion = DDP(diffusion, device_ids=[rank])
    print(os.path.join(config.result_root, f"{config.model_name}/"))
    trainer = Trainer(
        rank=rank,
        diffusion_model=diffusion,
        train_dl=train_data,
        val_dl=val_data,
        # sample_num_of_frame=config.init_num_of_frame + 1,
        sample_num_of_frame=args.output_len,
        init_num_of_frame=config.init_num_of_frame,
        scheduler_function=schedule_func,
        scheduler_checkpoint_step=config.scheduler_checkpoint_step,
        ema_decay=config.ema_decay,
        train_lr=config.lr,
        train_num_steps=config.n_step,
        step_start_ema=config.ema_start_step,
        update_ema_every=config.ema_step,
        save_and_sample_every=config.log_checkpoint_step,
        results_folder=os.path.join(config.result_root, f"{config.model_name}/"),
        tensorboard_dir=os.path.join(config.tensorboard_root, f"{config.model_name}/"),
        model_name=config.model_name,
        val_num_of_batch=config.val_num_of_batch,
        optimizer=config.optimizer
    )
    if config.load_model:
        trainer.load(idx=config.load_step, load_step=config.load_model)

    trainer.train()


if __name__ == "__main__":
    mp.spawn(main, args=(args.ndevice,), nprocs=args.ndevice, join=True)
    # main(args.ndevice, args.ndevice)
    dist.barrier()
    dist.destroy_process_group()
