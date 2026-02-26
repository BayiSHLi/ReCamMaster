#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1" 

nohup python train_recammaster.py \
--task data_process \
--dataset_path /mnt/hdd/dataset/MultiCamVideo-Dataset \
--output_path ./ckpts \
--text_encoder_path "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
--vae_path "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
--tiled \
--num_frames 81 \
--height 480 \
--width 832 \
--dataloader_num_workers 2 \
> data_process.log 2>&1 &

echo "Data process started in background."
echo "Log file: data_process.log"