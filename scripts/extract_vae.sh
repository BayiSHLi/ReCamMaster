#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1" 

nohup python batch_data_process.py \
--data_root /mnt/hdd/dataset/MultiCamVideo-Dataset \
--metadata_csv metadata_split_ws2.csv \
--batch_size 6 \
--num_workers 4 \
> data_process.log 2>&1 &

echo "Data process started in background."
echo "Log file: data_process.log"