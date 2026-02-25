import torch
import av
import numpy as np
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
from pathlib import Path
from multiprocessing import Process, Queue, current_process
import tqdm
import csv
import json
import sys
import argparse


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`list[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def read_video(video_path):
    '''
    Read video frames with PyAV.
    Args:
        video_path (str): Path to the video file.
        indices (list[int]): List of frame indices to read.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    return read_video_pyav(container, indices)


# =========================
# GPU Worker
# =========================
def gpu_worker(gpu_id, task_queue, result_queue, model_id, batch_size=4):
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    print(f"[Process {gpu_id}] Loading model on {device}")

    processor = VideoLlavaProcessor.from_pretrained(model_id)
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to(device)

    processor.patch_size = model.config.vision_config.patch_size
    processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy

    model.eval()

    while True:
        batch_paths = []
        batch_videos = []

        for _ in range(batch_size):
            if task_queue.empty():
                break
            try:
                video_path = task_queue.get_nowait()
            except:
                break

            video = read_video(video_path)
            batch_videos.append(video)
            batch_paths.append(video_path)

        if not batch_videos:
            break

        prompt = "USER: <video>\n Describe the video in detail. Assistant:"

        inputs = processor(
            text=[prompt]*len(batch_videos),
            videos=batch_videos,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=200,
            )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        captions = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        for p, c in zip(batch_paths, captions):
            result_queue.put((p, c))


# =========================
# CSV Writer
# =========================
# def csv_writer(result_queue, output_csv, total_videos):

#     pbar = tqdm.tqdm(desc="Writing CSV", total=total_videos)

#     with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         writer.writerow(["file_name", "text"])

#         while True:
#             item = result_queue.get()
#             if item is None:
#                 break
#             writer.writerow(item)
#             pbar.update(1)
#     pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Captioning with VideoLLaVA")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for GPU workers")
    parser.add_argument("--root_dir", type=str, default="/mnt/hdd/dataset/MultiCamVideo-Dataset", help="Root directory containing video files")
    parser.add_argument("--split_json", type=str, default="./split_A.json", help="Path to the JSON file containing list of video paths")
    parser.add_argument("--output_csv", type=str, default="/mnt/hdd/dataset/MultiCamVideo-Dataset/metadata.csv", help="Path to the output CSV file")
    parser.add_argument("--model_id", type=str, default="LanguageBind/Video-LLaVA-7B-hf", help="Model ID for VideoLLaVA")
    args = parser.parse_args()

    # =========================
    # 1. 加载模型
    # =========================
    model_id = args.model_id

    root_dir = Path(args.root_dir)

    if args.split_json:
        with open(args.split_json, "r") as f:
            mp4_files = json.load(f)
        mp4_files = [Path(p) for p in mp4_files]
        output_csv = Path(args.output_csv).parent / f"metadata_{Path(args.split_json).stem}.csv"
    else:
        mp4_files = sorted(list(root_dir.rglob("*.mp4")))
        output_csv = Path(args.output_csv) 

    total_videos = len(mp4_files)
    print(f"Total mp4 files: {total_videos}")

    batch_size = args.batch_size

    task_queue = Queue()
    result_queue = Queue()

    for f in mp4_files:
        task_queue.put(f)

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs")
    processes = []
    for gpu_id in range(num_gpus):
        p = Process(
            target=gpu_worker,
            args=(gpu_id, task_queue, result_queue, model_id, batch_size)
        )
        p.start()
        processes.append(p)

    # # 启动 CSV 写线程
    # writer_thread = threading.Thread(
    #     target=csv_writer,
    #     args=(result_queue, output_csv, total_videos),
    #     daemon=True
    # )
    # writer_thread.start()

    # 主进程负责写 CSV + 进度条
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "text"])

        pbar = tqdm.tqdm(desc="Writing CSV", total=total_videos)

        finished = 0
        while finished < len(mp4_files):
            result = result_queue.get()
            writer.writerow(result)
            pbar.update(1)
            finished += 1

        pbar.close()

    for p in processes:
        p.join()

    print("All done.")

    # # 启动两个 GPU worker
    # workers = []
    # for gpu_id in [0, 1]:
    #     t = threading.Thread(
    #         target=gpu_worker,
    #         args=(gpu_id, task_queue, result_queue, model_id),
    #         daemon=True
    #     )
    #     t.start()
    #     workers.append(t)

    # for w in workers:
    #     w.join()

    # result_queue.put(None)
    # writer_thread.join()

    # print("Done.")

    # with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
    #     writer = csv.writer(csv_file)
    #     writer.writerow(["file_name", "text"])  # 写入表头

    # for mp4_file in tqdm.tqdm(mp4_files):  # 只处理前5个视频文件
    #     # =========================
    #     # 2. 读取视频帧
    #     # =========================
    #     tqdm.tqdm.write(f"Processing video: {mp4_file}")
    #     video = read_video(mp4_file)
    #     # =========================
    #     # 3. 构造 prompt
    #     # =========================
    #     prompt = "USER: <video>\n Describe the video in detail."

    #     inputs = processor(
    #         text=prompt,
    #         videos=video,  # batch size 1
    #         return_tensors="pt"
    #     ).to(model.device)
    #     print(f"Processed inputs: {inputs.keys()}")

    #     # =========================
    #     # 4. 推理生成 caption
    #     # =========================
    #     print("Generating caption...")
    #     with torch.no_grad():
    #         output_ids = model.generate(
    #             **inputs,
    #             max_new_tokens=200,
    #         )
        
    #     caption = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    #     print("\n===== Video Caption =====")
    #     with open(output_csv, mode='a', newline='', encoding='utf-8') as csv_file:
    #         writer = csv.writer(csv_file)
    #         writer.writerow([mp4_file, caption])
