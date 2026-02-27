import os
import argparse
from pathlib import Path

from diffsynth import WanVideoReCamMasterPipeline, ModelManager
import torch
import lightning as pl
from PIL import Image
import pandas as pd
import cv2
import av
import numpy as np
from einops import rearrange
from concurrent.futures import ThreadPoolExecutor
from pytorch_lightning.callbacks import TQDMProgressBar


def get_pth_path(video_path):
    video_path = Path(video_path)
    scene_dir = video_path.parent.parent  
    precomputed_dir = scene_dir / "precomputed"
    precomputed_dir.mkdir(parents=True, exist_ok=True)
    pth_path = precomputed_dir / (video_path.stem + ".tensors.pth")
    return pth_path

class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=False):
        metadata = pd.read_csv(metadata_path)
        # 过滤掉已存在的 .tensors.pth 文件
        file_paths = [os.path.join(base_path, "train", f) for f in metadata["file_name"]]
        mask = [not get_pth_path(p).exists() for p in file_paths]

        filtered = metadata[mask].reset_index(drop=True)
        self.path = [file_paths[i] for i, m in enumerate(mask) if m]
        self.text = filtered["text"].to_list()

        # self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        # self.text = metadata["text"].to_list()
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v


    def crop_and_resize_np(self, img):
        # img: numpy HWC uint8

        h, w, _ = img.shape
        scale = max(self.width / w, self.height / h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # center crop
        start_x = (new_w - self.width) // 2
        start_y = (new_h - self.height) // 2

        img = img[start_y:start_y+self.height,
                start_x:start_x+self.width]

        return img


    def load_frames_using_pyav(
        self,
        file_path,
        max_num_frames,
        start_frame_id,
        interval,
        num_frames,
    ):
        try:
            container = av.open(file_path)
        except Exception as e:
            print(f"[Decode Error] {file_path}: {e}")
            return None

        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        total_frames = stream.frames

        # 某些编码格式可能为 0
        if total_frames == 0 and stream.duration is not None:
            total_frames = int(stream.duration * stream.average_rate)

        if (
            total_frames is None
            or total_frames < max_num_frames
            or total_frames - 1 < start_frame_id + (num_frames - 1) * interval
        ):
            container.close()
            return None
    
        target_ids = [
            start_frame_id + i * interval
            for i in range(num_frames)
        ]
        target_ptr = 0
        max_target_id = target_ids[-1]

        frames = []
        first_frame = None
        current_id = 0

        for frame in container.decode(stream):

            if current_id > max_target_id:
                break

            if current_id == target_ids[target_ptr]:
                
            
                img = frame.to_ndarray(format="rgb24")
                img = self.crop_and_resize_np(img)

                if first_frame is None:
                    first_frame = np.array(img)

                img_tensor = torch.from_numpy(img) 
                frames.append(img_tensor.permute(2, 0, 1)) # HWC -> CHW

                target_ptr += 1

                if target_ptr == len(target_ids):
                    break

            current_id += 1

        container.close()

        if len(frames) != num_frames:
            return None

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames
        

    def load_video(self, file_path):
        start_frame_id = 0
        frames = self.load_frames_using_pyav(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = np.array(frame) 
        frame = self.crop_and_resize_np(frame)
        first_frame = frame
        frame = torch.from_numpy(frame)            # HWC uint8 tensor
        frame = frame.permute(2, 0, 1)            # HWC -> CHW
        frame = frame.unsqueeze(1)                 # CHW -> C 1 H W
        return frame


    def __getitem__(self, data_id):
        while True:
            try:
                text = self.text[data_id]
                path = self.path[data_id]
                if self.is_image(path):
                    if self.is_i2v:
                        raise ValueError(f"{path} is not a video. I2V model doesn't support image-to-image training.")
                    video = self.load_image(path)
                else:
                    video = self.load_video(path)
                if self.is_i2v:
                    video, first_frame = video
                    data = {"text": text, "video": video, "path": path, "first_frame": first_frame}
                else:
                    data = {"text": text, "video": video, "path": path}
                break
            except:
                data_id += 1
        return data
    

    def __len__(self):
        return len(self.path)



class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager)
        self.pipe.vae = torch.compile(self.pipe.vae, mode="reduce-overhead")
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

    def test_step(self, batch, batch_idx):
        self.pipe.device = self.device
        
        text = batch["text"]
        path = batch["path"]
        videos = batch["video"]  # 保持 5D shape: 1 C T H W

        # prompt
        prompt_emb = self.pipe.encode_prompt(text)
        
        # video
        videos = videos.to(self.pipe.device, non_blocking=True)
        if videos.dtype == torch.uint8:
            videos = videos.to(self.pipe.torch_dtype) * (1/127.5) - 1.0
        else:
            videos = videos.to(self.pipe.torch_dtype)
        # latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
        latents = self.pipe.vae.batch_encode(videos, self.device)
        
        # image
        if "first_frame" in batch:
            first_frame = batch["first_frame"].to(self.pipe.device, non_blocking=True)
            if first_frame.dtype == torch.uint8:
                first_frame = first_frame.to(self.pipe.torch_dtype) * (1/127.5) - 1.0 
            else:
                first_frame = first_frame.to(self.pipe.torch_dtype)
        else:
            first_frame = None
        
        batch_size = videos.shape[0]
        for i in range(batch_size):
            path_i = path[i]
            latents_i = latents[i].cpu()
            prompt_emb_i = prompt_emb['context'][i].cpu()
            first_frame_i = first_frame[i].cpu() if first_frame is not None else None
            save_path = get_pth_path(path_i)
            self.executor.submit(self.save_data, save_path, path_i, latents_i, prompt_emb_i, first_frame_i)



    def save_data(self, save_path, path, latent, prompt_emb, first_frame):
        data = {
            "path": path,
            "latent": latent,
            "prompt_emb": prompt_emb,
        }
        if first_frame is not None:
            data["first_frame"] = first_frame
        torch.save(data, save_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/mnt/hdd/dataset/MultiCamVideo-Dataset")
    parser.add_argument("--metadata_csv", type=str, default="metadata_split_ws1.csv")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    data_root = args.data_root
    metadata_csv = args.metadata_csv
    batch_size = args.batch_size
    num_workers = args.num_workers

    dataset = TextVideoDataset(
        data_root,
        os.path.join(data_root, metadata_csv),
        max_num_frames=81,
        frame_interval=1,
        num_frames=81,
        height=480,
        width=832,
        is_i2v=True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers
    )

    model = LightningModelForDataProcess(
        text_encoder_path="models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        image_encoder_path=None,
        vae_path="models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" ,
        tiled=False,
        tile_size=(34, 34),
        tile_stride=(18, 16),
    )


    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir="./ckpts",
        callbacks=[TQDMProgressBar(refresh_rate=1)],
    )
    trainer.test(model, dataloader)
