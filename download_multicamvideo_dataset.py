from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="KlingTeam/MultiCamVideo-Dataset",
    repo_type="dataset",
    local_dir="/mnt/hdd/dataset/MultiCamVideo-Dataset",
)
