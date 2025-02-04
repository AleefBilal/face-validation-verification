from huggingface_hub import snapshot_download

snapshot_download(
    "fal/AuraFace-v1",
    local_dir="models/auraface",
)
snapshot_download(
    "Aleef/AI-Image-Classification",
    local_dir="models/ai_image_classifier",
)