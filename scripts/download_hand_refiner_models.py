import os
import requests

def download_file(url, filename):
    """
    Helper function to download a file from a URL and show progress.
    """
    with requests.get(url, stream=True) as response:
        total_length = int(response.headers.get('content-length', 0))
        if total_length is None:  # no content length header
            print(f"Downloading {filename}")
            with open(filename, 'wb') as file:
                file.write(response.content)
        else:
            print(f"Downloading {filename} ({total_length} bytes)")
            with open(filename, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
                        print("\r{:.2%}".format(file.tell()/total_length), end='')

# Setup
REPO_DIR = os.getcwd()
models_dir = os.path.join(REPO_DIR, "models")
os.makedirs(models_dir, exist_ok=True)
BLOB = 'https://datarelease.blob.core.windows.net/metro'

# Download pre-trained models (Graphormer)
graphormer_dir = os.path.join(models_dir, "graphormer_release")
os.makedirs(graphormer_dir, exist_ok=True)

graphormer_url = f"{BLOB}/models/graphormer_hand_state_dict.bin"
graphormer_file = os.path.join(graphormer_dir, "graphormer_hand_state_dict.bin")
if not os.path.exists(graphormer_file):
    download_file(graphormer_url, graphormer_file)

# Download the ImageNet pre-trained HRNet models
hrnet_dir = os.path.join(models_dir, "hrnet")
os.makedirs(hrnet_dir, exist_ok=True)

hrnet_model_url = f"{BLOB}/models/hrnetv2_w64_imagenet_pretrained.pth"
hrnet_model_file = os.path.join(hrnet_dir, "hrnetv2_w64_imagenet_pretrained.pth")
if not os.path.exists(hrnet_model_file):
    download_file(hrnet_model_url, hrnet_model_file)

hrnet_config_url = f"{BLOB}/models/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"
hrnet_config_file = os.path.join(hrnet_dir, "cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml")
if not os.path.exists(hrnet_config_file):
    download_file(hrnet_config_url, hrnet_config_file)