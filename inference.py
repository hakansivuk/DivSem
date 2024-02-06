from model.utils import get_config, tensor2im
from model.inference_handler import InferenceHandler
from model.dataset import Image_Editing_Dataset

import torch
import cv2

from torch.utils.data import DataLoader

def get_cfg():
    cfg = get_config("checkpoints/config.yaml")

    cfg['lab_dim'] = 151
    cfg['max_epoch'] = 500
    cfg['test_freq'] = 20

    cfg["is_train"] = False
    cfg["dataset_name"] = "flickr-landscape"
    return cfg

def get_inference_handler(cfg):
    inference_handler = InferenceHandler(cfg)
    inference_handler.eval()
    inference_handler.load_checkpoint(ckpt_filename="checkpoints/best.pth")
    return inference_handler

def get_dataloader(cfg):
    dataset_root = "gradio_files/samples"
    dataset = Image_Editing_Dataset(cfg, dataset_root, split='test', dataset_name="flickr-landscape")
    return DataLoader(dataset=dataset, batch_size=1, shuffle=False)

def start_inference():
    cfg = get_cfg()
    inference_handler = get_inference_handler(cfg)
    dataloader = get_dataloader(cfg)
    cached_codes = torch.load("checkpoints/style_codes.pt", map_location=torch.device("cpu"))
    save_path = 'gradio_files/samples/synthesized_image/result.png'
    with torch.no_grad():
        cfg['mask_type'] = '0'
        for i, data in enumerate(dataloader):
            inference_handler.set_input(data)
            inference_handler.forward(cached_codes)
            result = inference_handler.get_results()
            cv2.imwrite(save_path, tensor2im(result))
    return save_path

if __name__ == "__main__":
    start_inference()
