import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report

import utils.config as config
from pytorch.models import MobileNet
import h5py

def evaluate():
    from utils.data_generator import RespiratoryDataset
    from torch.utils.data import DataLoader
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileNet(class_num=9)
    
    print("[*] Nạp mô hình Custom MobileNet từ save_mobilenet/best_model.pth...")
    checkpoint = torch.load("save_mobilenet/best_model.pth", map_location=device, weights_only=False)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    print("[*] Đang tải tập DICOVA Test Dataset dùng cho Test Binary Phase 1...")
    
    # Let's get the standard dataset setup from test_mobilenet_binary.py
    # Since I don't know the exact class or path, let's copy what test_mobilenet_binary.py does
