import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
import sys
import os

from pytorch.models import MobileNet
import utils.config as config

def evaluate():
    print("[*] Đang tải dữ liệu từ workspace/features_flusense.hdf5 ...")
    hf = h5py.File("workspace/features_flusense.hdf5", 'r')
    x_all = hf['logmel'][:]
    
    # Label in HDF5 are byte strings like b'cough'
    y_all_bytes = hf['label'][:]
    hf.close()
    
    target_names = config.valid_labels
    
    # Decode bytes and map to int indices
    y_all = []
    for b_label in y_all_bytes:
        str_label = b_label.decode('utf-8')
        if str_label in target_names:
            y_all.append(target_names.index(str_label))
        else:
            y_all.append(0) # fallback
    y_all = np.array(y_all)
    
    print(f"    -> Shape của dữ liệu: {x_all.shape}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileNet(class_num=len(target_names))
    
    print("[*] Đang nạp trọng số mô hình Custom MobileNet...")
    model_path = "save_mobilenet/best_model.pth"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    
    batch_size = 128
    all_preds = []
    
    print("[*] Đang chạy suy luận (Inference) trên toàn bộ dataset...")
    with torch.no_grad():
        for i in range(0, len(x_all), batch_size):
            batch_x_np = x_all[i:i+batch_size]
            batch_x = torch.tensor(batch_x_np, dtype=torch.float32).to(device)
            outputs = model(batch_x)
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            
    print("[*] Hoàn tất suy luận!")
    
    all_preds = np.array(all_preds)
    
    print("\n" + "="*60)
    print(" BẢNG CLASSIFICATION REPORT (ĐÁNH GIÁ 9 CLASS TÍCH HỢP TẬP AUGMENTED)")
    print("="*60)
    report = classification_report(y_all, all_preds, target_names=target_names, digits=4)
    print(report)

if __name__ == '__main__':
    evaluate()
