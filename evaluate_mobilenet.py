import os
import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.join(sys.path[0], 'pytorch'))
from pytorch.models import MobileNet
from utils.config import device, random_seed, split_ratio, valid_labels, classes_num_flusense
from utils.data_generator import JSTSP2021, dev_fold_dataset

def evaluate_mobilenet():
    print("[*] Đang tải dữ liệu từ workspace/features_flusense.hdf5 ...")
    hdf5_path = 'workspace/features_flusense.hdf5'
    
    if not os.path.exists(hdf5_path):
        print(f"Không tìm thấy file dữ liệu: {hdf5_path}")
        return

    # Khởi tạo dataset theo cách train gốc
    dataset = JSTSP2021(hdf5_path=hdf5_path, flusense=True)
    _, val_dataset = dev_fold_dataset(dataset, split_ratio, shuffle=True, random_seed=random_seed)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    # Khởi tạo Label Encoder để lấy danh sách class name chuẩn xác
    le = LabelEncoder()
    le.fit(valid_labels)
    target_names = le.classes_ # Classes đã được sort theo alphabet

    # Khởi tạo mô hình
    print("[*] Đang tải mô hình Custom MobileNet (Phase 1)...")
    model = MobileNet(classes_num_flusense)
    
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load weights
    checkpoint_path = "save_mobilenet/best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        model.load_state_dict(state_dict)
    else:
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(state_dict)
        
    model.to(device_obj)
    model.eval()

    all_preds = []
    all_labels = []

    print(f"[*] Tiến hành suy luận trên tập Validation ({len(val_dataset)} mẫu)...")
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['logmel'].to(device_obj)
            labels = batch['label'].to(device_obj)

            outputs = model(inputs)
            s = nn.Softmax(dim=1)
            outputs = s(outputs)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n" + "="*80)
    print("       BẢNG CLASSIFICATION REPORT (MOBILE NET - GIAI ĐOẠN 1)")
    print("="*80)
    
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=target_names, 
        digits=4,
        zero_division=0
    )
    print(report)

if __name__ == '__main__':
    evaluate_mobilenet()
