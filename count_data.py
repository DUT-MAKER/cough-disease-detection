import h5py
import numpy as np

features_path = "workspace/features_flusense.hdf5"
try:
    with h5py.File(features_path, 'r') as hf:
        y_all = hf['target'][:]
        target_names = [name.decode('utf-8') for name in hf['target_name'][:]]
        
        unique, counts = np.unique(y_all, return_counts=True)
        print(f"\n[+] Tổng số phân đoạn âm thanh: {len(y_all)}")
        print("-" * 50)
        print(f"{'LỚP ÂM THANH (CLASS)':<20} | {'SỐ LƯỢNG MẪU':<15} | {'TỶ LỆ (%)'}")
        print("-" * 50)
        
        for u, c in zip(unique, counts):
            name = target_names[u]
            pct = 100.0 * c / len(y_all)
            print(f"{name:<20} | {c:<15} | {pct:.2f}%")
        print("-" * 50)
except Exception as e:
    print("Lỗi đọc file HDF5:", e)
