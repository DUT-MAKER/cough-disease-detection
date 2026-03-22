import h5py, os
from collections import Counter

path = '../workspace/features_flusense.hdf5'
print(f'File exists: {os.path.exists(path)}')
if os.path.exists(path):
    print(f'File size: {os.path.getsize(path) / 1024 / 1024:.1f} MB')
    with h5py.File(path, 'r') as hf:
        print(f'Keys: {list(hf.keys())}')
        for k in hf.keys():
            print(f'  {k}: shape={hf[k].shape}, dtype={hf[k].dtype}')
        # Count non-empty entries
        count = 0
        for i in range(hf['audio_name'].shape[0]):
            if hf['audio_name'][i] != b'':
                count += 1
            else:
                break
        print(f'Filled entries: {count}/{hf["audio_name"].shape[0]}')
        if count > 0:
            labels = [hf['label'][i].decode() for i in range(count)]
            c = Counter(labels)
            for label, cnt in c.most_common():
                print(f'  {label}: {cnt}')
