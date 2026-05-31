import h5py
with h5py.File("workspace/features_flusense.hdf5", 'r') as hf:
    print(hf.keys())
