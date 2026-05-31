import os
import pandas as pd
from pathlib import Path

def main():
    data_dir = Path("/home/ssd_120gb/test/detect-cough-disease-with-CovNet/public_dataset_v3/coughvid_20211012")
    csv_path = data_dir / "metadata_compiled.csv"
    output_path = Path("/home/ssd_120gb/test/detect-cough-disease-with-CovNet/public_dataset_v3/coughvid_clean.csv")

    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter only healthy and COVID-19
    df_filtered = df[df['status'].isin(['healthy', 'COVID-19'])].copy()
    df_filtered['label_binary'] = df_filtered['status'].map({'healthy': 0, 'COVID-19': 1})
    
    print(f"Found {len(df_filtered)} records with 'healthy' or 'COVID-19' status.")
    
    print("Reading directory contents for audio files...")
    # List all files in the directory
    all_files = os.listdir(data_dir)
    
    # Create a dictionary mapping uuid -> full file path
    file_map = {}
    valid_exts = {'.webm', '.ogg', '.wav'}
    for f in all_files:
        name, ext = os.path.splitext(f)
        if ext.lower() in valid_exts:
            file_map[name] = str(data_dir / f)
            
    print(f"Found {len(file_map)} valid audio files in the dataset folder.")
    
    # Use pandas map to quickly assign file paths
    # Ensure uuid is string and clean
    df_filtered['uuid'] = df_filtered['uuid'].astype(str).str.strip()
    df_filtered['file_path'] = df_filtered['uuid'].map(file_map)
    
    # Drop records that do not have a corresponding audio file
    df_valid = df_filtered.dropna(subset=['file_path']).copy()
    
    print(f"Records with existing files in metadata: {len(df_valid)}")
    
    # Filter out obvious non-cough noise (cough_detected >= 0.5)
    df_clean = df_valid[pd.to_numeric(df_valid['cough_detected'], errors='coerce') >= 0.5].copy()
    
    df_clean.to_csv(output_path, index=False)
    print(f"\nFinal cleaned dataset shape: {df_clean.shape}")
    print(f"Saved to {output_path}")
    print("\nClass distribution in final dataset:")
    print(df_clean['status'].value_counts())
    
    print("\nFile extensions in final set:")
    exts = df_clean['file_path'].apply(lambda x: Path(x).suffix)
    print(exts.value_counts())

if __name__ == "__main__":
    main()
