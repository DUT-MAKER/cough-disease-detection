#!/usr/bin/env python3
"""
Tạo CSV chỉ gồm mẫu CÓ TIẾNG HO, bài toán nhị phân:

  - Lớp 1 (ho COVID):   status == COVID-19  và cough_detected đủ cao
  - Lớp 0 (ho khác):    status == symptomatic (ho do triệu chứng khác, không gắn nhãn COVID-19)

  Cột nhãn: label_covid_cough (và label_binary trùng ý nghĩa để dùng chung code cũ).

  LOẠI BỎ hoàn toàn:
  - healthy (kể cả cough_detected cao — theo yêu cầu “bỏ qua healthy”)
  - status NaN / thiếu file âm thanh

Trong metadata CoughVID v3 chỉ có ba giá trị status: healthy, symptomatic, COVID-19.
Nhóm “symptomatic” được dùng làm proxy cho “tiếng ho không phải COVID” khi so sánh với
COVID-19; đây không phải “ho khỏe mạnh” mà là ho trong ngữ cảnh bệnh lý khác (cảm, cúm, …).

Cách dùng:
  uv run python prepare_coughvid_cough_covid_vs_other.py
  uv run python prepare_coughvid_cough_covid_vs_other.py --cough-threshold 0.5 --output ./out.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

VALID_AUDIO_EXT = {'.webm', '.ogg', '.wav'}


def build_file_map(data_dir: Path) -> dict[str, str]:
    file_map: dict[str, str] = {}
    for f in data_dir.iterdir():
        if not f.is_file():
            continue
        name, ext = f.stem, f.suffix.lower()
        if ext in VALID_AUDIO_EXT:
            file_map[name] = str(f.resolve())
    return file_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description='CSV: chỉ tiếng ho — COVID vs ho symptomatic (bỏ healthy).'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=None,
        help='Thư mục chứa metadata_compiled.csv và file âm thanh.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Đường dẫn CSV đầu ra.',
    )
    parser.add_argument(
        '--cough-threshold',
        type=float,
        default=0.5,
        help='Giữ mẫu có cough_detected >= giá trị này (mặc định 0.5, giống prepare_coughvid.py).',
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir = args.data_dir or (project_root / 'public_dataset_v3' / 'coughvid_20211012')
    output_path = args.output or (
        project_root / 'public_dataset_v3' / 'coughvid_covid_cough_vs_other_cough.csv'
    )

    meta_path = data_dir / 'metadata_compiled.csv'
    if not meta_path.is_file():
        raise FileNotFoundError(f'Không thấy {meta_path}')

    print(f'Đọc {meta_path} ...')
    df = pd.read_csv(meta_path, low_memory=False)

    cough = pd.to_numeric(df['cough_detected'], errors='coerce')
    df = df.assign(_cough=cough)
    df = df[df['_cough'] >= args.cough_threshold].copy()

    # Chỉ COVID ho vs symptomatic ho; bỏ healthy và NaN
    df = df[df['status'].isin(['COVID-19', 'symptomatic'])].copy()
    df['label_covid_cough'] = df['status'].map({'COVID-19': 1, 'symptomatic': 0})
    df['label_other_cough'] = 1 - df['label_covid_cough']
    # Tương thích pipeline cũ (vd. hybrid_coughvid): 1 = COVID ho, 0 = ho khác
    df['label_binary'] = df['label_covid_cough']

    df['uuid'] = df['uuid'].astype(str).str.strip()
    file_map = build_file_map(data_dir)
    df['file_path'] = df['uuid'].map(file_map)
    df = df.dropna(subset=['file_path']).copy()

    df = df.drop(columns=['_cough'], errors='ignore')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    n_cov = int((df['label_covid_cough'] == 1).sum())
    n_oth = int((df['label_covid_cough'] == 0).sum())
    print(f'\nĐã ghi: {output_path}')
    print(f'  Tổng mẫu (có file): {len(df)}')
    print(f'  Ho COVID-19 (label_covid_cough=1): {n_cov}')
    print(f'  Ho symptomatic / không COVID (label_covid_cough=0): {n_oth}')
    if n_cov and n_oth:
        print(f'  Tỉ lệ khối nhỏ / khối lớn ≈ 1 : {max(n_cov, n_oth) / min(n_cov, n_oth):.2f}')


if __name__ == '__main__':
    main()
