#!/usr/bin/env python3
"""
Đánh giá MobileNet (checkpoint huấn luyện FluSense 9 lớp) dưới dạng bài toán 2 lớp.

Cách làm: giữ nguyên backbone + đầu ra 9 lớp, gom xác suất các chỉ số lớp FluSense
được liệt kê trong --positive-flu-classes thành P(lớp dương = 1); phần còn lại là P(0).

- eval-mode flusense_cough: nhãn thật = 1 nếu mẫu là lớp "cough" (index 0 trong
  valid_labels của utils.config), 0 nếu là các âm thanh khác. Phù hợp checkpoint
  huấn luyện trên features_flusense.

- eval-mode dicova / compare: nhãn 0/1 lấy trực tiếp từ HDF5; bạn phải tự chọn
  --positive-flu-classes cho đúng nghiệp vụ (mặc định 0 = cough chỉ là ví dụ kỹ thuật).
"""
from __future__ import annotations

import argparse
import itertools
import logging
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from utils.config import random_seed, split_ratio
from utils.data_generator import JSTSP2021, dev_fold_dataset, dicova_fold_dataset, split_compare_dataset
from utils.utilities import scoring, create_logging

import test as covnet_test


def _parse_pos_indices(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(',') if p.strip()]
    return [int(p) for p in parts]


def _binary_forward(model, inputs, pos_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Trả về (truth_binary_01, bin_probs) với bin_probs shape (B,2) [P(0), P(1)]."""
    logits = model(inputs)
    probs = torch.softmax(logits, dim=1)
    p_pos = probs[:, pos_idx].sum(dim=1).clamp(0.0, 1.0)
    p_neg = 1.0 - p_pos
    bin_probs = torch.stack([p_neg, p_pos], dim=1)
    pred = (p_pos >= 0.5).long()
    return pred, bin_probs


def _resolve_runtime_device(device_arg: str) -> torch.device:
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device_arg == 'cuda' and not torch.cuda.is_available():
        logging.warning('CUDA duoc yeu cau nhung khong kha dung. Fallback sang CPU.')
        return torch.device('cpu')
    return torch.device(device_arg)


def _forward_with_fallback(
    model,
    x: torch.Tensor,
    pos_idx: torch.Tensor,
    runtime_device: torch.device,
    allow_cpu_fallback_on_oom: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.device]:
    try:
        pred, bin_probs = _binary_forward(model, x.to(runtime_device), pos_idx.to(runtime_device))
        return pred, bin_probs, runtime_device
    except torch.OutOfMemoryError:
        if runtime_device.type != 'cuda' or not allow_cpu_fallback_on_oom:
            raise
        logging.warning('CUDA OOM trong luc test. Tu dong fallback sang CPU va tiep tuc...')
        torch.cuda.empty_cache()
        runtime_device = torch.device('cpu')
        model.to(runtime_device)
        pred, bin_probs = _binary_forward(model, x.to(runtime_device), pos_idx.to(runtime_device))
        return pred, bin_probs, runtime_device


def eval_flusense_cough(
    model,
    workspace,
    arch,
    batch_size,
    pos_indices,
    num_workers,
    runtime_device: torch.device,
    allow_cpu_fallback_on_oom: bool,
):
    hdf5_path = os.path.join(workspace, 'features_flusense.hdf5')
    if not os.path.isfile(hdf5_path):
        raise FileNotFoundError(f'Thieu HDF5 FluSense: {hdf5_path}')

    dataset = JSTSP2021(hdf5_path=hdf5_path, flusense=True)
    _, val_dataset = dev_fold_dataset(dataset, split_ratio, shuffle=True, random_seed=random_seed)
    loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    pos_idx = torch.tensor(pos_indices, dtype=torch.long)
    y_scores, predicts, truth = [], [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch[arch]
            y_multi = batch['label']
            y_bin = (y_multi == 0).long()

            pred, bin_probs, runtime_device = _forward_with_fallback(
                model, x, pos_idx, runtime_device, allow_cpu_fallback_on_oom
            )

            y_scores.append(bin_probs.cpu().tolist())
            predicts.append(pred.cpu().tolist())
            truth.append(y_bin.tolist())

    y_scores = list(itertools.chain(*y_scores))
    predicts = list(itertools.chain(*predicts))
    truth = list(itertools.chain(*truth))

    cm, uar, auc = scoring(truth, predicts, y_scores, num_classes=2)
    return auc, uar, cm


def eval_dicova_folds(
    model,
    workspace,
    arch,
    batch_size,
    pos_indices,
    num_workers,
    runtime_device: torch.device,
    allow_cpu_fallback_on_oom: bool,
):
    hdf5_path = os.path.join(workspace, 'features_dicova.hdf5')
    if not os.path.isfile(hdf5_path):
        raise FileNotFoundError(f'Thieu HDF5 Dicova: {hdf5_path}')

    dataset = JSTSP2021(hdf5_path=hdf5_path, flusense=False)
    pos_idx = torch.tensor(pos_indices, dtype=torch.long)
    aucs, uars = [], []

    for fold in range(1, 6):
        logging.info('-' * 30 + f'\nFold {fold}')
        _, val_dataset, _ = dicova_fold_dataset(dataset, workspace, fold)
        loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        y_scores, predicts, truth = [], [], []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                x = batch[arch]
                y_bin = batch['label'].type(torch.LongTensor)

                pred, bin_probs, runtime_device = _forward_with_fallback(
                    model, x, pos_idx, runtime_device, allow_cpu_fallback_on_oom
                )

                y_scores.append(bin_probs.cpu().tolist())
                predicts.append(pred.cpu().tolist())
                truth.append(y_bin.tolist())

        y_scores = list(itertools.chain(*y_scores))
        predicts = list(itertools.chain(*predicts))
        truth = list(itertools.chain(*truth))

        cm, uar, auc = scoring(truth, predicts, y_scores, num_classes=2)
        logging.info(
            'fold %d: auc=%.4f uar=%.4f diag=%s', fold, auc, uar, np.diag(cm)
        )
        aucs.append(auc)
        uars.append(uar)

    return float(np.mean(aucs)), float(np.mean(uars))


def eval_compare_test(
    model,
    workspace,
    arch,
    batch_size,
    pos_indices,
    num_workers,
    runtime_device: torch.device,
    allow_cpu_fallback_on_oom: bool,
):
    hdf5_path = os.path.join(workspace, 'features_compare.hdf5')
    if not os.path.isfile(hdf5_path):
        raise FileNotFoundError(f'Thieu HDF5 ComParE: {hdf5_path}')

    dataset = JSTSP2021(hdf5_path=hdf5_path, flusense=False)
    _, _, test_dataset = split_compare_dataset(dataset)
    loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    pos_idx = torch.tensor(pos_indices, dtype=torch.long)
    y_scores, predicts, truth = [], [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch[arch]
            y_bin = batch['label'].type(torch.LongTensor)

            pred, bin_probs, runtime_device = _forward_with_fallback(
                model, x, pos_idx, runtime_device, allow_cpu_fallback_on_oom
            )

            y_scores.append(bin_probs.cpu().tolist())
            predicts.append(pred.cpu().tolist())
            truth.append(y_bin.tolist())

    y_scores = list(itertools.chain(*y_scores))
    predicts = list(itertools.chain(*predicts))
    truth = list(itertools.chain(*truth))

    cm, uar, auc = scoring(truth, predicts, y_scores, num_classes=2)
    return auc, uar, cm


def main():
    default_ckpt = os.path.join(REPO_ROOT, 'save_mobilenet', 'best_model.pth')
    default_ws = os.path.join(REPO_ROOT, 'workspace')

    parser = argparse.ArgumentParser(description='Test MobileNet 9-class -> metric nhi phan (2 lop)')
    parser.add_argument('--checkpoint', type=str, default=default_ckpt)
    parser.add_argument('--workspace', type=str, default=default_ws)
    parser.add_argument('--arch', type=str, default='logmel')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--no_cpu_fallback_on_oom', action='store_true')
    parser.add_argument(
        '--eval-mode',
        type=str,
        choices=['flusense_cough', 'dicova', 'compare'],
        default='flusense_cough',
        help='flusense_cough: nhan 1=ho (cough); dicova/compare: nhan tu HDF5',
    )
    parser.add_argument(
        '--positive-flu-classes',
        type=str,
        default='0',
        help='Chi so lop FluSense (0..8) gop vao lop duong, cach nhau dau phay. Mac dinh 0=cough.',
    )

    args = parser.parse_args()
    pos_indices = _parse_pos_indices(args.positive_flu_classes)

    logs_dir = os.path.join(
        args.workspace,
        'logs',
        'mobilenet',
        'binary_test',
        f"mode={args.eval_mode}_pos={args.positive_flu_classes.replace(',', '-')}",
    )
    os.makedirs(logs_dir, exist_ok=True)
    create_logging(logs_dir, 'w')
    logging.info(args)

    runtime_device = _resolve_runtime_device(args.device)
    allow_cpu_fallback_on_oom = not args.no_cpu_fallback_on_oom

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f'Missing checkpoint: {args.checkpoint}')

    state = covnet_test._load_checkpoint_state(
        args.checkpoint, map_location=runtime_device
    )
    n_flu = covnet_test._infer_num_classes_from_state_dict(state)
    for i in pos_indices:
        if i < 0 or i >= n_flu:
            raise ValueError(f'positive-flu-classes contains {i} but checkpoint has {n_flu} classes')

    model = covnet_test.get_network('mobilenet', num_classes=n_flu)
    model.load_state_dict(state)
    if runtime_device.type == 'cuda' and torch.cuda.device_count() > 1:
        logging.info('GPU number: %d', torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
    model.to(runtime_device)
    model.eval()

    if args.eval_mode == 'flusense_cough':
        auc, uar, cm = eval_flusense_cough(
            model,
            args.workspace,
            args.arch,
            args.batch_size,
            pos_indices,
            args.num_workers,
            runtime_device,
            allow_cpu_fallback_on_oom,
        )
        logging.info('FluSense cough vs rest (val split): auc=%.4f uar=%.4f diag=%s', auc, uar, np.diag(cm))
    elif args.eval_mode == 'dicova':
        mean_auc, mean_uar = eval_dicova_folds(
            model,
            args.workspace,
            args.arch,
            args.batch_size,
            pos_indices,
            args.num_workers,
            runtime_device,
            allow_cpu_fallback_on_oom,
        )
        logging.info('Dicova (5-fold val mean): auc=%.4f uar=%.4f', mean_auc, mean_uar)
    else:
        auc, uar, cm = eval_compare_test(
            model,
            args.workspace,
            args.arch,
            args.batch_size,
            pos_indices,
            args.num_workers,
            runtime_device,
            allow_cpu_fallback_on_oom,
        )
        logging.info('ComParE test: auc=%.4f uar=%.4f diag=%s', auc, uar, np.diag(cm))


if __name__ == '__main__':
    main()
