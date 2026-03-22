import itertools
import os
import sys

sys.path.append(os.path.join(sys.path[0], '../'))
import numpy as np
import argparse
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.config import (num_epochs, gamma, patience, device,
                          flusense_weights, classes_num_flusense, classes_num,
                          split_ratio, random_seed)
from utils.data_generator import JSTSP2021, split_compare_dataset, dev_fold_dataset
import utils.utilities as utt
from utils import model_functions as mf
from pytorch.models import BaselineCnn, Vggish, ResNet, MobileNet


def get_flusense_model(backbone):
    """Create a fresh model for FluSense 9-class classification."""
    n_classes = classes_num_flusense
    if 'baseline' in backbone:
        return BaselineCnn(n_classes)
    elif 'vgg' in backbone:
        return Vggish(n_classes)
    elif 'resnet' in backbone:
        return ResNet(n_classes)
    elif 'mobile' in backbone:
        return MobileNet(n_classes)
    else:
        raise Exception(f'Invalid backbone: {backbone}')


def plot_history(history, save_dir):
    """Plot and save loss and accuracy (UAR) curves."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    # --- Loss plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history['train_loss'], 'b-o', markersize=4, label='Train Loss')
    ax.plot(epochs, history['val_loss'], 'r-o', markersize=4, label='Val Loss')
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title('Training & Validation Loss', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=150)
    plt.close(fig)
    logging.info(f'Loss plot saved to {os.path.join(save_dir, "loss_curve.png")}')

    # --- UAR (Accuracy) plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history['train_uar'], 'b-o', markersize=4, label='Train UAR')
    ax.plot(epochs, history['val_uar'], 'r-o', markersize=4, label='Val UAR')
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('UAR (Unweighted Average Recall)', fontsize=13)
    ax.set_title('Training & Validation UAR', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'uar_curve.png'), dpi=150)
    plt.close(fig)
    logging.info(f'UAR plot saved to {os.path.join(save_dir, "uar_curve.png")}')

    # --- AUC plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history['train_auc'], 'b-o', markersize=4, label='Train AUC')
    ax.plot(epochs, history['val_auc'], 'r-o', markersize=4, label='Val AUC')
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('AUC', fontsize=13)
    ax.set_title('Training & Validation AUC', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'auc_curve.png'), dpi=150)
    plt.close(fig)
    logging.info(f'AUC plot saved to {os.path.join(save_dir, "auc_curve.png")}')


def train(args):
    # Arguments & parameters
    workspace = args.workspace
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    arch = args.arch
    flusense = args.flusense
    backbone = args.backbone
    num_workers = 0  # Use 0 for Windows compatibility
    save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)

    # Logging
    logs_dir = os.path.join(workspace, 'logs', backbone)
    os.makedirs(logs_dir, exist_ok=True)
    utt.create_logging(logs_dir, 'w')
    logging.info(args)

    # HDF5 path
    if flusense:
        hdf5_path = os.path.join(workspace, 'features_flusense.hdf5')
    else:
        hdf5_path = os.path.join(workspace, 'features_compare.hdf5')

    # Device
    use_device = device if torch.cuda.is_available() and 'cuda' in device else 'cpu'
    logging.info(f'Using device: {use_device}')

    # Dataset
    dataset = JSTSP2021(hdf5_path=hdf5_path, flusense=flusense)
    logging.info(f'Dataset size: {len(dataset)}')

    # Split dataset
    if flusense:
        # Use stratified random split for FluSense data
        train_dataset, val_dataset = dev_fold_dataset(
            dataset, split_ratio=split_ratio, shuffle=True, random_seed=random_seed
        )
        weights_tensor = torch.tensor(flusense_weights, dtype=torch.float32).to(use_device)
    else:
        val_dataset, train_dataset, _, weights = split_compare_dataset(dataset)
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(use_device)

    logging.info(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}')

    dataloaders = {
        'train': torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        'val': torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
    }

    # Model
    if flusense:
        model = get_flusense_model(backbone)
    else:
        model = mf.get_network(backbone)

    logging.info(f'Model: {backbone}, Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # DataParallel + device
    if torch.cuda.device_count() > 1:
        logging.info(f'GPU number: {torch.cuda.device_count()}')
        model = torch.nn.DataParallel(model)
    model.to(use_device)

    # Loss function
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate,
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=gamma, patience=patience,
        threshold=0.01, threshold_mode='abs'
    )

    # History tracking
    history = {
        'train_loss': [], 'val_loss': [],
        'train_uar': [], 'val_uar': [],
        'train_auc': [], 'val_auc': [],
    }

    best_val_uar = 0.0
    time_begin = time.time()

    for epoch in range(num_epochs):
        logging.info('=' * 50)
        logging.info(f'Epoch {epoch + 1}/{num_epochs}, lr: {optimizer.param_groups[0]["lr"]:.6f}')
        logging.info('=' * 50)

        epoch_metrics = {}

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            predicts = []
            truth = []
            y_scores = []
            n_samples = 0

            for i_batch, sample_batched in enumerate(dataloaders[phase]):
                inputs = sample_batched[arch].to(use_device)
                labels = sample_batched['label'].type(torch.LongTensor).to(use_device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    s = nn.Softmax(dim=1)
                    probs = s(outputs)
                    _, preds = torch.max(probs, dim=1)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                n_samples += inputs.size(0)
                y_scores.append(probs.detach().cpu().tolist())
                predicts.append(preds.detach().cpu().tolist())
                truth.append(labels.detach().cpu().tolist())

            epoch_loss = running_loss / n_samples
            y_scores = list(itertools.chain(*y_scores))
            predicts = list(itertools.chain(*predicts))
            truth = list(itertools.chain(*truth))

            confusion_mat, uar, auc = utt.scoring(truth, predicts, y_scores, flusense)

            epoch_metrics[phase] = {'loss': epoch_loss, 'uar': uar, 'auc': auc}

            logging.info(
                f'{phase:>5}: Loss={epoch_loss:.4f}, UAR={uar:.4f}, AUC={auc:.4f}, '
                f'Diag={np.diag(confusion_mat)}'
            )

        # Record history
        history['train_loss'].append(epoch_metrics['train']['loss'])
        history['val_loss'].append(epoch_metrics['val']['loss'])
        history['train_uar'].append(epoch_metrics['train']['uar'])
        history['val_uar'].append(epoch_metrics['val']['uar'])
        history['train_auc'].append(epoch_metrics['train']['auc'])
        history['val_auc'].append(epoch_metrics['val']['auc'])

        val_uar = epoch_metrics['val']['uar']
        scheduler.step(val_uar)

        # Save best model
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

        if val_uar > best_val_uar:
            best_val_uar = val_uar
            best_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model': model_state,
                'optimizer': optimizer.state_dict(),
                'val_uar': val_uar,
                'val_auc': epoch_metrics['val']['auc'],
                'backbone': backbone,
            }, best_path)
            logging.info(f'*** New best model saved (UAR={val_uar:.4f}) → {best_path}')

        # Save last model (each epoch)
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        last_path = os.path.join(save_dir, 'last_model.pth')
        torch.save({
            'epoch': epoch + 1,
            'model': model_state,
            'optimizer': optimizer.state_dict(),
            'val_uar': val_uar,
            'val_auc': epoch_metrics['val']['auc'],
            'backbone': backbone,
        }, last_path)

        # Plot history each epoch
        plot_dir = os.path.join(save_dir, 'plot')
        plot_history(history, plot_dir)

    time_elapsed = time.time() - time_begin
    logging.info('=' * 50)
    logging.info(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logging.info(f'Best Val UAR: {best_val_uar:.4f}')
    logging.info(f'Last model saved → {last_path}')
    logging.info(f'Best model saved → {os.path.join(save_dir, "best_model.pth")}')
    logging.info(f'Plots saved → {os.path.join(save_dir, "plot")}')

    logging.info('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CovNet FluSense Training')
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, default='../workspace')
    parser_train.add_argument('--flusense', action='store_true')
    parser_train.add_argument('--learning_rate', type=float, default=0.001)
    parser_train.add_argument('--batch_size', type=int, default=16)
    parser_train.add_argument('--arch', type=str, default='logmel')
    parser_train.add_argument('--backbone', type=str, default='baseline',
                              choices=['baseline', 'vgg', 'resnet', 'mobilenet'])
    parser_train.add_argument('--method', type=str, choices=['transfer', 'embedding'],
                              default='embedding')
    parser_train.add_argument('--layer', nargs='*')
    parser_train.add_argument('--save_dir', type=str, default='../save_model',
                              help='Directory to save best/last models and plots')

    # Parse arguments
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        raise Exception('Error argument!')