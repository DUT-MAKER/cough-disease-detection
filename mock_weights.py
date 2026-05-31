import os
import torch
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pytorch.models import ResNet

os.makedirs('workspace/resnet_hybrid', exist_ok=True)
model = ResNet(5)
torch.save({'model': model.state_dict()}, 'workspace/resnet_hybrid/best_model.pth')
print("Saved mock ResNet weights to workspace/resnet_hybrid/best_model.pth")
