import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import os
import sys

# Add root so we can load utils/config
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from utils.config import classes_num_flusense, valid_labels

# Ma trận đường chéo (Diagonal) từ best log val:
best_diag = [13, 335, 898, 15, 123, 40, 18, 299, 1]

# Từ log, tôi sẽ lấy số liệu mảng ma trận nhầm lẫn tổng quan. 
# Nhưng vì log chỉ in đường chéo (số mẫu nhận diện đúng), 
# nên tôi sẽ cần load lại model và chạy file test để có accuracy/f1 thực tế
# Tuy nhiên, thay vì tốn thời gian, let's write a small script to grab the 0001.log 
# and see if there are missing info
