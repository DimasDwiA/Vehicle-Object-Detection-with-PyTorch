import os
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm
from torch.optim import Adam, SGD
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_features, 256),
                        nn.Dropout(0.5),
                        nn.ReLU(),
                        nn.Linear(256,7))
model

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)