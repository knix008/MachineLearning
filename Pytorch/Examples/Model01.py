import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('car_evaluation.csv')
print("Dataset Head!!!")
print(dataset.head())
print("Dataset Tail!!!")
print(dataset.tail())
