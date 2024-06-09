# Distance measures for spatial analysis of latent encodings in DMs
# Written by Ye Zhu

import time
import numpy as np
import torch
import math


def euclidean_distance(x, y):
    return torch.sqrt(torch.sum((x - y) ** 2))


def cosine_similarity(x, y):
    dot_product = torch.dot(x, y)
    norm_x = torch.norm(x)
    norm_y = torch.norm(y)
    return dot_product / (norm_x * norm_y)


def jaccard_similarity(x, y):
    intersection = len(set(x).intersection(y))
    union = len(set(x).union(y))
    return intersection / union


def mahalanobis_distance(x, y, cov):
    diff = x - y
    inv_cov = np.linalg.inv(cov)
    return np.sqrt(np.dot(np.dot(diff, inv_cov), diff))
