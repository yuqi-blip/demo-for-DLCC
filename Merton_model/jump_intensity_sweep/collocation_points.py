import torch
from common.networks import CreatePoint


if __name__ == "__main__":
    print(CreatePoint(lb = 0, ub = 1, n_points = 100))
