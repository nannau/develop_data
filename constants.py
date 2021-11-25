"""
These codes are unique identifiers for a given experiment run. 
These are contained within the mlflow_experiments directory.
Form:
shortname: {num: code}
"""
import torch
import torch.nn as nn


device = torch.device('cuda:0')

region_stats_u10 = {
    # region : (mean, std)
    "florida": (-1.7083029, 4.2478147),
    "central": (0.5051231, 2.8140204),
    "west": (0.9174596, 3.0119643),
}

region_stats_v10 = {
    "florida": (-0.4409821, 4.2057953),
    "central": (0.07797589, 3.2136111),
    "west": (-0.11845481, 3.9589875),
}

florida = {
    "CNN": ("1", 'e25c6b40324643c3afc1cf42981b11b5'),
    "13x13": ("0", 'e1d15a0615ca489aa6a17ec60247d0af'),
    "9x9": ("0", '3f48868c52404eb0a833897aa4642871'),
    "5x5": ("0", '1824682ae27c48669665cf042052d584'),
    "nf": ("0", 'feda42500d2b45549be96f1bf62b0b03'),
}

central = {
    "CNN": ("1", 'fbe44b0423204805bc6af4d7d6ac562e'),
    "13x13": ("0", 'bcf7e7cfa8ab4c4196ad6a2bb18e8601'),
    "9x9": ("0", '079a94c41ad3482996cc2b9f95adba8d'),
    "5x5": ("0", '202ea9f8a73b401fa22e62c24d9ab2d0'),
    "nf": ("0", '0c5ee480663f4f9eb7200f8879aa1244'),
}

west = {
    "CNN": ("1",'f76c0170818244629de4544805f93a59'),
    "13x13": ("0",'c4ec13e65fe74b399fc9e325a9966fef'),
    "9x9": ("0",'6abe7a9940c04b47819689070100e5e6'),
    "5x5": ("0",'70f5be887eff42e8a216780752644b2f'),
    "nf": ("0",'db9f0fae83c949eaad5d1176a43dae47'),
}

filters = {
    "CNN": lambda x: torch.zeros_like(x),
    "13x13": nn.AvgPool2d(13, stride=1, padding=0),
    "9x9": nn.AvgPool2d(9, stride=1, padding=0),
    "5x5": nn.AvgPool2d(5, stride=1, padding=0),
    "nf": lambda x: torch.zeros_like(x)          # Placeholder. Returns 0 object
}

padding = {
    "CNN": lambda x: torch.zeros_like(x),
    "13x13": nn.ReplicationPad2d(6),
    "9x9": nn.ReplicationPad2d(4),
    "5x5": nn.ReplicationPad2d(2),
    "nf": lambda x: torch.zeros_like(x)          # Placeholder. Returns 0 object
}