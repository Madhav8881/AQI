"""Loading the processed dataset and slicing forecast windows (shared by training and the app)."""
import json

import numpy as np
import pandas as pd
import torch

import config as C


class AQIData:
    def __init__(self):
        d = np.load(C.DATASET_NPZ)
        self.features = torch.from_numpy(d["features"])            # [T, N, F]
        self.pm25 = d["pm25"]                                        # [T, N] µg/m³, NaN=missing
        self.observed = d["observed"]                                # [T, N] bool
        self.times = pd.to_datetime(d["times"].astype("datetime64[h]"))
        self.splits = {k: d[f"{k}_idx"] for k in ("train", "val", "test")}
        with open(C.SCALER_JSON) as f:
            sc = json.load(f)
        self.mean = np.array(sc["mean"], dtype=np.float32)
        self.std = np.array(sc["std"], dtype=np.float32)
        self.stations = pd.read_csv(C.STATIONS_CSV)
        self.adj = torch.from_numpy(np.load(C.ADJ_NPY))
        self.dist = np.load(C.DIST_NPY)
        self.horizons = np.array(C.HORIZONS)
        self.window = C.INPUT_WINDOW
        pm_norm = (np.nan_to_num(self.pm25) - self.mean) / self.std
        self.target_norm = torch.from_numpy(pm_norm.astype(np.float32))
        self.target_mask = torch.from_numpy(self.observed)
        self._win = torch.arange(-self.window + 1, 1)
        self._hor = torch.from_numpy(self.horizons)

    @property
    def n_nodes(self):
        return self.features.shape[1]

    def batch(self, origins):
        """origins: 1-D array of time indices (last input hour). Returns X, Y_norm, mask."""
        t = torch.as_tensor(origins, dtype=torch.long)
        X = self.features[t[:, None] + self._win]                   # [B, W, N, F]
        tgt = t[:, None] + self._hor                                 # [B, H]
        return X, self.target_norm[tgt], self.target_mask[tgt]       # [B, H, N]

    @staticmethod
    def hide_stations(X, stations):
        """Simulate failed sensors: the given stations report nothing in the input window.
        Matches how real long outages look after preprocessing (value = station mean, observed = 0)."""
        X = X.clone()
        X[:, :, stations, 0] = 0.0
        X[:, :, stations, 1] = 0.0
        return X

    def targets_raw(self, origins):
        tgt = np.asarray(origins)[:, None] + self.horizons
        return self.pm25[tgt], self.observed[tgt]                    # [B, H, N]

    def denorm(self, y_norm):
        return np.clip(y_norm * self.std + self.mean, 0, None)       # [.., N] broadcast
