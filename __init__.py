from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
from .config import PICLConfig
from .data import PICLDataset
from .graph import HybridCausalGraph
from .inference import causal_disablement_and_sufficiency
from .scm import LinearGaussianSCM
_GAS_PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]

def _dga_ratios(log_ppm):
    cols = [(log_ppm[:, i] - log_ppm[:, j]).unsqueeze(1) for i, j in _GAS_PAIRS]
    return torch.cat(cols, dim=1)

@dataclass
class ClassifierHead:
    models: list
    n_classes: int

def extract_scm_features(ds: PICLDataset, graph: HybridCausalGraph, scm: LinearGaussianSCM) -> np.ndarray:
    n_faults = graph.n_faults
    n_vars = graph.n_vars
    y = ds.gas_values
    N = y.shape[0]
    with torch.no_grad():
        A = graph.final_hard_adjacency()
        W_eff = (A * graph.weight_matrix()).detach()
        mu = scm.mu.detach()
        mu_y = mu[n_faults:]
        I_K = torch.eye(n_faults, dtype=W_eff.dtype)
        mus_by_class = scm.intervene_on_faults(W_eff, I_K, n_faults)
        sigma2 = scm.noise_variance(0).detach()
        I = torch.eye(n_vars, dtype=W_eff.dtype)
        T = torch.linalg.solve(I - W_eff, I)
        Sigma = T.T @ torch.diag(sigma2) @ T + 1e-06 * torch.eye(n_vars)
        S_fy = Sigma[:n_faults, n_faults:]
        S_yy = Sigma[n_faults:, n_faults:]
        rhs = (y - mu_y.unsqueeze(0)).T
        Efy = (S_fy @ torch.linalg.solve(S_yy, rhs)).T + mu[:n_faults].unsqueeze(0)
        res = y.unsqueeze(1) - mus_by_class.unsqueeze(0)
        res_norm = torch.norm(res, dim=2)
        post_proxy = Efy.clamp(min=0.0, max=1.0)
        post_proxy = post_proxy / post_proxy.sum(dim=1, keepdim=True).clamp(min=1e-08)
        Ed, Es = causal_disablement_and_sufficiency(y, post_proxy, W_eff, scm, n_faults)
        dga = _dga_ratios(ds.log_ppm) if ds.log_ppm is not None else torch.zeros(N, len(_GAS_PAIRS), dtype=y.dtype)
    feats = torch.cat([y, Efy, res.reshape(N, -1), res_norm, Ed, Es, dga], dim=1)
    return feats.cpu().numpy().astype(np.float32)

def _build(cfg: PICLConfig) -> List[object]:
    hc = cfg.raw.get('classifier_head', {})
    name = hc.get('model', 'gradient_boosting')
    n_est = int(hc.get('n_estimators', 300))
    max_depth = int(hc.get('max_depth', 4))
    lr = float(hc.get('learning_rate', 0.05))
    seed = int(cfg.raw['experiment']['seed'])
    ensemble = bool(hc.get('ensemble', False))

    def one(name):
        if name == 'gradient_boosting':
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(max_iter=n_est, max_depth=max_depth if max_depth > 0 else None, learning_rate=lr, random_state=seed)
        if name == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=n_est, max_depth=max_depth if max_depth > 0 else None, random_state=seed, n_jobs=-1)
        if name == 'logistic':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=5000, C=1.0, random_state=seed)
        raise ValueError(f'unknown classifier: {name}')
    if ensemble:
        return [one('gradient_boosting'), one('random_forest'), one('logistic')]
    return [one(name)]

def train_classifier_head(cfg: PICLConfig, ds_train: PICLDataset, graph: HybridCausalGraph, scm: LinearGaussianSCM) -> ClassifierHead:
    hc = cfg.raw.get('classifier_head', {})
    real_only = bool(hc.get('train_on_real_only', False))
    if real_only and ds_train.is_synthetic is not None:
        from .data import subset
        ds_use = subset(ds_train, ~ds_train.is_synthetic)
    else:
        ds_use = ds_train
    X = extract_scm_features(ds_use, graph, scm)
    y = ds_use.labels.cpu().numpy()
    models = _build(cfg)
    for clf in models:
        clf.fit(X, y)
    return ClassifierHead(models=models, n_classes=graph.n_faults)

def classifier_posterior(head: ClassifierHead, ds: PICLDataset, graph: HybridCausalGraph, scm: LinearGaussianSCM) -> torch.Tensor:
    X = extract_scm_features(ds, graph, scm)
    N = X.shape[0]
    out = np.zeros((N, head.n_classes), dtype=np.float32)
    for clf in head.models:
        proba = clf.predict_proba(X)
        for col, c in enumerate(clf.classes_):
            out[:, int(c)] += proba[:, col]
    out /= len(head.models)
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum <= 0, 1.0, row_sum)
    return torch.from_numpy(out / row_sum)
