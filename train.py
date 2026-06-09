#!/usr/bin/env python3
"""
MuGA-LIME: Neural Architecture Search using Micro Genetic Algorithm
===================================================================
Compares μGA, L-SHADE, and Random Search for MLP architecture optimization.
All algorithms stop at exactly NFE_MAX fitness evaluations.
"""

import os
import sys
import json
import random
import warnings
import time
import hashlib
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sklearn.datasets
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════
SEED = 42
N_RUNS = 30
MIN_N, MAX_N = 1, 64
NFE_MAX = 200
TRAIN_EPOCHS = 10
BATCH_SIZE = 64
N_FOLDS = 3
RESULTS_DIR = Path("experiment_results_nfe200")
CSV_DIR = RESULTS_DIR / "csv"
ALGORITHMS = ["muga", "lshade", "random"]


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def arch_hash(arch: tuple) -> str:
    return hashlib.md5(str(arch).encode()).hexdigest()[:8]


class ArchCache:
    def __init__(self):
        self._data = {}

    def get(self, arch):
        return self._data.get(arch_hash(arch))

    def put(self, arch, record):
        self._data[arch_hash(arch)] = record


# ═══════════════════════════════════════════════════════════════════════════
# Neural Network
# ═══════════════════════════════════════════════════════════════════════════
class MLPClassifier(nn.Module):
    """2-layer MLP with BatchNorm and Dropout."""

    def __init__(self, input_dim: int, hidden_dims: List[int], n_classes: int, dropout: float = 0.4):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def cv_fitness(hidden: List[int], input_dim: int, n_classes: int, X, y, device) -> float:
    """Evaluate architecture via 3-fold stratified cross-validation."""
    try:
        splits = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X, y))
    except ValueError:
        splits = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X))

    scores = []
    for tr, va in splits:
        model = MLPClassifier(input_dim, hidden, n_classes).to(device)
        Xtr = torch.tensor(X[tr], dtype=torch.float32).to(device)
        ytr = torch.tensor(y[tr], dtype=torch.long).to(device)
        Xva = torch.tensor(X[va], dtype=torch.float32).to(device)
        loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()

        model.train()
        for _ in range(TRAIN_EPOCHS):
            for bx, by in loader:
                opt.zero_grad()
                crit(model(bx), by).backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(Xva).argmax(1).cpu().numpy()
            scores.append(balanced_accuracy_score(y[va], pred))

    return float(np.mean(scores))


# ═══════════════════════════════════════════════════════════════════════════
# Data Loaders
# ═══════════════════════════════════════════════════════════════════════════
def load_iris():
    data = sklearn.datasets.load_iris()
    X = StandardScaler().fit_transform(data.data.astype(np.float32)).astype(np.float32)
    y = data.target
    return X, y, X.shape[1], len(np.unique(y))


def load_heart():
    p = Path("data/heart.csv")
    if not p.exists():
        p = Path("heart_v3.csv")
    cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    df = pd.read_csv(p, header=0, names=cols, na_values="?").dropna()
    X = StandardScaler().fit_transform(df.iloc[:, :-1].values.astype(np.float32)).astype(np.float32)
    y = (df.iloc[:, -1].values > 0).astype(np.int64)
    return X, y, X.shape[1], len(np.unique(y))


# ═══════════════════════════════════════════════════════════════════════════
# μGA — Micro Genetic Algorithm
# ═══════════════════════════════════════════════════════════════════════════
def run_muga(args) -> Tuple[List[int], float, Dict]:
    ds_name, run_id, X, y, input_dim, n_classes = args
    set_seed(SEED + run_id)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_bits = int(np.ceil(np.log2(MAX_N - MIN_N + 1)))
    chrom_len = 2 * n_bits

    def decode(chrom):
        mv = (1 << n_bits) - 1
        dims = []
        for i in range(2):
            bits = chrom[i * n_bits:(i + 1) * n_bits]
            val = int("".join(bits.astype(str)), 2)
            dims.append(max(MIN_N, int(MIN_N + val / mv * (MAX_N - MIN_N))))
        return dims

    pop_size = 4
    nfe = 0
    best_fit, best_chrom = -np.inf, None
    history = {"nfe": [], "best_fit": []}

    # Initialize population
    pop = np.random.randint(0, 2, (pop_size, chrom_len))
    fit = np.full(pop_size, -np.inf)
    for i in range(pop_size):
        h = decode(pop[i])
        fit[i] = cv_fitness(h, input_dim, n_classes, X, y, dev) if nfe < NFE_MAX else -np.inf
        nfe += 1
        if fit[i] > best_fit:
            best_fit, best_chrom = fit[i], pop[i].copy()
        history["nfe"].append(nfe)
        history["best_fit"].append(best_fit)

    # Main loop
    while nfe < NFE_MAX:
        # Tournament selection
        selected = []
        for _ in range(pop_size):
            a, b = np.random.choice(pop_size, 2, replace=False)
            selected.append(pop[a] if fit[a] >= fit[b] else pop[b])
        parents = np.array(selected)

        # 2-point crossover
        offspring = []
        for i in range(0, len(parents), 2):
            p1, p2 = parents[i], parents[(i + 1) % len(parents)]
            if np.random.rand() < 0.9 and chrom_len > 2:
                pts = sorted(np.random.choice(chrom_len - 1, 2, replace=False))
                c1 = np.concatenate([p1[:pts[0]], p2[pts[0]:pts[1]], p1[pts[1]:]])
                c2 = np.concatenate([p2[:pts[0]], p1[pts[0]:pts[1]], p2[pts[1]:]])
            else:
                c1, c2 = p1.copy(), p2.copy()
            offspring += [c1, c2]
        offspring = np.array(offspring[:pop_size])

        # Evaluate offspring
        off_fit = np.full(pop_size, -np.inf)
        for i in range(pop_size):
            if nfe >= NFE_MAX:
                break
            h = decode(offspring[i])
            off_fit[i] = cv_fitness(h, input_dim, n_classes, X, y, dev)
            nfe += 1
            if off_fit[i] > best_fit:
                best_fit, best_chrom = off_fit[i], offspring[i].copy()
            history["nfe"].append(nfe)
            history["best_fit"].append(best_fit)

        # (μ+λ) selection
        pool = np.vstack([pop, offspring])
        pool_fit = np.concatenate([fit, off_fit])
        order = np.argsort(pool_fit)[::-1][:pop_size]
        pop, fit = pool[order], pool_fit[order]

        # Diversity restart
        if pop_size > 1:
            div = float(np.mean([
                np.sum(pop[i] != pop[j]) / chrom_len
                for i in range(pop_size) for j in range(i + 1, pop_size)
            ]))
        else:
            div = 0

        if div < 0.05 and nfe < NFE_MAX:
            elite = pop[np.argmax(fit)].copy()
            new_pop = np.random.randint(0, 2, (pop_size - 1, chrom_len))
            pop = np.vstack([elite, new_pop])
            for i in range(1, pop_size):
                if nfe >= NFE_MAX:
                    break
                h = decode(pop[i])
                fit[i] = cv_fitness(h, input_dim, n_classes, X, y, dev)
                nfe += 1
                if fit[i] > best_fit:
                    best_fit, best_chrom = fit[i], pop[i].copy()
                history["nfe"].append(nfe)
                history["best_fit"].append(best_fit)

    return decode(best_chrom), best_fit, history


# ═══════════════════════════════════════════════════════════════════════════
# L-SHADE — Linear Population Size Reduction Differential Evolution
# ═══════════════════════════════════════════════════════════════════════════
def run_lshade(args) -> Tuple[List[int], float, Dict]:
    ds_name, run_id, X, y, input_dim, n_classes = args
    set_seed(SEED + run_id)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    D = 2
    NP_init, NP_min = 20, 4
    H = 5
    p_best_rate = 0.1
    NP = NP_init
    pop = np.random.rand(NP, D)
    fitness = np.full(NP, -np.inf)
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    nfe = 0
    best_fitness, best_arch_val = -np.inf, None
    g_best_f, g_best_a = -np.inf, None
    rng = np.random.default_rng(SEED + run_id)
    history = {"nfe": [], "best_fit": []}

    # Initialize
    for i in range(NP):
        if nfe >= NFE_MAX:
            break
        h = [int(round(MIN_N + pop[i][j] * (MAX_N - MIN_N))) for j in range(D)]
        h = [max(MIN_N, min(MAX_N, x)) for x in h]
        fitness[i] = cv_fitness(h, input_dim, n_classes, X, y, dev)
        nfe += 1
        if fitness[i] > best_fitness:
            best_fitness = fitness[i]
            best_arch_val = h
    if best_fitness > g_best_f:
        g_best_f = best_fitness
        g_best_a = best_arch_val
    history["nfe"].append(nfe)
    history["best_fit"].append(g_best_f)

    # Main loop
    while nfe < NFE_MAX:
        ratio = nfe / NFE_MAX
        NP_new = max(NP_min, int(round(NP_init + (NP_min - NP_init) * ratio)))
        if NP_new < NP:
            keep = np.argsort(fitness)[::-1][:NP_new]
            pop, fitness, NP = pop[keep], fitness[keep], NP_new

        p_num = max(2, int(p_best_rate * NP))
        p_best_idx = np.argsort(fitness)[::-1][:p_num]
        S_F, S_CR = [], []

        for i in range(NP):
            if nfe >= NFE_MAX:
                break
            r = rng.integers(0, H)
            F = np.clip(M_F[r] + 0.1 * np.tan(np.pi * (rng.random() - 0.5)), 0.0, 1.0)
            CR = np.clip(rng.normal(M_CR[r], 0.1), 0.0, 1.0)
            xpbest = pop[rng.choice(p_best_idx)]
            cand = [j for j in range(NP) if j != i]
            r1, r2 = rng.choice(cand, 2, replace=False)
            mutant = pop[i] + F * (xpbest - pop[i]) + F * (pop[r1] - pop[r2])
            mutant = np.clip(np.nan_to_num(mutant, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
            trial = pop[i].copy()
            jrand = rng.integers(D)
            for j in range(D):
                if rng.random() < CR or j == jrand:
                    trial[j] = mutant[j]
            h = [int(round(MIN_N + trial[j] * (MAX_N - MIN_N))) for j in range(D)]
            h = [max(MIN_N, min(MAX_N, x)) for x in h]
            fit_t = cv_fitness(h, input_dim, n_classes, X, y, dev)
            nfe += 1
            if fit_t >= fitness[i]:
                if fit_t > fitness[i]:
                    S_F.append(F)
                    S_CR.append(CR)
                pop[i], fitness[i] = trial, fit_t
                if fit_t > best_fitness:
                    best_fitness = fit_t
                    best_arch_val = h
            if nfe < NFE_MAX:
                history["nfe"].append(nfe)
                history["best_fit"].append(g_best_f)

        if best_fitness > g_best_f:
            g_best_f = best_fitness
            g_best_a = best_arch_val

        if S_F:
            w = np.array(S_F) / np.sum(S_F)
            M_F[k] = np.sum(w * np.array(S_F) ** 2) / np.sum(w * np.array(S_F))
            M_CR[k] = np.mean(S_CR)
            k = (k + 1) % H

        # Diversity restart
        div = float(np.mean(np.std(pop, axis=0))) if NP >= 2 else 0
        if div < 0.05 and nfe < NFE_MAX:
            bi = np.argmax(fitness)
            best = pop[bi].copy()
            bf = fitness[bi]
            pop = np.vstack([best, rng.random((NP - 1, D))])
            fitness = np.full(NP, -np.inf)
            fitness[0] = bf

    return g_best_a, g_best_f, history


# ═══════════════════════════════════════════════════════════════════════════
# Random Search (Baseline)
# ═══════════════════════════════════════════════════════════════════════════
def run_random(args) -> Tuple[List[int], float, Dict]:
    ds_name, run_id, X, y, input_dim, n_classes = args
    set_seed(SEED + run_id)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = {"nfe": [], "best_fit": []}
    best_arch, best_fit = None, -np.inf
    nfe = 0

    for _ in range(NFE_MAX):
        h = [np.random.randint(MIN_N, MAX_N + 1) for _ in range(2)]
        fit = cv_fitness(h, input_dim, n_classes, X, y, dev)
        nfe += 1
        if fit > best_fit:
            best_fit, best_arch = fit, h
        history["nfe"].append(nfe)
        history["best_fit"].append(best_fit)

    return best_arch, best_fit, history


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════
RUNNERS = {"muga": run_muga, "lshade": run_lshade, "random": run_random}


def run_trial(args):
    ds_name, alg_name, run_id = args
    X, y, input_dim, n_classes = DATA_CACHE[ds_name]
    try:
        best_arch, best_fit, history = RUNNERS[alg_name]((ds_name, run_id, X, y, input_dim, n_classes))
        return {
            "dataset": ds_name,
            "algorithm": alg_name,
            "run": run_id,
            "best_arch": best_arch,
            "best_fitness": best_fit,
            "history": history,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} (NFE_MAX={NFE_MAX}, N_RUNS={N_RUNS})", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    CSV_DIR.mkdir(exist_ok=True)

    # Load datasets
    DATA_CACHE = {}
    for name, fn in [("iris", load_iris), ("heart", load_heart)]:
        X, y, d, c = fn()
        DATA_CACHE[name] = (X, y, d, c)
        print(f"  {name}: {X.shape[0]} samples, {d} features, {c} classes", flush=True)

    all_results = {}
    total = len(ALGORITHMS) * N_RUNS * 2
    count = 0
    t0_all = time.time()

    for ds_name in ["iris", "heart"]:
        args_list = [(ds_name, alg, run) for alg in ALGORITHMS for run in range(N_RUNS)]
        ds_results = []

        with multiprocessing.Pool(processes=4) as pool:
            for result in pool.imap_unordered(run_trial, args_list):
                count += 1
                if result:
                    ds_results.append(result)
                    print(
                        f"  [{count}/{total}] {result['dataset']}/{result['algorithm']} "
                        f"run {result['run'] + 1} | best={result['best_fitness']:.4f} "
                        f"nfe={result['history']['nfe'][-1]}",
                        flush=True,
                    )

        all_results[ds_name] = ds_results

        # Save JSON
        json_path = RESULTS_DIR / f"{ds_name}_results.json"
        with open(json_path, "w") as f:
            json.dump(ds_results, f, indent=2)

        # Save convergence CSVs
        for alg in ALGORITHMS:
            alg_r = [r for r in ds_results if r["algorithm"] == alg]
            rows = []
            for r in alg_r:
                run = r["run"]
                hist = r["history"]
                for n, ft in zip(hist["nfe"], hist["best_fit"]):
                    rows.append({"run": run, "nfe": n, "best_fitness": ft})
            csv_path = CSV_DIR / f"{ds_name}_{alg}_convergence.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)

        # Save cost CSV
        cost_rows = [
            {
                "run": r["run"],
                "algorithm": r["algorithm"],
                "best_fitness": r["best_fitness"],
                "best_arch": str(r["best_arch"]),
                "total_nfe": r["history"]["nfe"][-1],
            }
            for r in ds_results
        ]
        pd.DataFrame(cost_rows).to_csv(CSV_DIR / f"{ds_name}_cost.csv", index=False)

        print(f"  {ds_name} done.", flush=True)

    elapsed_all = time.time() - t0_all
    print(f"\nTotal time: {elapsed_all:.0f}s", flush=True)

    # Print summary
    for ds_name in ["iris", "heart"]:
        print(f"\n  {ds_name.upper()} (NFE={NFE_MAX}):")
        for alg in ALGORITHMS:
            r = [x for x in all_results[ds_name] if x["algorithm"] == alg]
            fits = [x["best_fitness"] for x in r]
            nfes = [x["history"]["nfe"][-1] for x in r]
            print(f"    {alg}: fitness={np.mean(fits):.4f}±{np.std(fits):.4f}  nfe={np.mean(nfes):.0f}")

    with open(RESULTS_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR}/")
