"""
Brain Graph Super-Resolution using GCN with Bilinear Edge Decoder (BrainGCN-SR)

This script implements a 3-layer GCN with multi-scale features, learned upscaling,
and a bilinear edge decoder to predict high-resolution (268x268) brain graphs from
low-resolution (160x160) brain graphs. Permutation equivariant throughout.

Usage:
    python model.py
"""

import os
import random
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from MatrixVectorizer import MatrixVectorizer

# ─── Reproducibility (from https://github.com/basiralab/DGL/blob/main/Project/reproducibility.py) ───
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")


# ─── GCN Layer ───────────────────────────────────────────────────────────────
class GCNLayer(nn.Module):
    """Single GCN layer: X' = sigma(D^{-1/2} A D^{-1/2} X W)"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj_norm):
        support = x @ self.weight
        out = adj_norm @ support
        return out + self.bias


# ─── Model ───────────────────────────────────────────────────────────────────
# HR vector size: upper triangular of 268x268 = 268*267/2 = 35778
HR_VEC_SIZE = 35778

class BrainGCN_SR(nn.Module):
    """
    3-layer GCN encoder with multi-scale features + learned upscaling + bilinear edge decoder.

    Pipeline (permutation equivariant throughout):
        1. Normalize LR adjacency with self-loops
        2. Use LR adjacency rows as initial node features
        3. 3-layer GCN encoder, concatenate all layer outputs (multi-scale)
        4. Project multi-scale features to edge embedding space
        5. Learned upscaling matrix U maps 160 LR -> 268 HR node embeddings
        6. Bilinear edge decoder: A[i,j] = sigma(z_i^T W z_j)
    """

    def __init__(self, lr_dim=160, hr_dim=268, hidden_dims=None,
                 edge_dim=128, dropout=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]
        self.lr_dim = lr_dim
        self.hr_dim = hr_dim
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)

        # GCN encoder layers (dynamic)
        self.gcn_layers = nn.ModuleList()
        in_dim = lr_dim
        for out_dim in hidden_dims:
            self.gcn_layers.append(GCNLayer(in_dim, out_dim))
            in_dim = out_dim
        self.dropout = nn.Dropout(dropout)

        # Multi-scale feature dimension (concat ALL layer outputs)
        multi_scale_dim = sum(hidden_dims)

        # 2-layer MLP projection: adds per-node nonlinearity (equivariant)
        self.proj = nn.Sequential(
            nn.Linear(multi_scale_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim),
        )

        # Learned upscaling matrix: maps LR (160) node space -> HR (268) node space
        self.U = nn.Parameter(torch.empty(hr_dim, lr_dim))
        nn.init.xavier_uniform_(self.U)

        # Bilinear edge decoder weight matrix
        self.W_edge = nn.Parameter(torch.empty(edge_dim, edge_dim))
        nn.init.xavier_uniform_(self.W_edge)

        # Precompute vectorization indices (column-major upper triangle, matching MatrixVectorizer)
        rows, cols = [], []
        for col in range(hr_dim):
            for row in range(col):
                rows.append(row)
                cols.append(col)
        self.register_buffer("vec_rows", torch.LongTensor(rows))
        self.register_buffer("vec_cols", torch.LongTensor(cols))

    def normalize_adj(self, adj):
        """Symmetric normalization: D^{-1/2} (A + I) D^{-1/2}"""
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        degree = adj.sum(dim=1)
        d_inv_sqrt = degree.pow(-0.5)
        d_inv_sqrt[d_inv_sqrt == float("inf")] = 0.0
        return (adj * d_inv_sqrt.unsqueeze(1)) * d_inv_sqrt.unsqueeze(0)

    def forward(self, adj_lr):
        """
        Args:
            adj_lr: (160, 160) LR adjacency matrix
        Returns:
            hr_vec: (35778,) predicted HR edge vector
        """
        adj_norm = self.normalize_adj(adj_lr)
        x = adj_lr  # (160, 160) — node features = connectivity profile

        # N-layer GCN with multi-scale feature collection + residual connections
        layer_outputs = []
        h = x
        for i, gcn in enumerate(self.gcn_layers):
            h_new = gcn(h, adj_norm)
            # ReLU + dropout on all layers except the last
            if i < self.num_layers - 1:
                h_new = self.dropout(F.relu(h_new))
            # Residual: add most recent previous layer with matching dimension
            for j in range(len(layer_outputs) - 1, -1, -1):
                if layer_outputs[j].shape[1] == h_new.shape[1]:
                    h_new = h_new + layer_outputs[j]
                    break
            layer_outputs.append(h_new)
            h = h_new

        # Concatenate multi-scale features (all layer outputs)
        z_lr = torch.cat(layer_outputs, dim=1)  # (160, sum(hidden_dims))

        # 2-layer MLP projection to edge embedding space
        z_lr = self.proj(z_lr)                      # (160, edge_dim)

        # Upscale to HR node space: U (268, 160) @ z_lr (160, edge_dim) -> (268, edge_dim)
        z_hr = self.U @ z_lr                        # (268, edge_dim)

        # Bilinear edge decoder: A = sigmoid( (Z W Z^T + Z W^T Z^T) / 2 )
        a_raw = z_hr @ self.W_edge @ z_hr.t()      # (268, 268)
        a_sym = (a_raw + a_raw.t()) / 2             # ensure symmetry
        a_pred = torch.sigmoid(a_sym)               # (268, 268), values in [0, 1]

        # Vectorize upper triangle (column-major order, matching MatrixVectorizer)
        hr_vec = a_pred[self.vec_rows, self.vec_cols]  # (35778,)

        return hr_vec


# ─── Data loading ────────────────────────────────────────────────────────────
def load_data(lr_path, hr_path):
    """Load LR as matrices (for GCN input) and HR as vectors (for loss target)."""
    if lr_path.endswith(".csv"):
        lr_vecs = pd.read_csv(lr_path).values   # (N, 12720)
        hr_vecs = pd.read_csv(hr_path).values   # (N, 35778)
    else:
        lr_vecs = np.load(lr_path)
        hr_vecs = np.load(hr_path)

    # Anti-vectorize LR into matrices for GCN input
    lr_matrices = np.array([MatrixVectorizer.anti_vectorize(lr_vecs[i], 160)
                            for i in range(lr_vecs.shape[0])])

    return lr_matrices, hr_vecs


def load_test_data(test_lr_path):
    """Load test LR as matrices."""
    if test_lr_path.endswith(".csv"):
        lr_vecs = pd.read_csv(test_lr_path).values
    else:
        lr_vecs = np.load(test_lr_path)

    lr_matrices = np.array([MatrixVectorizer.anti_vectorize(lr_vecs[i], 160)
                            for i in range(lr_vecs.shape[0])])
    return lr_matrices


# ─── Training ────────────────────────────────────────────────────────────────
def train_model(model, train_lr_matrices, train_hr_vecs, val_lr_matrices=None,
                val_hr_vecs=None, epochs=200, lr=0.001, device="cpu", patience=50,
                noise_std=0.005):
    """
    Train the model with early stopping. LR input is matrices (for GCN), HR target is vectors.
    Adds Gaussian noise to LR inputs during training for data augmentation.
    """
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Early stopping state
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        perm = np.random.permutation(len(train_lr_matrices))
        for idx in perm:
            adj_lr = torch.FloatTensor(train_lr_matrices[idx]).to(device)
            # Data augmentation: add small Gaussian noise to LR input
            if noise_std > 0:
                noise = torch.randn_like(adj_lr) * noise_std
                adj_lr = (adj_lr + noise).clamp(min=0.0)
                # Re-symmetrize after noise
                adj_lr = (adj_lr + adj_lr.t()) / 2
            hr_vec = torch.FloatTensor(train_hr_vecs[idx]).to(device)

            optimizer.zero_grad()
            pred_vec = model(adj_lr)
            loss = F.l1_loss(pred_vec, hr_vec)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        avg_train_loss = epoch_loss / len(train_lr_matrices)

        # Validation
        avg_val_loss = None
        if val_lr_matrices is not None and val_hr_vecs is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for idx in range(len(val_lr_matrices)):
                    adj_lr = torch.FloatTensor(val_lr_matrices[idx]).to(device)
                    hr_vec = torch.FloatTensor(val_hr_vecs[idx]).to(device)
                    pred_vec = model(adj_lr)
                    val_loss += F.l1_loss(pred_vec, hr_vec).item()
            avg_val_loss = val_loss / len(val_lr_matrices)

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1} (best val MAE: {best_val_loss:.6f})")
                break

        if (epoch + 1) % 20 == 0:
            msg = f"Epoch {epoch+1}/{epochs} | Train MAE: {avg_train_loss:.6f}"
            if avg_val_loss is not None:
                msg += f" | Val MAE: {avg_val_loss:.6f}"
            print(msg)

    # Restore best model weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best model (val MAE: {best_val_loss:.6f})")


# ─── Prediction ──────────────────────────────────────────────────────────────
def predict(model, lr_matrices, device="cpu"):
    """Generate HR vector predictions for a set of LR matrices."""
    model.eval()
    pred_vecs = []
    with torch.no_grad():
        for i in range(len(lr_matrices)):
            adj_lr = torch.FloatTensor(lr_matrices[i]).to(device)
            pred_vec = model(adj_lr)
            # Post-process: clamp negatives to 0 (data is in [0,1])
            pred_vec = pred_vec.clamp(min=0.0)
            pred_vecs.append(pred_vec.cpu().numpy())
    return np.array(pred_vecs)  # (N, 35778)


# ─── Evaluation (following official evaluation_measures.py) ──────────────────
def compute_evaluation_metrics(pred_vecs, gt_vecs):
    """
    Compute all 8 evaluation metrics following the official evaluation code.

    Args:
        pred_vecs: (N, 35778) predicted HR vectors
        gt_vecs: (N, 35778) ground truth HR vectors
    """
    num_samples = len(pred_vecs)
    metrics = {}

    mae_bc = []
    mae_ec = []
    mae_pc = []
    mae_cc = []  # Extra: Closeness Centrality
    mae_dc = []  # Extra: Degree Centrality

    for i in range(num_samples):
        print(f"    Sample {i+1}/{num_samples}...", end="\r")

        # Anti-vectorize to matrices for centrality computation
        pred_mat = MatrixVectorizer.anti_vectorize(pred_vecs[i], 268)
        gt_mat = MatrixVectorizer.anti_vectorize(gt_vecs[i], 268)

        # Build weighted graphs
        pred_graph = nx.from_numpy_array(pred_mat, edge_attr="weight")
        gt_graph = nx.from_numpy_array(gt_mat, edge_attr="weight")

        # Centrality measures (approximate betweenness for speed)
        pred_bc = nx.betweenness_centrality(pred_graph, weight="weight", k=50)
        pred_ec = nx.eigenvector_centrality(pred_graph, weight="weight", max_iter=1000)
        pred_pc = nx.pagerank(pred_graph, weight="weight")

        gt_bc = nx.betweenness_centrality(gt_graph, weight="weight", k=50)
        gt_ec = nx.eigenvector_centrality(gt_graph, weight="weight", max_iter=1000)
        gt_pc = nx.pagerank(gt_graph, weight="weight")

        mae_bc.append(mean_absolute_error(list(pred_bc.values()), list(gt_bc.values())))
        mae_ec.append(mean_absolute_error(list(pred_ec.values()), list(gt_ec.values())))
        mae_pc.append(mean_absolute_error(list(pred_pc.values()), list(gt_pc.values())))

        # Extra metrics (closeness: no distance= since weights are similarities)
        pred_cc = nx.closeness_centrality(pred_graph)
        gt_cc = nx.closeness_centrality(gt_graph)
        mae_cc.append(mean_absolute_error(list(pred_cc.values()), list(gt_cc.values())))

        pred_dc = nx.degree_centrality(pred_graph)
        gt_dc = nx.degree_centrality(gt_graph)
        mae_dc.append(mean_absolute_error(list(pred_dc.values()), list(gt_dc.values())))

    # Concatenate all vectors into single 1D arrays for global metrics
    pred_1d = np.concatenate([pred_vecs[i] for i in range(num_samples)])
    gt_1d = np.concatenate([gt_vecs[i] for i in range(num_samples)])

    metrics["MAE"] = mean_absolute_error(pred_1d, gt_1d)
    metrics["PCC"] = pearsonr(pred_1d, gt_1d)[0]
    metrics["JSD"] = jensenshannon(pred_1d, gt_1d)
    metrics["MAE(PC)"] = np.mean(mae_pc)
    metrics["MAE(EC)"] = np.mean(mae_ec)
    metrics["MAE(BC)"] = np.mean(mae_bc)
    metrics["MAE(CC)"] = np.mean(mae_cc)
    metrics["MAE(DC)"] = np.mean(mae_dc)

    return metrics


# ─── Plotting ────────────────────────────────────────────────────────────────
def plot_fold_results(all_fold_metrics, output_dir="outputs"):
    """Generate bar plots for each fold and averaged across folds."""
    os.makedirs(output_dir, exist_ok=True)

    metric_names = list(all_fold_metrics[0].keys())
    n_folds = len(all_fold_metrics)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = ["#00CED1", "#FF6347", "#FFD700", "#32CD32", "#8A2BE2", "#FF69B4", "#FF8C00", "#1E90FF"]

    for fold_idx in range(n_folds):
        ax = axes[fold_idx]
        values = [all_fold_metrics[fold_idx][m] for m in metric_names]
        ax.bar(range(len(metric_names)), values, color=colors[:len(metric_names)])
        ax.set_title(f"Fold {fold_idx + 1}", fontsize=13)
        ax.set_xticks(range(len(metric_names)))
        ax.set_xticklabels(metric_names, rotation=45, ha="right", fontsize=9)
        ax.set_ylim(bottom=0)

    # Average across folds with std error bars
    ax = axes[n_folds]
    avg_values = []
    std_values = []
    for m in metric_names:
        vals = [all_fold_metrics[f][m] for f in range(n_folds)]
        avg_values.append(np.mean(vals))
        std_values.append(np.std(vals))
    ax.bar(range(len(metric_names)), avg_values, yerr=std_values,
           color=colors[:len(metric_names)], capsize=4)
    ax.set_title("Avg. Across Folds", fontsize=13)
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation_barplots.png"), dpi=150)
    plt.close()
    print(f"Saved evaluation bar plots to {output_dir}/evaluation_barplots.png")


# ─── Kaggle CSV generation ───────────────────────────────────────────────────
def save_predictions_csv(pred_vectors, filepath):
    """
    Save predictions in Kaggle submission format.
    pred_vectors: (N, 35778) array -> flatten to 1D, ID from 1..N*35778
    """
    flat = pred_vectors.flatten()
    ids = np.arange(1, len(flat) + 1)
    df = pd.DataFrame({"ID": ids, "Predicted": flat})
    df.to_csv(filepath, index=False)
    print(f"Saved predictions to {filepath} ({len(flat)} entries)")


# ─── Main pipeline ───────────────────────────────────────────────────────────
def run_3fold_cv(lr_path, hr_path, test_lr_path=None,
                 epochs=200, lr=0.001, hidden_dims=None,
                 edge_dim=128, output_dir="outputs", dropout=0.1):
    """Run the full 3-fold cross-validation pipeline."""

    start_time = time.time()

    # Load data: LR as matrices, HR as vectors
    print("Loading data...")
    lr_matrices, hr_vecs = load_data(lr_path, hr_path)
    print(f"  LR matrices: {lr_matrices.shape}, HR vectors: {hr_vecs.shape}")

    os.makedirs(output_dir, exist_ok=True)

    # 3-fold CV
    kf = KFold(n_splits=3, shuffle=True, random_state=random_seed)

    all_fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(lr_matrices)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/3")
        print(f"{'='*60}")
        print(f"  Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

        train_lr = lr_matrices[train_idx]
        train_hr = hr_vecs[train_idx]
        val_lr = lr_matrices[val_idx]
        val_hr = hr_vecs[val_idx]

        # Create model
        model = BrainGCN_SR(
            lr_dim=160, hr_dim=268,
            hidden_dims=hidden_dims,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {total_params:,}")

        # Train
        train_model(model, train_lr, train_hr, val_lr, val_hr,
                     epochs=epochs, lr=lr, device=device.type)

        # Predict on validation set (returns vectors directly)
        pred_vecs = predict(model, val_lr, device=device.type)

        # Save fold predictions CSV
        csv_path = os.path.join(output_dir, f"predictions_fold_{fold_idx + 1}.csv")
        save_predictions_csv(pred_vecs, csv_path)

        # Evaluate
        print(f"\n  Evaluating fold {fold_idx + 1}...")
        metrics = compute_evaluation_metrics(pred_vecs, val_hr)
        all_fold_metrics.append(metrics)

        print(f"  Fold {fold_idx + 1} results:")
        for name, val in metrics.items():
            print(f"    {name}: {val:.6f}")

    # Plot results
    plot_fold_results(all_fold_metrics, output_dir=output_dir)

    # Print summary
    print(f"\n{'='*60}")
    print("AVERAGE ACROSS FOLDS")
    print(f"{'='*60}")
    metric_names = list(all_fold_metrics[0].keys())
    for m in metric_names:
        vals = [all_fold_metrics[f][m] for f in range(3)]
        print(f"  {m}: {np.mean(vals):.6f} +/- {np.std(vals):.6f}")

    total_time = time.time() - start_time
    print(f"\nTotal 3F-CV training time: {total_time:.1f}s")
    print(f"RAM usage: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB" if torch.cuda.is_available()
          else "RAM usage: N/A (CPU mode)")

    # ─── Kaggle test predictions (ensemble of N models) ─────────────────
    if test_lr_path is not None:
        n_ensemble = 5
        print(f"\n{'='*60}")
        print(f"GENERATING KAGGLE TEST PREDICTIONS (ensemble of {n_ensemble})")
        print(f"{'='*60}")

        test_lr_matrices = load_test_data(test_lr_path)
        print(f"  Test LR matrices: {test_lr_matrices.shape}")

        ensemble_preds = []
        for ens_idx in range(n_ensemble):
            seed = random_seed + ens_idx
            torch.manual_seed(seed)
            np.random.seed(seed)
            print(f"\n  Ensemble model {ens_idx+1}/{n_ensemble} (seed={seed})...")

            model = BrainGCN_SR(
                lr_dim=160, hr_dim=268,
                hidden_dims=hidden_dims,
                edge_dim=edge_dim,
                dropout=dropout,
            )
            train_model(model, lr_matrices, hr_vecs,
                         epochs=epochs, lr=lr, device=device.type)

            preds = predict(model, test_lr_matrices, device=device.type)
            ensemble_preds.append(preds)

        # Average predictions across ensemble
        test_pred_vecs = np.mean(ensemble_preds, axis=0)
        print(f"\n  Ensemble averaging complete ({n_ensemble} models)")

        kaggle_path = os.path.join(output_dir, "kaggle_submission.csv")
        save_predictions_csv(test_pred_vecs, kaggle_path)

    return all_fold_metrics


# ─── Hyperparameter search ──────────────────────────────────────────────────
def run_hyperparam_search(lr_path, hr_path, epochs=500, lr_rate=0.001,
                          edge_dim=128, output_dir="outputs"):
    """Search over GCN layer configurations via full 3-fold CV."""

    configs = [
        {"hidden_dims": [256, 128],          "label": "2L-[256,128]"},
        {"hidden_dims": [256, 256],          "label": "2L-[256,256]"},
        {"hidden_dims": [256, 256, 128],     "label": "3L-[256,256,128] (baseline)"},
        {"hidden_dims": [256, 256, 256],     "label": "3L-[256,256,256]"},
        {"hidden_dims": [192, 192, 128],     "label": "3L-[192,192,128]"},
        {"hidden_dims": [256, 256, 256, 128], "label": "4L-[256,256,256,128]"},
        {"hidden_dims": [256, 256, 128, 128], "label": "4L-[256,256,128,128]"},
    ]

    results = []

    for i, cfg in enumerate(configs):
        print(f"\n{'#'*70}")
        print(f"CONFIG {i+1}/{len(configs)}: {cfg['label']}")
        print(f"  hidden_dims = {cfg['hidden_dims']}")
        print(f"{'#'*70}")

        config_output_dir = os.path.join(output_dir, f"search_{i+1}")

        fold_metrics = run_3fold_cv(
            lr_path=lr_path,
            hr_path=hr_path,
            test_lr_path=None,
            epochs=epochs,
            lr=lr_rate,
            hidden_dims=cfg["hidden_dims"],
            edge_dim=edge_dim,
            output_dir=config_output_dir,
        )

        # Average metrics across folds
        avg_metrics = {}
        metric_names = list(fold_metrics[0].keys())
        for m in metric_names:
            vals = [fold_metrics[f][m] for f in range(len(fold_metrics))]
            avg_metrics[m] = np.mean(vals)
            avg_metrics[f"{m}_std"] = np.std(vals)

        # Count parameters
        model = BrainGCN_SR(hidden_dims=cfg["hidden_dims"], edge_dim=edge_dim)
        param_count = sum(p.numel() for p in model.parameters())

        results.append({
            "label": cfg["label"],
            "hidden_dims": cfg["hidden_dims"],
            "params": param_count,
            **avg_metrics,
        })

    # Print comparison table sorted by MAE
    print(f"\n{'='*90}")
    print("HYPERPARAMETER SEARCH RESULTS")
    print(f"{'='*90}")

    results.sort(key=lambda r: r["MAE"])

    print(f"{'Config':<35} {'Params':>8} {'MAE':>12} {'PCC':>12} {'JSD':>12}")
    print("-" * 90)

    for r in results:
        print(f"{r['label']:<35} {r['params']:>8,} "
              f"{r['MAE']:.4f}+-{r['MAE_std']:.4f} "
              f"{r['PCC']:.4f}+-{r['PCC_std']:.4f} "
              f"{r['JSD']:.4f}+-{r['JSD_std']:.4f}")

    print(f"\nBest config: {results[0]['label']} (MAE={results[0]['MAE']:.6f})")

    return results


# ─── CLI entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BrainGCN-SR: Brain Graph Super-Resolution")
    data_dir = "dgl-2026-brain-graph-super-resolution-challenge"
    parser.add_argument("--lr_path", type=str, default=os.path.join(data_dir, "lr_train.csv"))
    parser.add_argument("--hr_path", type=str, default=os.path.join(data_dir, "hr_train.csv"))
    parser.add_argument("--test_lr_path", type=str, default=os.path.join(data_dir, "lr_test.csv"))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256, 256],
                        help="Hidden dimensions for GCN layers, e.g. 256 256 128")
    parser.add_argument("--edge_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--search", action="store_true",
                        help="Run hyperparameter search over layer configurations")
    args = parser.parse_args()

    if args.search:
        run_hyperparam_search(
            lr_path=args.lr_path,
            hr_path=args.hr_path,
            epochs=args.epochs,
            lr_rate=args.lr,
            edge_dim=args.edge_dim,
            output_dir=args.output_dir,
        )
    else:
        run_3fold_cv(
            lr_path=args.lr_path,
            hr_path=args.hr_path,
            test_lr_path=args.test_lr_path,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dims=args.hidden_dims,
            edge_dim=args.edge_dim,
            output_dir=args.output_dir,
            dropout=args.dropout,
        )
