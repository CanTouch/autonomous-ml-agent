"""
K-Means Clustering helper based on skills/k-means-clustering.md.
Usage:
  ./.venv/bin/python src/clustering_kmeans.py data/processed/titanic_features.csv [--target Survived] [--k 3] [--k-range 1 10]
"""
import argparse
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    elif p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")


def evaluate_k_range(X_scaled: np.ndarray, k_min: int = 1, k_max: int = 10):
    k_values = list(range(k_min, k_max + 1))
    inertias = []
    silhouettes = {}

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        if k >= 2:
            sil = silhouette_score(X_scaled, km.labels_)
            silhouettes[k] = sil

    return k_values, inertias, silhouettes


def generate_elbow_plot(k_values: list[int], inertias: list[float], silhouettes: dict[int, float],
                        dataset_name: str, chosen_k: int, output_dir: str = "reports") -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    out_path = Path(output_dir) / f"kmeans_elbow_{dataset_name}_{today}.png"

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(k_values, inertias, marker="o", color="royalblue", linewidth=2, label="Inertia (Within-Cluster Sum of Squares)")
    ax1.axvline(chosen_k, color="crimson", linestyle="--", alpha=0.8, label=f"Selected K = {chosen_k}")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Total Inertia / Variation", color="royalblue")
    ax1.tick_params(axis="y", labelcolor="royalblue")
    ax1.set_title(f"K-Means Elbow Plot & Silhouette Scores: {dataset_name}")

    if silhouettes:
        ax2 = ax1.twinx()
        sil_ks = sorted(silhouettes.keys())
        sil_vals = [silhouettes[k] for k in sil_ks]
        ax2.plot(sil_ks, sil_vals, marker="s", color="darkorange", linestyle=":", linewidth=2, label="Silhouette Score")
        ax2.set_ylabel("Silhouette Score (higher is better)", color="darkorange")
        ax2.tick_params(axis="y", labelcolor="darkorange")
        ax2.set_ylim([-0.1, 1.0])

    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Elbow plot written to: {out_path}")
    return str(out_path)


def fit_kmeans(df: pd.DataFrame, target: str | None, k: int):
    if target and target in df.columns:
        y = df[target]
        X = df.drop(columns=[target])
    else:
        y = None
        X = df.copy()

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric features found for K-Means.")
    X_num = X[numeric_cols]

    if X_num.isnull().any().any():
        bad_cols = X_num.columns[X_num.isnull().any()].tolist()
        raise ValueError(f"Numeric features contain missing values: {bad_cols}.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    df_clustered = df.copy()
    df_clustered["cluster"] = cluster_labels

    # Unscaled centroids (means per cluster)
    centroids = df_clustered.groupby("cluster")[numeric_cols].mean()
    cluster_sizes = df_clustered["cluster"].value_counts().sort_index()

    sil = silhouette_score(X_scaled, cluster_labels) if k >= 2 else None

    return kmeans, df_clustered, centroids, cluster_sizes, sil, X_scaled, numeric_cols


def write_report(dataset_name: str, chosen_k: int, inertia: float, sil: float | None,
                 cluster_sizes: pd.Series, centroids: pd.DataFrame, elbow_path: str,
                 output_dir: str = "reports"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    out_path = Path(output_dir) / f"kmeans_{dataset_name}_{today}.md"

    lines = [
        f"# K-Means Clustering Report: {dataset_name}",
        f"Generated: {today}",
        "",
        "## Summary",
        f"- Number of clusters ($K$): {chosen_k}",
        f"- Total Within-Cluster Inertia: {inertia:,.2f}",
        f"- Silhouette Score: {sil:.4f}" if sil is not None else "- Silhouette Score: N/A ($K=1$)",
        "",
        "## Cluster Size Distribution",
        "| Cluster | Sample Count | Percentage |",
        "|---|---|---|",
    ]
    total_samples = cluster_sizes.sum()
    for cl_id, count in cluster_sizes.items():
        pct = 100 * count / total_samples
        lines.append(f"| Cluster {cl_id} | {count:,} | {pct:.1f}% |")

    lines.append("")
    lines.append("## Cluster Centroids (Feature Means per Cluster)")
    header = "| Feature | " + " | ".join(f"Cluster {i}" for i in centroids.index) + " |"
    sep = "|---|" + "|".join("---" for _ in centroids.index) + "|"
    lines.append(header)
    lines.append(sep)

    for col in centroids.columns:
        row_vals = " | ".join(f"{centroids.loc[i, col]:.3f}" for i in centroids.index)
        lines.append(f"| `{col}` | {row_vals} |")

    lines.append("")
    lines.append("## Plots")
    lines.append(f"- Elbow curve & silhouette: `{elbow_path}`")
    lines.append("")
    lines.append("Note: The elbow plot visualizes the reduction in within-cluster variation as K increases. Choose K where the marginal drop in inertia slows down.")

    out_path.write_text("\n".join(lines))
    print(f"K-Means report written to: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run K-Means clustering.")
    parser.add_argument("dataset_path", help="Path to input dataset (CSV/Excel)")
    parser.add_argument("--target", default=None, help="Target column name to exclude from clustering")
    parser.add_argument("--k", type=int, default=None, help="Specific number of clusters K to fit")
    parser.add_argument("--k-min", type=int, default=1, help="Min K for elbow scan")
    parser.add_argument("--k-max", type=int, default=8, help="Max K for elbow scan")
    parser.add_argument("--output-dir", default="reports", help="Output directory for reports/plots")
    parser.add_argument("--data-dir", default="data/processed", help="Output directory for clustered dataset")
    args = parser.parse_args()

    df = load_data(args.dataset_path)
    dataset_name = Path(args.dataset_path).stem.replace("_features", "").replace("_selected", "")

    # Prepare features for elbow scan
    X = df.drop(columns=[args.target]) if args.target and args.target in df.columns else df.copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[num_cols])

    # Scan K range
    k_vals, inertias, silhouettes = evaluate_k_range(X_scaled, args.k_min, args.k_max)

    if args.k is not None:
        chosen_k = args.k
    elif silhouettes:
        chosen_k = max(silhouettes, key=silhouettes.get)
    else:
        chosen_k = 3

    kmeans, df_clustered, centroids, cluster_sizes, sil, _, _ = fit_kmeans(df, args.target, chosen_k)

    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    csv_out = Path(args.data_dir) / f"{dataset_name}_kmeans.csv"
    df_clustered.to_csv(csv_out, index=False)
    print(f"Clustered dataset written to: {csv_out}")

    elbow_file = generate_elbow_plot(k_vals, inertias, silhouettes, dataset_name, chosen_k, args.output_dir)
    write_report(dataset_name, chosen_k, kmeans.inertia_, sil, cluster_sizes, centroids, elbow_file, args.output_dir)
