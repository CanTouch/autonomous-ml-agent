"""
Hierarchical Agglomerative Clustering helper based on skills/hierarchical-clustering.md.
Usage:
  ./.venv/bin/python src/clustering_hierarchical.py data/processed/titanic_features.csv [--target Survived] [--n-clusters 3] [--metric euclidean] [--linkage ward]
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
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    elif p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")


def fit_hierarchical(df: pd.DataFrame, target: str | None, n_clusters: int = 3,
                     metric: str = "euclidean", method: str = "ward"):
    if target and target in df.columns:
        y = df[target]
        X = df.drop(columns=[target])
    else:
        y = None
        X = df.copy()

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric features found for Hierarchical Clustering.")
    X_num = X[numeric_cols]

    if X_num.isnull().any().any():
        bad_cols = X_num.columns[X_num.isnull().any()].tolist()
        raise ValueError(f"Numeric features contain missing values: {bad_cols}.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    # Ward linkage strictly requires euclidean distance
    if method == "ward" and metric != "euclidean":
        metric = "euclidean"

    # Compute linkage matrix for dendrogram
    # If dataset is very large, downsample for dendrogram readability
    max_dendrogram_samples = 250
    if len(X_scaled) > max_dendrogram_samples:
        indices = np.random.RandomState(42).choice(len(X_scaled), size=max_dendrogram_samples, replace=False)
        X_dendro = X_scaled[indices]
    else:
        X_dendro = X_scaled

    linkage_matrix = linkage(X_dendro, method=method, metric=metric)

    # Fit AgglomerativeClustering on entire dataset
    model = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=method)
    labels = model.fit_predict(X_scaled)

    df_clustered = df.copy()
    df_clustered["cluster"] = labels

    centroids = df_clustered.groupby("cluster")[numeric_cols].mean()
    cluster_sizes = df_clustered["cluster"].value_counts().sort_index()

    return model, linkage_matrix, df_clustered, centroids, cluster_sizes, numeric_cols


def generate_dendrogram_plot(linkage_matrix: np.ndarray, dataset_name: str,
                             metric: str, method: str, output_dir: str = "reports") -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    out_path = Path(output_dir) / f"hierarchical_dendrogram_{dataset_name}_{today}.png"

    plt.figure(figsize=(10, 6))
    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90,
        leaf_font_size=10,
        show_contracted=True
    )
    plt.title(f"Hierarchical Clustering Dendrogram: {dataset_name}\n(Linkage: {method}, Metric: {metric})")
    plt.xlabel("Cluster / Sample Index (p=30 truncated)")
    plt.ylabel("Distance / Branch Height (Dissimilarity)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Dendrogram written to: {out_path}")
    return str(out_path)


def write_report(dataset_name: str, n_clusters: int, metric: str, method: str,
                 cluster_sizes: pd.Series, centroids: pd.DataFrame, dendrogram_path: str,
                 output_dir: str = "reports"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    out_path = Path(output_dir) / f"hierarchical_{dataset_name}_{today}.md"

    lines = [
        f"# Hierarchical Clustering Report: {dataset_name}",
        f"Generated: {today}",
        "",
        "## Configuration",
        f"- Number of clusters ($K$): {n_clusters}",
        f"- Distance metric: `{metric}` (Euclidean / Manhattan)",
        f"- Linkage method: `{method}`",
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
    lines.append(f"- Dendrogram: `{dendrogram_path}`")
    lines.append("")
    lines.append("Note: In a dendrogram, branch height represents the dissimilarity at which clusters are merged. Shorter branches denote higher similarity.")

    out_path.write_text("\n".join(lines))
    print(f"Hierarchical report written to: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Agglomerative Hierarchical Clustering.")
    parser.add_argument("dataset_path", help="Path to input dataset (CSV/Excel)")
    parser.add_argument("--target", default=None, help="Target column name (optional)")
    parser.add_argument("--n-clusters", type=int, default=3, help="Number of clusters to extract")
    parser.add_argument("--metric", default="euclidean", choices=["euclidean", "cityblock", "manhattan", "cosine"])
    parser.add_argument("--linkage", default="ward", choices=["ward", "complete", "average", "single"])
    parser.add_argument("--output-dir", default="reports", help="Output directory for reports/plots")
    parser.add_argument("--data-dir", default="data/processed", help="Output directory for clustered dataset")
    args = parser.parse_args()

    # Normalize manhattan to cityblock for scipy
    dist_metric = "cityblock" if args.metric == "manhattan" else args.metric

    df = load_data(args.dataset_path)
    dataset_name = Path(args.dataset_path).stem.replace("_features", "").replace("_selected", "")

    model, link_mat, df_clustered, centroids, cluster_sizes, num_cols = fit_hierarchical(
        df, args.target, args.n_clusters, dist_metric, args.linkage
    )

    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    csv_out = Path(args.data_dir) / f"{dataset_name}_hierarchical.csv"
    df_clustered.to_csv(csv_out, index=False)
    print(f"Hierarchically clustered dataset written to: {csv_out}")

    dendro_file = generate_dendrogram_plot(link_mat, dataset_name, args.metric, args.linkage, args.output_dir)
    write_report(dataset_name, args.n_clusters, args.metric, args.linkage, cluster_sizes, centroids, dendro_file, args.output_dir)
