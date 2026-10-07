"""
PCA Dimensionality Reduction helper based on skills/pca.md.
Usage:
  ./.venv/bin/python src/pca_analysis.py data/processed/titanic_features.csv [--target Survived] [--n-components 5]
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
from sklearn.decomposition import PCA


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    elif p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")


def run_pca(df: pd.DataFrame, target: str | None = None, n_components: int | None = None):
    if target and target in df.columns:
        y = df[target]
        X = df.drop(columns=[target])
    else:
        y = None
        X = df.copy()

    # Select only numeric features
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric features found for PCA.")
    X_num = X[numeric_cols]

    if X_num.isnull().any().any():
        bad_cols = X_num.columns[X_num.isnull().any()].tolist()
        raise ValueError(f"Numeric features contain missing values: {bad_cols}. Clean data first.")

    # Center and scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    max_pcs = min(X_scaled.shape[0], X_scaled.shape[1])
    n_comps = n_components if n_components is not None else min(max_pcs, 10)
    n_comps = min(n_comps, max_pcs)

    pca = PCA(n_components=n_comps, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    pc_names = [f"PC{i+1}" for i in range(n_comps)]
    pca_df = pd.DataFrame(X_pca, columns=pc_names, index=df.index)
    if y is not None:
        pca_df[target] = y

    # Loadings (eigenvectors): features x components
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=pc_names,
        index=numeric_cols
    )

    return pca, pca_df, loadings, numeric_cols, y


def generate_scree_plot(pca, dataset_name: str, output_dir: str = "reports") -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    out_path = Path(output_dir) / f"pca_scree_{dataset_name}_{today}.png"

    n_comps = len(pca.explained_variance_ratio_)
    pc_labels = [f"PC{i+1}" for i in range(n_comps)]
    var_exp = pca.explained_variance_ratio_ * 100
    cum_var = np.cumsum(var_exp)

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.bar(pc_labels, var_exp, color="steelblue", alpha=0.8, label="Individual Variance (%)")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_title(f"PCA Scree Plot & Cumulative Variance: {dataset_name}")

    ax2 = ax1.twinx()
    ax2.plot(pc_labels, cum_var, color="darkorange", marker="o", linewidth=2, label="Cumulative Variance (%)")
    ax2.axhline(80, color="gray", linestyle="--", alpha=0.7, label="80% Threshold")
    ax2.set_ylabel("Cumulative Variance (%)", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax2.set_ylim([0, 105])

    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Scree plot written to: {out_path}")
    return str(out_path)


def generate_2d_plot(pca_df: pd.DataFrame, target: str | None, dataset_name: str, output_dir: str = "reports") -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    out_path = Path(output_dir) / f"pca_2d_{dataset_name}_{today}.png"

    plt.figure(figsize=(8, 6))
    if target and target in pca_df.columns:
        y = pca_df[target]
        scatter = plt.scatter(pca_df["PC1"], pca_df["PC2"], c=y, cmap="viridis", alpha=0.6, s=20)
        plt.colorbar(scatter, label=target)
    else:
        plt.scatter(pca_df["PC1"], pca_df["PC2"], color="teal", alpha=0.6, s=20)

    plt.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    plt.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA 2D Projection (PC1 vs PC2): {dataset_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"2D PCA plot written to: {out_path}")
    return str(out_path)


def write_report(pca, loadings: pd.DataFrame, dataset_name: str, scree_path: str, plot2d_path: str, output_dir: str = "reports"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    out_path = Path(output_dir) / f"pca_{dataset_name}_{today}.md"

    var_exp = pca.explained_variance_ratio_ * 100
    cum_var = np.cumsum(var_exp)
    eigenvalues = pca.explained_variance_

    lines = [
        f"# Principal Component Analysis (PCA) Report: {dataset_name}",
        f"Generated: {today}",
        "",
        "## Summary",
        f"- Components evaluated: {len(var_exp)}",
        f"- Variance explained by PC1: {var_exp[0]:.2f}%",
        f"- Variance explained by PC1 + PC2: {cum_var[1]:.2f}%" if len(cum_var) > 1 else "",
        "",
        "## Explained Variance Table",
        "| Component | Eigenvalue | Variance (%) | Cumulative Variance (%) |",
        "|---|---|---|---|",
    ]
    for i in range(len(var_exp)):
        lines.append(f"| PC{i+1} | {eigenvalues[i]:.4f} | {var_exp[i]:.2f}% | {cum_var[i]:.2f}% |")

    lines.append("")
    lines.append("## Top Loading Scores for PC1 & PC2")
    lines.append("Loading scores represent the weight/contribution of each feature to the principal component unit vector.")
    lines.append("")

    for pc in ["PC1", "PC2"]:
        if pc in loadings.columns:
            top_loadings = loadings[pc].abs().sort_values(ascending=False).head(5)
            lines.append(f"### Top 5 Influential Features for `{pc}`")
            lines.append("| Feature | Loading Score | Absolute Magnitude |")
            lines.append("|---|---|---|")
            for feat in top_loadings.index:
                val = loadings.loc[feat, pc]
                lines.append(f"| `{feat}` | {val:+.4f} | {abs(val):.4f} |")
            lines.append("")

    lines.append("## Plots")
    lines.append(f"- Scree plot: `{scree_path}`")
    lines.append(f"- 2D projection: `{plot2d_path}`")
    lines.append("")
    lines.append("Note: The elbow on the scree plot indicates the dimensionality reduction cutoff point.")

    out_path.write_text("\n".join(lines))
    print(f"PCA report written to: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PCA analysis.")
    parser.add_argument("dataset_path", help="Path to input dataset (CSV/Excel)")
    parser.add_argument("--target", default=None, help="Target column name (optional)")
    parser.add_argument("--n-components", type=int, default=None, help="Number of principal components")
    parser.add_argument("--output-dir", default="reports", help="Output directory for reports/plots")
    parser.add_argument("--data-dir", default="data/processed", help="Output directory for transformed dataset")
    args = parser.parse_args()

    df = load_data(args.dataset_path)
    dataset_name = Path(args.dataset_path).stem.replace("_features", "").replace("_selected", "")

    pca, pca_df, loadings, num_cols, y = run_pca(df, args.target, args.n_components)

    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    csv_out = Path(args.data_dir) / f"{dataset_name}_pca.csv"
    pca_df.to_csv(csv_out, index=False)
    print(f"PCA transformed dataset written to: {csv_out}")

    scree_file = generate_scree_plot(pca, dataset_name, args.output_dir)
    plot2d_file = generate_2d_plot(pca_df, args.target, dataset_name, args.output_dir)
    write_report(pca, loadings, dataset_name, scree_file, plot2d_file, args.output_dir)
