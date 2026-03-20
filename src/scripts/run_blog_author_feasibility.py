"""
Blog Authorship Corpus — Feasibility Test

Quick check: are there measurable distributional differences between
blog authors at the embedding level, WITHOUT fine-tuning?

Steps:
  1. Load Blog Authorship Corpus from a local XML directory
  2. Pick top-N authors by post count
  3. Embed posts with sentence-transformers
  4. Measure inter-author vs intra-author cosine distances
  5. Run PCA/t-SNE, visualize clusters
  6. Train a simple logistic classifier on embeddings → author ID

Usage:
  python src/scripts/run_blog_author_feasibility.py --data-dir /path/to/blogs --out-dir results/blog_feasibility
"""

import argparse
import os
import re
import glob
import json
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path


def parse_blog_file(filepath: str) -> list[dict]:
    """Parse a single blog XML file. Each file is one author, multiple posts."""
    with open(filepath, "r", encoding="latin-1") as f:
        text = f.read()

    # Extract metadata from filename: {blogger_id}.{gender}.{age}.{industry}.{sign}.xml
    fname = Path(filepath).stem
    parts = fname.split(".")
    author_id = parts[0] if parts else fname

    # Extract posts between <post>...</post> tags
    posts = re.findall(r"<post>(.*?)</post>", text, re.DOTALL)
    posts = [p.strip() for p in posts if p.strip() and len(p.strip()) > 50]

    return [{"author": author_id, "text": post} for post in posts]


def load_corpus(data_dir: str, min_posts: int = 50, max_authors: int = 10) -> dict:
    """Load corpus, return top authors by post count."""
    files = glob.glob(os.path.join(data_dir, "*.xml"))
    if not files:
        raise FileNotFoundError(f"No XML files found in {data_dir}")

    print(f"Found {len(files)} blog files")

    author_posts = defaultdict(list)
    for f in files:
        try:
            posts = parse_blog_file(f)
            for p in posts:
                author_posts[p["author"]].append(p["text"])
        except Exception:
            continue

    # Filter authors with enough posts
    eligible = {a: posts for a, posts in author_posts.items() if len(posts) >= min_posts}
    print(f"{len(eligible)} authors with >= {min_posts} posts")

    # Take top N by post count
    top_authors = sorted(eligible.keys(), key=lambda a: len(eligible[a]), reverse=True)[
        :max_authors
    ]
    corpus = {a: eligible[a] for a in top_authors}

    for a in top_authors:
        print(f"  Author {a}: {len(corpus[a])} posts")

    return corpus


def embed_posts(corpus: dict, max_posts_per_author: int = 100) -> tuple:
    """Embed posts with sentence-transformers. Returns (embeddings, labels, texts)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    all_texts = []
    all_labels = []
    for author, posts in corpus.items():
        selected = posts[:max_posts_per_author]
        all_texts.extend(selected)
        all_labels.extend([author] * len(selected))

    print(f"Embedding {len(all_texts)} posts...")
    embeddings = model.encode(all_texts, show_progress_bar=True, batch_size=64)

    return np.array(embeddings), all_labels, all_texts


def compute_distances(embeddings: np.ndarray, labels: list) -> dict:
    """Compute intra-author vs inter-author cosine distances."""
    from sklearn.metrics.pairwise import cosine_distances

    dist_matrix = cosine_distances(embeddings)
    unique_authors = sorted(set(labels))
    label_arr = np.array(labels)

    intra_dists = []
    inter_dists = []

    for i, a1 in enumerate(unique_authors):
        mask_a1 = label_arr == a1
        idx_a1 = np.where(mask_a1)[0]

        # Intra: distances within this author
        for j in range(len(idx_a1)):
            for k in range(j + 1, len(idx_a1)):
                intra_dists.append(dist_matrix[idx_a1[j], idx_a1[k]])

        # Inter: distances to other authors
        for a2 in unique_authors[i + 1 :]:
            mask_a2 = label_arr == a2
            idx_a2 = np.where(mask_a2)[0]
            for j in idx_a1[:20]:  # sample to keep it fast
                for k in idx_a2[:20]:
                    inter_dists.append(dist_matrix[j, k])

    return {
        "intra_mean": float(np.mean(intra_dists)),
        "intra_std": float(np.std(intra_dists)),
        "inter_mean": float(np.mean(inter_dists)),
        "inter_std": float(np.std(inter_dists)),
        "separation_ratio": float(np.mean(inter_dists) / np.mean(intra_dists)),
    }


def classify_authors(embeddings: np.ndarray, labels: list) -> dict:
    """Train logistic regression on embeddings → author. Return accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(labels)

    clf = LogisticRegression(max_iter=1000, C=1.0)
    scores = cross_val_score(clf, embeddings, y, cv=5, scoring="accuracy")

    return {
        "mean_accuracy": float(np.mean(scores)),
        "std_accuracy": float(np.std(scores)),
        "num_classes": len(set(labels)),
        "chance_level": 1.0 / len(set(labels)),
    }


def plot_results(
    embeddings: np.ndarray,
    labels: list,
    distances: dict,
    classification: dict,
    out_dir: str,
):
    """Generate visualization: t-SNE clusters + summary stats."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    unique_authors = sorted(set(labels))
    label_arr = np.array(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_authors)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Blog Author Feasibility Test — Are Authors Separable?", fontsize=13, fontweight="bold")

    # Panel 1: t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(embeddings)

    for i, author in enumerate(unique_authors):
        mask = label_arr == author
        axes[0].scatter(
            coords[mask, 0], coords[mask, 1],
            c=[colors[i]], label=f"Author {author}", alpha=0.5, s=15,
        )
    axes[0].set_title("t-SNE of Post Embeddings")
    axes[0].legend(fontsize=6, markerscale=2, loc="best")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Panel 2: Intra vs Inter distance distributions
    axes[1].bar(
        ["Intra-author", "Inter-author"],
        [distances["intra_mean"], distances["inter_mean"]],
        yerr=[distances["intra_std"], distances["inter_std"]],
        color=["#4477AA", "#EE6677"], capsize=8,
    )
    axes[1].set_title(f"Cosine Distance (ratio: {distances['separation_ratio']:.2f})")
    axes[1].set_ylabel("Mean Cosine Distance")

    # Panel 3: Classification accuracy
    acc = classification["mean_accuracy"]
    chance = classification["chance_level"]
    axes[2].bar(
        ["Classifier\nAccuracy", "Chance\nLevel"],
        [acc, chance],
        color=["#228833", "#BBBBBB"],
    )
    axes[2].set_title(f"5-Fold LogReg: {acc:.1%} vs {chance:.1%} chance")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0, 1)
    for j, v in enumerate([acc, chance]):
        axes[2].text(j, v + 0.02, f"{v:.1%}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(out_dir, "blog_author_feasibility.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Blog Authorship Feasibility Test")
    parser.add_argument("--data-dir", required=True, help="Path to extracted blog XML files")
    parser.add_argument("--out-dir", default="results/blog_feasibility", help="Output directory")
    parser.add_argument("--num-authors", type=int, default=10, help="Number of top authors to test")
    parser.add_argument("--min-posts", type=int, default=50, help="Minimum posts per author")
    parser.add_argument("--max-posts", type=int, default=100, help="Max posts per author for embedding")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=== Loading Blog Authorship Corpus ===")
    corpus = load_corpus(args.data_dir, min_posts=args.min_posts, max_authors=args.num_authors)

    print("\n=== Embedding Posts ===")
    embeddings, labels, texts = embed_posts(corpus, max_posts_per_author=args.max_posts)

    print("\n=== Computing Distances ===")
    distances = compute_distances(embeddings, labels)
    print(f"  Intra-author: {distances['intra_mean']:.4f} ± {distances['intra_std']:.4f}")
    print(f"  Inter-author: {distances['inter_mean']:.4f} ± {distances['inter_std']:.4f}")
    print(f"  Separation ratio: {distances['separation_ratio']:.2f}")

    print("\n=== Classifying Authors ===")
    classification = classify_authors(embeddings, labels)
    print(f"  5-fold accuracy: {classification['mean_accuracy']:.1%} ± {classification['std_accuracy']:.1%}")
    print(f"  Chance level: {classification['chance_level']:.1%}")

    print("\n=== Generating Visualization ===")
    fig_path = plot_results(embeddings, labels, distances, classification, args.out_dir)

    # Save metrics
    metrics = {
        "distances": distances,
        "classification": classification,
        "num_authors": len(corpus),
        "posts_per_author": {a: len(p) for a, p in corpus.items()},
        "figure": fig_path,
    }
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Verdict
    print("\n=== VERDICT ===")
    if classification["mean_accuracy"] > 2 * classification["chance_level"]:
        print("YES — Authors are clearly separable. Proceed with persona experiments.")
    elif classification["mean_accuracy"] > 1.5 * classification["chance_level"]:
        print("MAYBE — Some signal exists. Fine-tuning may amplify it.")
    else:
        print("NO — Authors not separable at embedding level. Reconsider approach.")


if __name__ == "__main__":
    main()
