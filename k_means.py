"""k-means法を手順ごとに体験するハンズオン。

実行: python k_means.py
例:   python k_means.py --k 4 --seed 10
必要なライブラリ: numpy, matplotlib
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_data(seed=42):
    """正解ラベルを使わずに分類する、2次元のサンプルデータ。"""
    rng = np.random.default_rng(seed)
    return np.vstack([
        rng.normal(loc=center, scale=0.8, size=(100, 2))
        for center in [(-4, -2), (0, 4), (4, -1)]
    ])


def k_means(data, k=3, seed=42, max_iter=100, tol=1e-6):
    """割り当てと平均の計算を繰り返す。履歴も可視化用に返す。"""
    if not 1 <= k <= len(data):
        raise ValueError("kは1以上、データ数以下にしてください。")
    if max_iter < 1:
        raise ValueError("max_iterは1以上にしてください。")

    rng = np.random.default_rng(seed)
    # 1. データから重複なくk個を選び、初期重心にする。
    centers = data[rng.choice(len(data), size=k, replace=False)].copy()
    history = [centers.copy()]

    for iteration in range(1, max_iter + 1):
        # 2. 各点から各重心までのユークリッド距離の二乗を求める。
        #    distancesの形は「データ数 × クラスタ数」。
        distances = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)

        # 3. クラスタに属する点の平均を、新しい重心にする。
        new_centers = centers.copy()
        for cluster in range(k):
            members = data[labels == cluster]
            if len(members) > 0:
                new_centers[cluster] = members.mean(axis=0)
            # 空のクラスタができた場合は、前の重心を維持する。

        movement = np.linalg.norm(new_centers - centers, axis=1).max()
        centers = new_centers
        history.append(centers.copy())
        print(f"反復 {iteration:2d}: 重心の最大移動距離 = {movement:.6f}")

        # 4. 重心がほぼ動かなくなったら終了。
        if movement <= tol:
            break

    # 最終的な重心に対して割り当てと目的関数を計算し直す。
    distances = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = distances.argmin(axis=1)
    inertia = distances[np.arange(len(data)), labels].sum()
    return labels, centers, np.array(history), float(inertia)


def plot_result(data, labels, centers, history, output, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    axes[0].scatter(data[:, 0], data[:, 1], c="gray", s=20, alpha=0.65)
    axes[0].scatter(history[0, :, 0], history[0, :, 1],
                    c="red", marker="X", s=180, label="Initial centers")
    axes[0].set_title("Before clustering")
    axes[0].legend()

    axes[1].scatter(data[:, 0], data[:, 1], c=labels, cmap="tab10", s=20, alpha=0.7)
    for cluster in range(len(centers)):
        axes[1].plot(history[:, cluster, 0], history[:, cluster, 1],
                     "k--", linewidth=1, alpha=0.6)
    axes[1].scatter(centers[:, 0], centers[:, 1], c="red", marker="X",
                    s=180, label="Final centers")
    axes[1].set_title(f"After k-means (k={len(centers)})")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.grid(alpha=0.2)
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"グラフ保存先: {output.resolve()}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="k-means法の学習用サンプル")
    parser.add_argument("--k", type=int, default=3, help="クラスタ数（既定: 3）")
    parser.add_argument("--seed", type=int, default=42, help="初期重心の乱数シード")
    parser.add_argument("--max-iter", type=int, default=100, help="反復回数の上限")
    parser.add_argument("--no-show", action="store_true", help="画面表示せず画像を保存")
    parser.add_argument("--output", type=Path, default=Path("k_means_result.png"))
    args = parser.parse_args()

    # データは固定し、--seedで初期重心だけを変えて比較できるようにする。
    data = make_data(seed=42)
    try:
        labels, centers, history, inertia = k_means(
            data, k=args.k, seed=args.seed, max_iter=args.max_iter
        )
    except ValueError as error:
        parser.error(str(error))

    print(f"\nクラスタ数: {args.k}")
    print(f"クラスタごとの点の数: {np.bincount(labels, minlength=args.k)}")
    print(f"最終重心:\n{centers}")
    print(f"クラスタ内平方和（各点と所属重心の距離の二乗の合計）: {inertia:.3f}")
    plot_result(data, labels, centers, history, args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
