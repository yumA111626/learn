"""ユーザーベース協調フィルタリングの学習用サンプル。

実行: python collaborative_filtering.py --user Alice --top-n 2
必要なライブラリ: numpy, matplotlib
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_data():
    """行はユーザー、列は映画。1〜5が評価、NaNが未評価。"""
    users = ["Alice", "Bob", "Carol", "Dave", "Eve"]
    items = ["Movie A", "Movie B", "Movie C", "Movie D", "Movie E"]
    ratings = np.array([
        [5, 4, 1, np.nan, np.nan],
        [5, 4, 1, 5, 2],
        [4, 5, 2, 4, 1],
        [1, 2, 5, 1, 5],
        [2, 1, 4, 2, 4],
    ], dtype=float)
    return users, items, ratings


def user_similarities(ratings, target, min_common=2):
    """共通して評価した映画だけでPearson相関係数を求める。"""
    similarities = np.zeros(len(ratings))
    for other in range(len(ratings)):
        if other == target:
            continue
        common = np.isfinite(ratings[target]) & np.isfinite(ratings[other])
        if common.sum() < min_common:
            continue
        left = ratings[target, common]
        right = ratings[other, common]
        left = left - left.mean()
        right = right - right.mean()
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        # 一定の評価しかない場合は相関を定義できないので0とする。
        if denominator > 0:
            similarities[other] = np.clip(left @ right / denominator, -1, 1)
    return similarities


def recommend(ratings, target, top_n=3, neighbors=2):
    """正の相関を持つ近傍ユーザーの、平均からの評価差で予測する。"""
    ratings = np.asarray(ratings, dtype=float)
    if ratings.ndim != 2 or not 0 <= target < len(ratings):
        raise ValueError("評価行列と対象ユーザーの番号を確認してください。")
    if top_n < 1 or neighbors < 1:
        raise ValueError("top_nとneighborsは1以上にしてください。")
    known = np.isfinite(ratings)
    if np.isinf(ratings).any() or np.any((ratings[known] < 1) | (ratings[known] > 5)):
        raise ValueError("評価は1〜5、未評価はNaNにしてください。")
    if not known[target].any():
        raise ValueError("対象ユーザーの評価がないため類似度を計算できません。")

    # 1. ユーザー同士の好みの近さを計算する。
    similarities = user_similarities(ratings, target)
    counts = known.sum(axis=1)
    means = np.divide(np.nansum(ratings, axis=1), counts,
                      out=np.zeros(len(ratings)), where=counts > 0)
    predictions = []
    for item in np.flatnonzero(~known[target]):
        # 2. この映画を評価済みで、好みが似たユーザーを選ぶ。
        candidates = np.flatnonzero((similarities > 0) & known[:, item])
        nearest = candidates[np.argsort(-similarities[candidates], kind="stable")[:neighbors]]
        if not len(nearest):
            # 根拠となる近傍がなければ、この映画は推薦候補にしない。
            continue
        # 3. 甘口・辛口の差を補正して、未評価の映画の評価を予測する。
        weights = similarities[nearest]
        deviations = ratings[nearest, item] - means[nearest]
        score = means[target] + weights @ deviations / weights.sum()
        predictions.append((int(item), float(np.clip(score, 1, 5))))

    # 4. 未評価の映画を予測評価の高い順に推薦する。
    predictions.sort(key=lambda result: result[1], reverse=True)
    return predictions[:top_n], similarities


def plot_result(users, items, ratings, target, recommendations, output, show=True):
    if not show:
        plt.switch_backend("Agg")
    predicted = ratings.copy()
    for item, score in recommendations:
        predicted[target, item] = score
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, values, title in zip(axes, [ratings, predicted],
                                 ["Observed ratings", "Recommended ratings (*)"]):
        heatmap = ax.imshow(np.ma.masked_invalid(values), vmin=1, vmax=5, cmap="YlGnBu")
        ax.set_xticks(range(len(items)), items, rotation=30, ha="right")
        ax.set_yticks(range(len(users)), users)
        ax.set_title(title)
        for row in range(len(users)):
            for col in range(len(items)):
                value = values[row, col]
                label = f"{value:.1f}" if np.isfinite(value) else "?"
                if np.isnan(ratings[row, col]) and np.isfinite(value):
                    label += "*"
                ax.text(col, row, label, ha="center", va="center",
                        color="white" if value >= 3.5 else "black")
    fig.colorbar(heatmap, ax=axes.tolist(), label="Rating (1-5)", shrink=0.8)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"グラフ保存先: {output.resolve()}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    users, items, ratings = make_data()
    parser = argparse.ArgumentParser(description="ユーザーベース協調フィルタリング")
    parser.add_argument("--user", choices=users, default="Alice", help="推薦対象")
    parser.add_argument("--top-n", type=int, default=3, help="推薦件数")
    parser.add_argument("--neighbors", type=int, default=2, help="映画ごとに使う近傍人数")
    parser.add_argument("--no-show", action="store_true", help="画面表示せず画像を保存")
    parser.add_argument("--output", type=Path, default=Path("collaborative_filtering_result.png"))
    args = parser.parse_args()
    target = users.index(args.user)
    try:
        recommendations, similarities = recommend(ratings, target, args.top_n, args.neighbors)
    except ValueError as error:
        parser.error(str(error))
    print(f"推薦対象: {args.user}")
    print("\nユーザー間の類似度（Pearson相関係数）:")
    for user, similarity in zip(users, similarities):
        if user != args.user:
            print(f"  {user}: {similarity:+.3f}")
    print("\nおすすめ映画（未評価のみ・予測評価順）:")
    for rank, (item, score) in enumerate(recommendations, start=1):
        print(f"  {rank}. {items[item]}: 予測評価 {score:.2f}")
    if not recommendations:
        print("  推薦候補がありません（未評価の映画や似たユーザーが不足）。")
    plot_result(users, items, ratings, target, recommendations, args.output, not args.no_show)


if __name__ == "__main__":
    main()
