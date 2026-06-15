# -*- coding: utf-8 -*-


import os
import warnings
import pandas as pd
import numpy as np
from scipy.stats import chi2
from scipy.spatial import ConvexHull
from scipy.spatial.distance import mahalanobis
from matplotlib.path import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# 0. 数据加载
# ═══════════════════════════════════════════════════════════════════
files = os.listdir('.')
f1 = [f for f in files if '20250908' in f and f.endswith('.xlsx')][0]
f2 = [f for f in files if '20260423' in f and f.endswith('.xlsx')][0]

cols = ['avg_a_radius', 'avg_b_radius', 'avg_a_en', 'avg_b_en',
        'avg_a_sa', 'avg_b_sa', 'avg_a_ie', 'avg_b_ie', 't']

X_train = pd.read_excel(f1)[cols].values.astype(np.float64)
X_pred  = pd.read_excel(f2)[cols].values.astype(np.float64)

n_train, n_pred = len(X_train), len(X_pred)
d = len(cols)

print(f'Training: {n_train} x {d}  |  Predicted: {n_pred} x {d}')

# ═══════════════════════════════════════════════════════════════════
# 1. 标准化 + PCA (与 pca_plot.py 一致)
# ═══════════════════════════════════════════════════════════════════
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_pred_s  = scaler.transform(X_pred)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_s)
X_pred_pca  = pca.transform(X_pred_s)

evr = pca.explained_variance_ratio_
print(f'PCA: PC1={evr[0]:.3f}, PC2={evr[1]:.3f}, sum={evr.sum():.3f}')

# ═══════════════════════════════════════════════════════════════════
# 2. 2D 凸包分析 (在 PCA 空间)
# ═══════════════════════════════════════════════════════════════════
hull = ConvexHull(X_train_pca)
hull_path = Path(X_train_pca[hull.vertices])  # 闭合多边形

inside_train = hull_path.contains_points(X_train_pca, radius=1e-8)
inside_pred  = hull_path.contains_points(X_pred_pca,  radius=1e-8)

n_inside_train = inside_train.sum()
n_inside_pred  = inside_pred.sum()

print(f'2D Convex Hull: {hull.vertices.shape[0]}/{n_train} vertices')
print(f'Training inside hull:  {n_inside_train}/{n_train} ({100*n_inside_train/n_train:.1f}%)')
print(f'Predicted inside hull: {n_inside_pred}/{n_pred} ({100*n_inside_pred/n_pred:.1f}%)')

# ═══════════════════════════════════════════════════════════════════
# 3. 马氏距离 (9D 标准化空间，与之前一致)
# ═══════════════════════════════════════════════════════════════════
lw = LedoitWolf().fit(X_train_s)
cov_inv = np.linalg.pinv(lw.covariance_)
mean_train = np.mean(X_train_s, axis=0)

md_train = np.array([mahalanobis(X_train_s[i], mean_train, cov_inv)
                      for i in range(n_train)])
md_pred  = np.array([mahalanobis(X_pred_s[i], mean_train, cov_inv)
                      for i in range(n_pred)])

thresh_95 = np.sqrt(chi2.ppf(0.95, d))
thresh_99 = np.sqrt(chi2.ppf(0.99, d))

pct_outside_95 = np.mean(md_pred > thresh_95) * 100
pct_outside_99 = np.mean(md_pred > thresh_99) * 100

print(f'Mahalanobis: train mean={md_train.mean():.2f}, '
      f'pred mean={md_pred.mean():.2f}')
print(f'Pred outside 95%: {pct_outside_95:.1f}%  |  '
      f'outside 99%: {pct_outside_99:.1f}%')

# ═══════════════════════════════════════════════════════════════════
# 4. 合并画图
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# ── 4a. PCA + 2D 凸包 ────────────────────────────────────────────
ax = axes[0]

# 先画预测点：凸包外 = 橙色，凸包内 = 绿色
mask_in  = inside_pred
mask_out = ~inside_pred

ax.scatter(X_pred_pca[mask_out, 0], X_pred_pca[mask_out, 1],
           c='#FF8C00', marker='o', s=18, alpha=0.45, edgecolors='none',
           label=f'Predicted OUTSIDE hull ({mask_out.sum()})')
ax.scatter(X_pred_pca[mask_in, 0], X_pred_pca[mask_in, 1],
           c='#2E7D32', marker='o', s=18, alpha=0.55, edgecolors='none',
           label=f'Predicted INSIDE hull ({mask_in.sum()})')

# 训练点（红色）
ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1],
           c='red', marker='o', s=55, alpha=0.85,
           edgecolors='darkred', linewidth=0.6,
           label=f'Training (n={n_train})')

# 凸包边界
for simplex in hull.simplices:
    ax.plot(X_train_pca[simplex, 0], X_train_pca[simplex, 1],
            'k-', linewidth=1.2, alpha=0.8)

# 填充凸包
hull_verts = X_train_pca[hull.vertices]
ax.fill(hull_verts[:, 0], hull_verts[:, 1],
        facecolor='red', alpha=0.06, edgecolor='none')

ax.set_xlabel(f'PC1 ({evr[0]:.1%} variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({evr[1]:.1%} variance)', fontsize=12)
ax.set_title(f'2D PCA Convex Hull\n'
             f'Training hull: {n_inside_pred}/{n_pred} predicted inside '
             f'({100*n_inside_pred/n_pred:.1f}%)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='best', framealpha=0.8)
ax.grid(True, alpha=0.25)

# ── 4b. 马氏距离分布 ─────────────────────────────────────────────
ax = axes[1]
bins = np.linspace(0, max(md_train.max(), md_pred.max(), thresh_99) * 1.1, 45)

ax.hist(md_train, bins=bins, color='red', alpha=0.5,
        label=f'Training (n={n_train})', density=True)
ax.hist(md_pred, bins=bins, color='blue', alpha=0.35,
        label=f'Predicted (n={n_pred})', density=True)

ax.axvline(thresh_95, color='orange', linestyle='--', linewidth=2,
           label=f'χ² 95% threshold ({thresh_95:.2f})')
ax.axvline(thresh_99, color='darkred', linestyle=':', linewidth=2,
           label=f'χ² 99% threshold ({thresh_99:.2f})')

# 标注关键数值 — 放在右下角，避开密度峰值区域
ax.text(0.98, 0.12,
        f'Training: mean={md_train.mean():.2f}, median={np.median(md_train):.2f}\n'
        f'Predicted: mean={md_pred.mean():.2f}, median={np.median(md_pred):.2f}\n'
        f'Pred outside 95%: {pct_outside_95:.1f}%  |  outside 99%: {pct_outside_99:.1f}%',
        transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))

ax.set_xlabel('Mahalanobis Distance', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Mahalanobis Distance Distribution\n'
             '(9D standardized space, Ledoit-Wolf covariance)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig('convex_hull_mahalanobis.png', dpi=200)
print(f'\nFigure saved: convex_hull_mahalanobis.png')


