# -*- coding: utf-8 -*-


import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 1. 定位文件 ──────────────────────────────────────────────────
files = os.listdir('.')
f1 = [f for f in files if '20250908' in f and f.endswith('.xlsx')][0]
f2 = [f for f in files if '20260423' in f and f.endswith('.xlsx')][0]

print(f'File 1: {f1}')
print(f'File 2: {f2}')

# ── 2. 读取数据 ─────────────────────────────────────────────────
cols = ['avg_a_radius', 'avg_b_radius', 'avg_a_en', 'avg_b_en',
        'avg_a_sa', 'avg_b_sa', 'avg_a_ie', 'avg_b_ie', 't']

X1 = pd.read_excel(f1)[cols].values.astype(np.float64)
X2 = pd.read_excel(f2)[cols].values.astype(np.float64)

n1, n2 = len(X1), len(X2)
print(f'Experimental: {n1} rows, Predicted: {n2} rows')

# ── 3. 标准化 → t-SNE 合并拟合 ──────────────────────────────────
X_all = np.vstack([X1, X2])
X_scaled = StandardScaler().fit_transform(X_all)

# perplexity 须 < 样本数; 取 min(30, n-1)
perp = min(30, len(X_scaled) - 1)
tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
            n_iter=1000, verbose=0)
X_tsne = tsne.fit_transform(X_scaled)

# 拆分回两组
X1_tsne = X_tsne[:n1]
X2_tsne = X_tsne[n1:]

# ── 4. 绘制散点图 ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(X1_tsne[:, 0], X1_tsne[:, 1],
           c='red', marker='o', s=60, alpha=0.75,
           edgecolors='darkred', linewidth=0.5,
           label=f'Experimental (n={n1})')

ax.scatter(X2_tsne[:, 0], X2_tsne[:, 1],
           c='blue', marker='o', s=30, alpha=0.35,
           edgecolors='darkblue', linewidth=0.3,
           label=f'Predicted (n={n2})')

ax.set_xlabel('t-SNE Component 1', fontsize=13)
ax.set_ylabel('t-SNE Component 2', fontsize=13)
ax.set_title(f't-SNE of ABO3 Descriptors (9 features)\n'
             f'perplexity={perp}, experimental + predicted combined',
             fontsize=14)
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()

out_path = 'tsne_scatter.png'
fig.savefig(out_path, dpi=200)
print(f'\nPlot saved to: {out_path}')
