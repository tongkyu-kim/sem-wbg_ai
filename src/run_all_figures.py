"""
Generate all Section 9 + Section 10 outputs in one run.
Run from repo root:  python src/run_all_figures.py

Outputs
-------
output/tables/item_regressions_DATE.xlsx      (3 sheets)
output/tables/item_regressions_DATE.csv
output/figures/shap_importance_DATE.png       (cross-outcome bar chart)
output/figures/shap_beeswarm_Attitude_DATE.png
output/figures/shap_beeswarm_Beh_Environmental_DATE.png
output/figures/shap_beeswarm_Beh_Responsible_DATE.png
output/tables/shap_importance_DATE.xlsx       (2 sheets)
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # headless – no display needed
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import t as t_dist
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import shap

warnings.filterwarnings('ignore')

DATE    = datetime.today().strftime('%Y%m%d')
OUT_FIG = 'output/figures'
OUT_TAB = 'output/tables'
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_TAB, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_excel('src/raw/wbg_ai.xlsx')
print(f'Loaded: N={len(df)}, cols={df.shape[1]}')

# ── Constructs & helpers ───────────────────────────────────────────────────
CONSTRUCTS = {
    'Awareness_Worry':   ['aw_error', 'aw_ethics', 'aw_privacy', 'aw_deepfake'],
    'Awareness_Env':     ['aw_env_dc', 'aw_env_use'],
    'Attitude':          ['att_pos', 'att_easy', 'att_skill_worry'],
    'Beh_Environmental': ['beh_env_avoid', 'beh_advise'],
    'Beh_Responsible':   ['beh_resp', 'beh_check', 'beh_harmful', 'beh_prompt'],
}
REVERSE_ITEMS = ['att_skill_worry']

STRUCT_MODEL = {
    'Attitude':          ['Awareness_Worry', 'Awareness_Env'],
    'Beh_Environmental': ['Attitude'],
    'Beh_Responsible':   ['Attitude'],
}

def prep(items):
    sub = df[items].copy()
    for c in REVERSE_ITEMS:
        if c in sub.columns:
            sub[c] = 6 - sub[c]
    return sub

def pls_nipals(constructs, struct_model, n_iter=300, tol=1e-7):
    # PLS-SEM NIPALS: centroid inner weighting, reflective outer OLS.
    n = len(df)
    blocks = {}
    for cname, items in constructs.items():
        blocks[cname] = StandardScaler().fit_transform(prep(items))

    lv = {}
    for cname in constructs:
        s = blocks[cname].mean(axis=1)
        std = s.std(ddof=1)
        lv[cname] = (s - s.mean()) / (std if std > 0 else 1.0)

    adj = {c: set() for c in constructs}
    for outcome, preds in struct_model.items():
        for p in preds:
            adj[outcome].add(p)
            adj[p].add(outcome)

    outer_w = {}
    for _ in range(n_iter):
        old = {c: lv[c].copy() for c in constructs}

        inner = {}
        for c in constructs:
            s = np.zeros(n)
            for c2 in adj[c]:
                s += np.sign(np.corrcoef(lv[c], lv[c2])[0, 1]) * lv[c2]
            inner[c] = s

        for c in constructs:
            X, z = blocks[c], inner[c]
            denom = float(z @ z)
            w = X.T @ z / denom if denom > 0 else np.ones(X.shape[1]) / X.shape[1]
            outer_w[c] = w
            raw = X @ w
            std = raw.std(ddof=1)
            lv[c] = (raw - raw.mean()) / (std if std > 0 else 1.0)

        if sum(np.sum((lv[c] - old[c]) ** 2) for c in constructs) < tol:
            break

    outer_loadings = {}
    for c, items in constructs.items():
        X, y = blocks[c], lv[c]
        outer_loadings[c] = np.array(
            [np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])]
        )

    return pd.DataFrame({c: lv[c] for c in constructs}), outer_w, outer_loadings

factor_scores, pls_outer_w, pls_outer_loadings = pls_nipals(CONSTRUCTS, STRUCT_MODEL)
print('PLS-SEM NIPALS converged.')

def ols_path(y_name, x_names, data):
    y     = data[y_name].values
    X     = data[x_names].values
    n, k  = X.shape
    Xa    = np.column_stack([np.ones(n), X])
    beta  = np.linalg.lstsq(Xa, y, rcond=None)[0]
    yp    = Xa @ beta
    rss   = np.sum((y - yp) ** 2)
    tss   = np.sum((y - y.mean()) ** 2)
    r2    = 1 - rss / tss
    adj   = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    se2   = rss / (n - k - 1)
    cov   = se2 * np.linalg.inv(Xa.T @ Xa)
    sy    = y.std(ddof=1)
    rows  = []
    for i, xn in enumerate(x_names):
        b    = beta[i + 1]
        se_b = np.sqrt(cov[i + 1, i + 1])
        t_v  = b / se_b
        p_v  = 2 * t_dist.sf(abs(t_v), df=n - k - 1)
        rows.append({
            'Outcome':   y_name,
            'Predictor': xn,
            'β (std)':   round(b * data[xn].std(ddof=1) / sy, 3),
            'B (unstd)': round(b, 3),
            'SE':        round(se_b, 3),
            't':         round(t_v, 3),
            'p':         round(p_v, 4),
            'R²':        round(r2, 3),
            'Adj.R²':    round(adj, 3),
        })
    return rows

def sig_tag(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''

# ── Item metadata ──────────────────────────────────────────────────────────
AW_ITEMS  = ['aw_error', 'aw_ethics', 'aw_privacy', 'aw_deepfake',
             'aw_env_dc', 'aw_env_use']
ATT_ITEMS = ['att_pos', 'att_easy', 'att_skill_worry']

ITEM_LABELS = {
    'aw_error':        'AI error risk',
    'aw_ethics':       'AI ethics risk',
    'aw_privacy':      'AI privacy risk',
    'aw_deepfake':     'Deepfake risk',
    'aw_env_dc':       'Data-centre energy',
    'aw_env_use':      'Cumulative AI energy',
    'att_pos':         'Positive attitude',
    'att_easy':        'Ease of use',
    'att_skill_worry': 'Skill worry (R)',
}

iv_df    = prep(AW_ITEMS + ATT_ITEMS).reset_index(drop=True)
fs_df    = factor_scores.reset_index(drop=True)
reg_data = pd.concat([iv_df, fs_df], axis=1)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9 — Item-level regressions
# ═══════════════════════════════════════════════════════════════════════════
print('\n── Section 9: Item-level simple regressions ──────────────────────')

all_rows = []
for item in AW_ITEMS:
    for dv in ['Attitude', 'Beh_Environmental', 'Beh_Responsible']:
        all_rows.extend(ols_path(dv, [item], reg_data))

for item in ATT_ITEMS:
    for dv in ['Beh_Environmental', 'Beh_Responsible']:
        all_rows.extend(ols_path(dv, [item], reg_data))

reg_df = pd.DataFrame(all_rows)
reg_df['Item Label'] = reg_df['Predictor'].map(ITEM_LABELS)
reg_df['Sig']        = reg_df['p'].apply(sig_tag)

HDR = '  {:<22} {:<22} {:>6} {:>7} {:>7} {:>7} {:>7} {:>6}'.format(
      'Item', 'DV', 'beta', 'B', 'SE', 't', 'p', 'R2')
SEP = '  ' + '-' * 82
print(HDR); print(SEP)
for _, r in reg_df.iterrows():
    print('  {:<22} {:<22} {:>6.3f} {:>7.3f} {:>7.3f} {:>7.3f} {:>7.4f} {:>6.3f}  {}'.format(
        r['Predictor'], r['Outcome'], r['β (std)'], r['B (unstd)'],
        r['SE'], r['t'], r['p'], r['R²'], r['Sig']))

cols_out = ['Predictor', 'Item Label', 'Outcome', 'β (std)', 'B (unstd)',
            'SE', 't', 'p', 'R²', 'Adj.R²', 'Sig']
reg_df = reg_df[cols_out]

csv_path  = f'{OUT_TAB}/item_regressions_{DATE}.csv'
xlsx_path = f'{OUT_TAB}/item_regressions_{DATE}.xlsx'
reg_df.to_csv(csv_path, index=False)

with pd.ExcelWriter(xlsx_path, engine='openpyxl') as w:
    aw_mask = reg_df['Predictor'].isin(AW_ITEMS)
    reg_df[aw_mask].to_excel(w, sheet_name='9a_Awareness',  index=False)
    reg_df[~aw_mask].to_excel(w, sheet_name='9b_Attitude',   index=False)
    reg_df.to_excel(w,           sheet_name='All',           index=False)

print(f'\nSaved -> {xlsx_path}')
print(f'Saved -> {csv_path}')

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10 — SHAP (TreeSHAP via GradientBoostingRegressor)
# ═══════════════════════════════════════════════════════════════════════════
print('\n── Section 10: SHAP Analysis ─────────────────────────────────────')

MODELS_10 = {
    'Attitude':          AW_ITEMS,
    'Beh_Environmental': AW_ITEMS + ATT_ITEMS,
    'Beh_Responsible':   AW_ITEMS + ATT_ITEMS,
}

fitted_10  = {}
sv_dict_10 = {}
r2_dict_10 = {}

for dv, features in MODELS_10.items():
    labels = [ITEM_LABELS[f] for f in features]
    X = pd.DataFrame(iv_df[features].values, columns=labels)
    y = fs_df[dv].values

    gbm = GradientBoostingRegressor(n_estimators=100, max_depth=2,
                                     learning_rate=0.05, subsample=0.8,
                                     min_samples_leaf=8, random_state=42)
    gbm.fit(X, y)
    fitted_10[dv] = gbm

    cv_r2 = cross_val_score(gbm, X, y, cv=5, scoring='r2').mean()
    r2_dict_10[dv] = round(float(cv_r2), 3)
    print(f'  {dv:<22}  CV R² (5-fold) = {cv_r2:.3f}')

    explainer = shap.TreeExplainer(gbm)
    sv_dict_10[dv] = explainer(X)

# ── Importance table ───────────────────────────────────────────────────────
imp_rows = []
for dv, sv in sv_dict_10.items():
    mean_abs = np.abs(sv.values).mean(axis=0)
    for feat, val in zip(sv.feature_names, mean_abs):
        imp_rows.append({'Label': feat, 'Outcome': dv, 'Mean |SHAP|': round(val, 4)})
imp_df = pd.DataFrame(imp_rows)

# ── Cross-outcome grouped bar chart ───────────────────────────────────────
all_labels_ord = (imp_df[imp_df['Outcome'] == 'Beh_Responsible']
                  .sort_values('Mean |SHAP|', ascending=False)['Label'].tolist())

fig, ax = plt.subplots(figsize=(12, 5))
clrs = ['#1565C0', '#2E7D32', '#C62828']
w_bar = 0.26
x     = np.arange(len(all_labels_ord))

for idx, (dv, clr) in enumerate(zip(MODELS_10.keys(), clrs)):
    sub  = imp_df[imp_df['Outcome'] == dv].set_index('Label')
    vals = [sub.loc[lbl, 'Mean |SHAP|'] if lbl in sub.index else 0.0
            for lbl in all_labels_ord]
    ax.bar(x + idx * w_bar, vals, w_bar,
           label=f'{dv}  (CV R²={r2_dict_10[dv]})', color=clr, alpha=0.82)

ax.set_xticks(x + w_bar)
ax.set_xticklabels(all_labels_ord, rotation=32, ha='right', fontsize=9)
ax.set_ylabel('Mean |SHAP value|', fontsize=10)
ax.set_title('SHAP Feature Importance — AI Use Survey  (N = 117)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
bar_path = f'{OUT_FIG}/shap_importance_{DATE}.png'
plt.savefig(bar_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved -> {bar_path}')

# ── Beeswarm plots ─────────────────────────────────────────────────────────
for dv, sv in sv_dict_10.items():
    plt.figure()
    shap.plots.beeswarm(sv, max_display=len(sv.feature_names), show=False)
    plt.title(f'SHAP Beeswarm: {dv}  (CV R²={r2_dict_10[dv]})',
              fontsize=11, pad=14)
    bee_path = f'{OUT_FIG}/shap_beeswarm_{dv}_{DATE}.png'
    plt.savefig(bee_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved -> {bee_path}')

# ── Section 9 regression heatmap (bonus figure) ───────────────────────────
pivot = reg_df.pivot_table(index='Item Label', columns='Outcome',
                            values='β (std)', aggfunc='first')
# Keep canonical order
label_order = [ITEM_LABELS[i] for i in AW_ITEMS + ATT_ITEMS
               if ITEM_LABELS[i] in pivot.index]
pivot = pivot.reindex(label_order)

fig, ax = plt.subplots(figsize=(8, 5))
cmap = plt.cm.RdBu_r
im = ax.imshow(pivot.values.astype(float), cmap=cmap, vmin=-0.4, vmax=0.4,
               aspect='auto')
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, fontsize=9)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=9)

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if np.isnan(val):
            continue
        row = reg_df[(reg_df['Item Label'] == pivot.index[i]) &
                     (reg_df['Outcome'] == pivot.columns[j])]
        sig = row['Sig'].values[0] if len(row) else ''
        ax.text(j, i, f'{val:.2f}{sig}', ha='center', va='center',
                fontsize=8, color='white' if abs(val) > 0.25 else 'black')

plt.colorbar(im, ax=ax, label='β (std)')
ax.set_title('Item-Level Regressions on Outcome Factor Scores  (β standardized)',
             fontsize=10, fontweight='bold', pad=10)
plt.tight_layout()
hm_path = f'{OUT_FIG}/item_regression_heatmap_{DATE}.png'
plt.savefig(hm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved -> {hm_path}')

# ── Save SHAP Excel ────────────────────────────────────────────────────────
shap_xlsx = f'{OUT_TAB}/shap_importance_{DATE}.xlsx'
with pd.ExcelWriter(shap_xlsx, engine='openpyxl') as w:
    imp_df.to_excel(w, sheet_name='Mean_SHAP', index=False)
    pd.DataFrame([{'Outcome': k, 'CV_R2_5fold': v}
                  for k, v in r2_dict_10.items()]).to_excel(
        w, sheet_name='Model_Fit', index=False)
print(f'Saved -> {shap_xlsx}')

# ═══════════════════════════════════════════════════════════════════════════
# BLACK-AND-WHITE VERSIONS
# ═══════════════════════════════════════════════════════════════════════════
print('\n-- B&W figures --')

BW_GRAYS   = ['#111111', '#666666', '#bbbbbb']
BW_HATCHES = ['///', '\\\\\\', 'xxx']

# ── B&W grouped bar chart ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
for idx, (dv, gray, hatch) in enumerate(zip(MODELS_10.keys(), BW_GRAYS, BW_HATCHES)):
    sub  = imp_df[imp_df['Outcome'] == dv].set_index('Label')
    vals = [sub.loc[lbl, 'Mean |SHAP|'] if lbl in sub.index else 0.0
            for lbl in all_labels_ord]
    ax.bar(x + idx * w_bar, vals, w_bar,
           label=f'{dv}  (CV R²={r2_dict_10[dv]})',
           color=gray, hatch=hatch, edgecolor='white', linewidth=0.5)

ax.set_xticks(x + w_bar)
ax.set_xticklabels(all_labels_ord, rotation=32, ha='right', fontsize=9)
ax.set_ylabel('Mean |SHAP value|', fontsize=10)
ax.set_title('SHAP Feature Importance — AI Use Survey  (N = 117)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
bar_bw = f'{OUT_FIG}/shap_importance_{DATE}_bw.png'
plt.savefig(bar_bw, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved -> {bar_bw}')

# ── B&W beeswarm plots ────────────────────────────────────────────────────
gray_cmap = plt.get_cmap('gray_r')
for dv, sv in sv_dict_10.items():
    plt.figure()
    shap.plots.beeswarm(sv, max_display=len(sv.feature_names),
                        color=gray_cmap, show=False)
    plt.title(f'SHAP Beeswarm: {dv}  (CV R²={r2_dict_10[dv]})',
              fontsize=11, pad=14)
    bee_bw = f'{OUT_FIG}/shap_beeswarm_{dv}_{DATE}_bw.png'
    plt.savefig(bee_bw, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved -> {bee_bw}')

# ── B&W regression heatmap ────────────────────────────────────────────────
# Use |β| for shade intensity; sign shown in annotation text
abs_vals = pivot.values.astype(float)
abs_disp = np.abs(np.where(np.isnan(abs_vals), 0, abs_vals))

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(abs_disp, cmap='Greys', vmin=0, vmax=0.45, aspect='auto')

ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, fontsize=9)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=9)

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if np.isnan(val):
            continue
        row = reg_df[(reg_df['Item Label'] == pivot.index[i]) &
                     (reg_df['Outcome'] == pivot.columns[j])]
        sig = row['Sig'].values[0] if len(row) else ''
        color = 'white' if abs(val) > 0.25 else 'black'
        ax.text(j, i, f'{val:+.2f}{sig}', ha='center', va='center',
                fontsize=8, color=color)

plt.colorbar(im, ax=ax, label='|β (std)|')
ax.set_title(
    'Item-Level Regressions on Outcome Factor Scores  (β std; shade = magnitude)',
    fontsize=10, fontweight='bold', pad=10)
plt.tight_layout()
hm_bw = f'{OUT_FIG}/item_regression_heatmap_{DATE}_bw.png'
plt.savefig(hm_bw, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved -> {hm_bw}')

print('\nAll done.')
