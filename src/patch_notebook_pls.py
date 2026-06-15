"""
Patch the SEM notebook to replace CB-SEM (semopy CFA + Bartlett scores)
with PLS-SEM (NIPALS centroid scheme) + bootstrap path CIs.

Run from repo root:
    python src/patch_notebook_pls.py
"""
import json, pathlib

NB_PATH = pathlib.Path('notebooks/Structural Equation Modelling.ipynb')

# ── New cell sources (keyed by cell id) ────────────────────────────────────

CELL_5f68a2b4 = """\
CONSTRUCTS = {
    'Awareness_Worry':   ['aw_error', 'aw_ethics', 'aw_privacy', 'aw_deepfake'],
    'Awareness_Env':     ['aw_env_dc', 'aw_env_use'],
    'Attitude':          ['att_pos', 'att_easy', 'att_skill_worry'],
    'Beh_Environmental': ['beh_env_avoid', 'beh_advise'],
    'Beh_Responsible':   ['beh_resp', 'beh_check', 'beh_harmful', 'beh_prompt'],
}
REVERSE_ITEMS = ['att_skill_worry']

# Inner (structural) model
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

def cronbach_alpha(data):
    arr = np.array(data)
    n = arr.shape[1]
    if n < 2:
        return np.nan
    vt = arr.sum(axis=1).var(ddof=1)
    if vt == 0:
        return np.nan
    return (n / (n - 1)) * (1 - arr.var(axis=0, ddof=1).sum() / vt)

def compute_cr(loadings):
    "Dillon-Goldstein composite reliability from PLS outer loadings."
    sl = np.sum(loadings)
    return sl ** 2 / (sl ** 2 + np.sum(1 - loadings ** 2))

def compute_ave(loadings):
    "Average Variance Extracted from PLS outer loadings."
    return np.mean(loadings ** 2)

def compute_rho_a(loadings, R):
    "Dijkstra-Henseler rho_A."
    p = len(loadings)
    Rv = R.values.copy()
    np.fill_diagonal(Rv, 0)
    r_bar = Rv.sum() / (p * (p - 1))
    ave_lam2 = np.mean(loadings ** 2)
    cr = compute_cr(loadings)
    if ave_lam2 == 0:
        return np.nan
    return float(np.clip(cr * (r_bar / ave_lam2), 0.0, 1.0))

# ── PLS-SEM: NIPALS with centroid inner weighting (all reflective blocks) ──
def pls_nipals(constructs, struct_model, n_iter=300, tol=1e-7):
    # Centroid inner weighting, reflective outer OLS estimation.
    # Returns: lv_scores (DataFrame), outer_weights (dict), outer_loadings (dict).
    n = len(df)
    blocks = {}
    for cname, items in constructs.items():
        blocks[cname] = StandardScaler().fit_transform(prep(items))

    # Initialise: mean of standardised indicators
    lv = {}
    for cname in constructs:
        s = blocks[cname].mean(axis=1)
        std = s.std(ddof=1)
        lv[cname] = (s - s.mean()) / (std if std > 0 else 1.0)

    # Adjacency set for centroid inner step
    adj = {c: set() for c in constructs}
    for outcome, preds in struct_model.items():
        for p in preds:
            adj[outcome].add(p)
            adj[p].add(outcome)

    outer_w = {}
    for it in range(n_iter):
        old = {c: lv[c].copy() for c in constructs}

        # Inner step: centroid weighting
        inner = {}
        for c in constructs:
            s = np.zeros(n)
            for c2 in adj[c]:
                s += np.sign(np.corrcoef(lv[c], lv[c2])[0, 1]) * lv[c2]
            inner[c] = s

        # Outer step: reflective OLS
        for c in constructs:
            X, z = blocks[c], inner[c]
            denom = float(z @ z)
            w = X.T @ z / denom if denom > 0 else np.ones(X.shape[1]) / X.shape[1]
            outer_w[c] = w
            raw = X @ w
            std = raw.std(ddof=1)
            lv[c] = (raw - raw.mean()) / (std if std > 0 else 1.0)

        if sum(np.sum((lv[c] - old[c]) ** 2) for c in constructs) < tol:
            print(f'NIPALS converged in {it + 1} iterations.')
            break

    # Outer loadings: Pearson r(indicator, LV score)
    outer_loadings = {}
    for c, items in constructs.items():
        X, y = blocks[c], lv[c]
        outer_loadings[c] = np.array(
            [np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])]
        )

    return pd.DataFrame({c: lv[c] for c in constructs}), outer_w, outer_loadings

factor_scores, pls_outer_w, pls_outer_loadings = pls_nipals(CONSTRUCTS, STRUCT_MODEL)
print('\\nPLS composite score inter-construct correlations:')
print(factor_scores.corr().round(3).to_string())
"""

CELL_a018d3d8 = """\
cr_rows = []
for cname, items in CONSTRUCTS.items():
    loadings = pls_outer_loadings[cname]
    cr_val  = compute_cr(loadings)
    ave_val = compute_ave(loadings)
    cr_rows.append({
        'Construct':   cname,
        'n_items':     len(items),
        'Loadings':    [round(float(l), 3) for l in loadings],
        'CR':          round(cr_val, 3),
        'AVE':         round(ave_val, 3),
        'CR ≥ 0.70':  cr_val  >= 0.70,
        'AVE ≥ 0.50': ave_val >= 0.50,
    })

cr_df = pd.DataFrame(cr_rows)

print(f'  {"Construct":<22} {"k":>3}  {"CR":>6}  {"AVE":>6}  {"CR>=.70":>7}  {"AVE>=.50":>8}  Loadings')
print('  ' + '-' * 85)
for _, row in cr_df.iterrows():
    print(f'  {row["Construct"]:<22} {row["n_items"]:>3}  {row["CR"]:>6.3f}  '
          f'{row["AVE"]:>6.3f}  {str(row["CR ≥ 0.70"]):>7}  '
          f'{str(row["AVE ≥ 0.50"]):>8}  {row["Loadings"]}')
"""

CELL_c9749847 = """\
rho_rows = []
for cname, items in CONSTRUCTS.items():
    sub = prep(items)
    R = sub.corr()
    loadings = pls_outer_loadings[cname]
    rho = compute_rho_a(loadings, R)
    rho_rows.append({
        'Construct':        cname,
        'n_items':          len(items),
        'rho_A':            round(float(rho), 3),
        'rho_A ≥ 0.70': float(rho) >= 0.70,
        'Note':             'just-identified (2 items)' if len(items) == 2 else '',
    })

rho_df = pd.DataFrame(rho_rows)
print(rho_df.to_string(index=False))
"""

CELL_81f60bd4 = """\
# PLS composite scores (from pls_nipals above)
scores = factor_scores.copy()
"""

CELL_9173d36f = """\
scores = factor_scores.copy()

all_items = sum(CONSTRUCTS.values(), [])
item_matrix = prep(all_items)

scores.describe().round(3)
"""

CELL_2c2a3360 = """\
# PLS-SEM Outer Model Assessment
# Replaces CFA model fit: no chi2/CFI/RMSEA in PLS.
# Reports outer loadings, indicator reliability, block VIF, and SRMR.

print('PLS-SEM Outer Model (Measurement Model)')
print('=' * 68)
print(f'\\n  {"Construct":<22} {"Indicator":<22} {"Loading":>9} {"Rel (λ²)":>9} {"VIF":>7}')
print('  ' + '-' * 72)

outer_model_rows = []
item_to_construct = {}
item_to_loading   = {}

for cname, items in CONSTRUCTS.items():
    sub  = prep(items)
    z    = StandardScaler().fit_transform(sub)
    loadings = pls_outer_loadings[cname]
    for j, item in enumerate(items):
        item_to_construct[item] = cname
        item_to_loading[item]   = loadings[j]
        lam  = loadings[j]
        lam2 = lam ** 2
        # Block VIF: regress indicator on all others in same block
        others = [k for k in range(len(items)) if k != j]
        if others:
            X_v   = np.column_stack([np.ones(len(df)), z[:, others]])
            yv    = z[:, j]
            bv    = np.linalg.lstsq(X_v, yv, rcond=None)[0]
            rss_v = np.sum((yv - X_v @ bv) ** 2)
            tss_v = np.sum((yv - yv.mean()) ** 2)
            r2_v  = 1 - rss_v / tss_v if tss_v > 0 else 0.0
            vif   = 1.0 / max(1.0 - r2_v, 0.001)
        else:
            vif = 1.0
        tag = '' if lam >= 0.70 else '  < 0.70'
        print(f'  {cname:<22} {item:<22} {lam:>9.3f} {lam2:>9.3f} {vif:>7.2f}{tag}')
        outer_model_rows.append({
            'Construct':   cname,
            'Item':        item,
            'Loading':     round(lam,  3),
            'Reliability': round(lam2, 3),
            'VIF':         round(vif,  2),
        })
    print()

outer_model_df = pd.DataFrame(outer_model_rows)
print('  Loading threshold > 0.70 (Hair et al. 2022);  Reliability = λ² > 0.50')

# SRMR: implied vs observed correlation matrix
all_items_ord = sum(CONSTRUCTS.values(), [])
R_obs = prep(all_items_ord).corr()

lv_corr  = factor_scores.corr()
items_n  = len(all_items_ord)
R_impl   = np.eye(items_n)
for i, ii in enumerate(all_items_ord):
    for j, jj in enumerate(all_items_ord):
        if i == j:
            continue
        ci, cj = item_to_construct[ii], item_to_construct[jj]
        li, lj = item_to_loading[ii],   item_to_loading[jj]
        if ci == cj:
            R_impl[i, j] = li * lj
        else:
            R_impl[i, j] = li * lv_corr.loc[ci, cj] * lj

mask = np.triu(np.ones((items_n, items_n), dtype=bool), k=1)
srmr = float(np.sqrt(np.mean((R_obs.values[mask] - R_impl[mask]) ** 2)))
print(f'\\nSRMR = {srmr:.4f}  (< 0.08 acceptable;  < 0.05 good fit)')

pls_fit_df = pd.DataFrame([{
    'Model':   'PLS-SEM NIPALS',
    'Scheme':  'Centroid inner weighting',
    'SRMR':    round(srmr, 4),
    'N':       len(df),
}])
sem_fit_df = pls_fit_df  # alias for downstream save cell

outer_model_df.to_csv(f'../output/tables/pls_outer_model_{DATE}.csv', index=False)
pls_fit_df.to_csv(f'../output/stats/pls_fit_{DATE}.csv', index=False)
print(f'\\nSaved -> ../output/tables/pls_outer_model_{DATE}.csv')
print(f'Saved -> ../output/stats/pls_fit_{DATE}.csv')
"""

CELL_9d2d4206 = """\
from scipy.stats import t as t_dist

# ── OLS structural paths on PLS composite scores ───────────────────────────
def ols_path(y_name, x_names, data):
    y = data[y_name].values
    X = data[x_names].values
    n, k = X.shape
    X_aug  = np.column_stack([np.ones(n), X])
    beta   = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    y_pred = X_aug @ beta
    rss    = np.sum((y - y_pred) ** 2)
    tss    = np.sum((y - y.mean()) ** 2)
    r2     = 1 - rss / tss
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    se2    = rss / (n - k - 1)
    cov_b  = se2 * np.linalg.inv(X_aug.T @ X_aug)
    sy     = y.std(ddof=1)
    rows   = []
    for i, xname in enumerate(x_names):
        b    = beta[i + 1]
        se_b = np.sqrt(cov_b[i + 1, i + 1])
        t_v  = b / se_b
        p_v  = 2 * t_dist.sf(abs(t_v), df=n - k - 1)
        rows.append({
            'Outcome':   y_name,
            'Predictor': xname,
            'β (std)':   round(b * data[xname].std(ddof=1) / sy, 3),
            'B (unstd)': round(b, 3),
            'SE':        round(se_b, 3),
            't':         round(t_v, 3),
            'p':         round(p_v, 4),
            'R²':       round(r2, 3),
            'Adj.R²':   round(adj_r2, 3),
        })
    return rows

STRUCT_PATHS = [
    ('Attitude',          ['Awareness_Worry', 'Awareness_Env']),
    ('Beh_Environmental', ['Attitude']),
    ('Beh_Responsible',   ['Attitude']),
]

all_path_rows = []
for y_name, x_names in STRUCT_PATHS:
    all_path_rows.extend(ols_path(y_name, x_names, factor_scores))
sem_path_df = pd.DataFrame(all_path_rows)

# ── Bootstrap path CIs (B=1000, percentile, seed=42) ──────────────────────
N_BOOT  = 1000
np.random.seed(42)
n_obs   = len(factor_scores)
fs_arr  = factor_scores.values
fs_cols = list(factor_scores.columns)
boot_dist = {(r['Predictor'], r['Outcome']): [] for _, r in sem_path_df.iterrows()}

for _ in range(N_BOOT):
    idx  = np.random.choice(n_obs, n_obs, replace=True)
    samp = pd.DataFrame(fs_arr[idx], columns=fs_cols)
    for y_name, x_names in STRUCT_PATHS:
        for r in ols_path(y_name, x_names, samp):
            key = (r['Predictor'], r['Outcome'])
            if key in boot_dist:
                boot_dist[key].append(r['β (std)'])

ci_lo, ci_hi, p_boot_list = [], [], []
for _, row in sem_path_df.iterrows():
    dist = np.array(boot_dist[(row['Predictor'], row['Outcome'])])
    lo, hi = np.percentile(dist, [2.5, 97.5])
    p_b = 2 * min(np.mean(dist >= 0), np.mean(dist <= 0))
    ci_lo.append(round(lo, 3))
    ci_hi.append(round(hi, 3))
    p_boot_list.append(round(max(p_b, 1 / N_BOOT), 4))

sem_path_df['CI 2.5%']  = ci_lo
sem_path_df['CI 97.5%'] = ci_hi
sem_path_df['p (boot)'] = p_boot_list

# ── Display ────────────────────────────────────────────────────────────────
def sig_tag(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''

print('PLS-SEM Structural Paths  (OLS on PLS composite scores + bootstrap B=1000)')
hdr = (f'  {"Predictor":<22} {"Outcome":<22} {"β":>7} {"B":>7} {"SE":>6} '
       f'{"t":>7} {"p(OLS)":>7} {"R²":>6}  {"Boot 95% CI":<17}  {"p(boot)":>7}')
print(hdr)
print('  ' + '-' * 110)
prev_y = None
for _, row in sem_path_df.iterrows():
    if row['Outcome'] != prev_y:
        if prev_y is not None:
            print()
        prev_y = row['Outcome']
    ci_str = f'[{row["CI 2.5%"]:>6.3f}, {row["CI 97.5%"]:>6.3f}]'
    sig = sig_tag(row['p (boot)'])
    print(f'  {row["Predictor"]:<22} {row["Outcome"]:<22} '
          f'{row["β (std)"]:>7.3f} {row["B (unstd)"]:>7.3f} {row["SE"]:>6.3f} '
          f'{row["t"]:>7.3f} {row["p"]:>7.4f} {row["R²"]:>6.3f}  {ci_str}  '
          f'{row["p (boot)"]:>7.4f}  {sig}')

print('\\n  β = standardised;  Boot CI = percentile 95% (B=1000);  * p<.05  ** p<.01  *** p<.001')

print('\\nEndogenous LV fit:')
for y_name, x_names in STRUCT_PATHS:
    rows_y = [r for _, r in sem_path_df.iterrows() if r['Outcome'] == y_name]
    r2 = rows_y[0]['R²'] if rows_y else 0
    f2 = r2 / (1 - r2) if r2 < 1 else float('inf')
    print(f'  {y_name:<22}  R² = {r2:.3f}  f² = {f2:.3f}')
"""

CELL_sem_diagram = """\
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

FIG_W, FIG_H = 14, 7
BW, BH = 2.7, 1.0

NODES = {
    'Awareness\\n(Worry)':      (2.2, 5.5),
    'Awareness\\n(Env)':        (2.2, 1.5),
    'Attitude':                (7.0, 3.5),
    'Beh\\nEnvironmental':      (11.8, 5.5),
    'Beh\\nResponsible':        (11.8, 1.5),
}

NAME_MAP = {
    'Awareness_Worry':   'Awareness\\n(Worry)',
    'Awareness_Env':     'Awareness\\n(Env)',
    'Attitude':          'Attitude',
    'Beh_Environmental': 'Beh\\nEnvironmental',
    'Beh_Responsible':   'Beh\\nResponsible',
}

# Dynamic: read from sem_path_df (uses bootstrap p)
PATHS = [
    (NAME_MAP[r['Predictor']], NAME_MAP[r['Outcome']],
     r['β (std)'], r['p (boot)'])
    for _, r in sem_path_df.iterrows()
]

R2_VALS = {}
seen = set()
for _, r in sem_path_df.iterrows():
    key = NAME_MAP[r['Outcome']]
    if key not in seen:
        R2_VALS[key] = r['R²']
        seen.add(key)

COLORS = {
    'Awareness\\n(Worry)': ('#DDEEFF', '#1565C0'),
    'Awareness\\n(Env)':   ('#DDEEFF', '#1565C0'),
    'Attitude':           ('#D6EAD6', '#2E7D32'),
    'Beh\\nEnvironmental': ('#FFF0DB', '#E65100'),
    'Beh\\nResponsible':   ('#FFF0DB', '#E65100'),
}

def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.10:  return '†'
    return 'ns'

def box_edge_pt(cx, cy, tx, ty, hw, hh):
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx, cy
    t_x = hw / abs(dx) if dx != 0 else float('inf')
    t_y = hh / abs(dy) if dy != 0 else float('inf')
    return cx + dx * min(t_x, t_y), cy + dy * min(t_x, t_y)

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

for src, dst, beta, p_val in PATHS:
    sx, sy = NODES[src]
    dx, dy = NODES[dst]
    x0, y0 = box_edge_pt(sx, sy, dx, dy, BW/2, BH/2)
    x1, y1 = box_edge_pt(dx, dy, sx, sy, BW/2, BH/2)
    is_sig = p_val < 0.05
    color  = '#1A1A1A' if is_sig else '#AAAAAA'
    lw     = 2.3 if is_sig else 1.7
    lstyle = 'solid' if is_sig else (0, (5, 3))
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                               linestyle=lstyle, mutation_scale=20,
                               connectionstyle='arc3,rad=0.0'), zorder=2)
    mid_x  = (x0 + x1) / 2
    mid_y  = (y0 + y1) / 2
    angle  = np.arctan2(y1 - y0, x1 - x0)
    off    = 0.46
    lx = mid_x - np.sin(angle) * off
    ly = mid_y + np.cos(angle) * off
    sign  = '+' if beta >= 0 else ''
    label = 'β = {}{:.3f}{}'.format(sign, beta, sig_stars(p_val))
    ax.text(lx, ly, label, ha='center', va='center', fontsize=10.5,
            color='#111111' if is_sig else '#888888', zorder=5,
            bbox=dict(boxstyle='round,pad=0.28', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.92, linewidth=0.8))

for name, (cx, cy) in NODES.items():
    fc, ec = COLORS[name]
    ax.add_patch(FancyBboxPatch((cx - BW/2, cy - BH/2), BW, BH,
                               boxstyle='round,pad=0.10',
                               facecolor=fc, edgecolor=ec, linewidth=2.4, zorder=3))
    lines = name.split('\\n')
    if len(lines) == 1:
        ax.text(cx, cy, lines[0], ha='center', va='center',
                fontsize=11.5, fontweight='bold', zorder=4)
    else:
        ax.text(cx, cy + 0.19, lines[0], ha='center', va='center',
                fontsize=11.5, fontweight='bold', zorder=4)
        ax.text(cx, cy - 0.19, lines[1], ha='center', va='center',
                fontsize=11.5, fontweight='bold', zorder=4)
    if name in R2_VALS:
        ax.text(cx, cy - BH/2 - 0.28, 'R² = {:.3f}'.format(R2_VALS[name]),
                ha='center', va='top', fontsize=9.5,
                color='#444444', style='italic', zorder=4)

legend_handles = [
    mpatches.Patch(facecolor='#DDEEFF', edgecolor='#1565C0', lw=1.5,
                   label='Predictors (Awareness)'),
    mpatches.Patch(facecolor='#D6EAD6', edgecolor='#2E7D32', lw=1.5,
                   label='Mediator (Attitude)'),
    mpatches.Patch(facecolor='#FFF0DB', edgecolor='#E65100', lw=1.5,
                   label='Outcomes (Behavior)'),
    mpatches.Patch(facecolor='none', edgecolor='none', label=''),
    plt.Line2D([0],[0], color='#1A1A1A', lw=2.3, label='Significant (p < .05)'),
    plt.Line2D([0],[0], color='#AAAAAA', lw=1.7, linestyle=(0,(5,3)),
               label='Marginal (p < .10)'),
]
ax.legend(handles=legend_handles, loc='lower center',
          bbox_to_anchor=(0.5, -0.01), ncol=6,
          fontsize=9, framealpha=0.95, edgecolor='#CCCCCC')

ax.text(0.995, 0.995,
        '*** p < .001   ** p < .01   * p < .05   † p < .10',
        transform=ax.transAxes, ha='right', va='top', fontsize=9, color='#555555',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#DDDDDD', alpha=0.92))

ax.text(0.5, 1.025,
        'PLS-SEM — AI Use Survey (N = 117)',
        transform=ax.transAxes, ha='center', va='bottom',
        fontsize=13, fontweight='bold')
ax.text(0.5, 0.997,
        'NIPALS centroid scheme • OLS paths + bootstrap 95% CI (B=1000)',
        transform=ax.transAxes, ha='center', va='bottom',
        fontsize=8.8, color='#666666')

plt.tight_layout(pad=0.6)
out = '../output/figures/sem_diagram_{}.png'.format(DATE)
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='#FAFAFA')
plt.show()
print('Saved ->', out)
"""

UPDATES = {
    '5f68a2b4':       CELL_5f68a2b4,
    'a018d3d8':       CELL_a018d3d8,
    'c9749847':       CELL_c9749847,
    '81f60bd4':       CELL_81f60bd4,
    '9173d36f':       CELL_9173d36f,
    '2c2a3360':       CELL_2c2a3360,
    '9d2d4206':       CELL_9d2d4206,
    'sem_diagram_code': CELL_sem_diagram,
}

# ── Patch the notebook ──────────────────────────────────────────────────────
def src_to_lines(s):
    """Convert a raw string to a list of notebook source lines."""
    lines = s.split('\n')
    result = [line + '\n' for line in lines[:-1]]
    if lines[-1]:
        result.append(lines[-1])
    return result

with open(NB_PATH, encoding='utf-8-sig') as f:
    nb = json.load(f)

patched = []
for cell in nb['cells']:
    cid = cell.get('id', '')
    if cid in UPDATES:
        cell['source'] = src_to_lines(UPDATES[cid])
        cell['outputs'] = []
        cell['execution_count'] = None
        patched.append(cid)

with open(NB_PATH, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Patched {len(patched)} cells: {patched}')
