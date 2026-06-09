"""
Section 9: Item-Level Simple Regressions (standalone)
Run from repo root:  python src/run_section9.py
Outputs: output/tables/item_regressions_YYYYMMDD.xlsx
         output/tables/item_regressions_YYYYMMDD.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import t as t_dist
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

DATE = datetime.today().strftime('%Y%m%d')

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_excel('src/raw/wbg_ai.xlsx')

# ── Construct definitions (same as notebook) ───────────────────────────────
CONSTRUCTS = {
    'Awareness_Worry':   ['aw_error', 'aw_ethics', 'aw_privacy', 'aw_deepfake'],
    'Awareness_Env':     ['aw_env_dc', 'aw_env_use'],
    'Attitude':          ['att_pos', 'att_easy', 'att_skill_worry'],
    'Beh_Environmental': ['beh_env_avoid', 'beh_advise'],
    'Beh_Responsible':   ['beh_resp', 'beh_check', 'beh_harmful', 'beh_prompt'],
}
REVERSE_ITEMS = ['att_skill_worry']

def prep(items):
    sub = df[items].copy()
    for c in REVERSE_ITEMS:
        if c in sub.columns:
            sub[c] = 6 - sub[c]
    return sub

# ── Factor loading helper ──────────────────────────────────────────────────
def get_loadings(data):
    if data.shape[1] < 2:
        return np.array([1.0])
    fa = FactorAnalysis(n_components=1, random_state=0)
    fa.fit(StandardScaler().fit_transform(data))
    lam = fa.components_[0]
    # flip sign so majority of loadings are positive
    if np.sum(lam < 0) > np.sum(lam >= 0):
        lam = -lam
    return lam

# ── Bartlett factor scores ─────────────────────────────────────────────────
def bartlett_score(items):
    sub = prep(items)
    z   = StandardScaler().fit_transform(sub)
    lam = np.abs(get_loadings(pd.DataFrame(z, columns=sub.columns)))
    psi = np.maximum(1 - lam ** 2, 0.05)
    w   = lam / psi
    return z @ w / (lam @ w)

factor_scores = pd.DataFrame({
    cname: bartlett_score(items)
    for cname, items in CONSTRUCTS.items()
})

# ── OLS path helper (same as notebook) ────────────────────────────────────
def ols_path(y_name, x_names, data):
    y = data[y_name].values
    X = data[x_names].values
    n, k = X.shape
    X_aug = np.column_stack([np.ones(n), X])
    beta  = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    y_pred = X_aug @ beta
    rss = np.sum((y - y_pred) ** 2)
    tss = np.sum((y - y.mean()) ** 2)
    r2     = 1 - rss / tss
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    se2    = rss / (n - k - 1)
    cov_beta = se2 * np.linalg.inv(X_aug.T @ X_aug)
    sy = y.std(ddof=1)
    rows = []
    for i, xname in enumerate(x_names):
        b    = beta[i + 1]
        se_b = np.sqrt(cov_beta[i + 1, i + 1])
        t_val = b / se_b
        p_val = 2 * t_dist.sf(abs(t_val), df=n - k - 1)
        beta_std = b * data[xname].std(ddof=1) / sy
        rows.append({
            'Outcome':   y_name,
            'Predictor': xname,
            'β (std)':   round(beta_std, 3),
            'B (unstd)': round(b, 3),
            'SE':        round(se_b, 3),
            't':         round(t_val, 3),
            'p':         round(p_val, 4),
            'R²':        round(r2, 3),
            'Adj.R²':    round(adj_r2, 3),
        })
    return rows

# ── Section 9 regressions ──────────────────────────────────────────────────
AW_ITEMS  = ['aw_error', 'aw_ethics', 'aw_privacy', 'aw_deepfake',
             'aw_env_dc', 'aw_env_use']
ATT_ITEMS = ['att_pos', 'att_easy', 'att_skill_worry']

ITEM_LABELS_9 = {
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

beta_key = 'β (std)'
r2_key   = 'R²'

def sig_tag(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''

all_rows = []

HDR = ('  {:<22} {:<22} {:>6} {:>7} {:>7} {:>7} {:>7} {:>6}'.format(
       'Item', 'DV', 'beta', 'B', 'SE', 't', 'p', 'R2'))
SEP = '  ' + '-' * 82

print('\n9a. Awareness items -> Attitude / Beh_Environmental / Beh_Responsible')
print(HDR)
print(SEP)
for item in AW_ITEMS:
    for dv in ['Attitude', 'Beh_Environmental', 'Beh_Responsible']:
        r = ols_path(dv, [item], reg_data)[0]
        all_rows.append(r)
        print('  {:<22} {:<22} {:>6.3f} {:>7.3f} {:>7.3f} {:>7.3f} {:>7.4f} {:>6.3f}  {}'.format(
            r['Predictor'], r['Outcome'], r[beta_key], r['B (unstd)'],
            r['SE'], r['t'], r['p'], r[r2_key], sig_tag(r['p'])))
    print()

print('9b. Attitude items -> Beh_Environmental / Beh_Responsible')
print(HDR)
print(SEP)
for item in ATT_ITEMS:
    for dv in ['Beh_Environmental', 'Beh_Responsible']:
        r = ols_path(dv, [item], reg_data)[0]
        all_rows.append(r)
        print('  {:<22} {:<22} {:>6.3f} {:>7.3f} {:>7.3f} {:>7.3f} {:>7.4f} {:>6.3f}  {}'.format(
            r['Predictor'], r['Outcome'], r[beta_key], r['B (unstd)'],
            r['SE'], r['t'], r['p'], r[r2_key], sig_tag(r['p'])))
    print()

print('* p<.05  ** p<.01  *** p<.001')

# ── Save outputs ──────────────────────────────────────────────────────────
item_reg_df = pd.DataFrame(all_rows)
item_reg_df['Item Label'] = item_reg_df['Predictor'].map(ITEM_LABELS_9)
item_reg_df['Sig'] = item_reg_df['p'].apply(sig_tag)
cols_out = ['Predictor', 'Item Label', 'Outcome', beta_key, 'B (unstd)', 'SE', 't', 'p', r2_key, 'Adj.R²', 'Sig']
item_reg_df = item_reg_df[cols_out]

csv_path  = f'output/tables/item_regressions_{DATE}.csv'
xlsx_path = f'output/tables/item_regressions_{DATE}.xlsx'

item_reg_df.to_csv(csv_path, index=False)

with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
    aw_mask = item_reg_df['Predictor'].isin(AW_ITEMS)
    item_reg_df[aw_mask].to_excel(writer, sheet_name='9a_Awareness', index=False)
    item_reg_df[~aw_mask].to_excel(writer, sheet_name='9b_Attitude', index=False)
    item_reg_df.to_excel(writer, sheet_name='All', index=False)

print(f'\nSaved CSV  -> {csv_path}')
print(f'Saved XLSX -> {xlsx_path}')
