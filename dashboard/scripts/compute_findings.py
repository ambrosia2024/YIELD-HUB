#!/usr/bin/env python3
"""
compute_findings.py — Compute all benchmarking findings experiment results
and save to CSV files in dashboard/data/.

Outputs:
  findings_summary.csv       — A1: model summary + baseline beats
  findings_anova.csv         — A2: ANOVA variance decomposition
  findings_rankstab.csv      — A3: rank stability (rank per country per model)
  findings_lpjml.csv         — A4: LPJmL advantage over tabpfn
  findings_wins.csv          — A5: who wins most often per country
  findings_admin.csv         — A6: admin level accuracy breakdown
  findings_koppen.csv        — A7: Koppen climate zone accuracy

Run before gen_findings_tab.py.
"""
import pandas as pd
import numpy as np
import json
import statistics
from pathlib import Path
from collections import defaultdict

ROOT     = Path('/gpfs/home4/vsaxena1/AgML-CY-Bench')
YHUB     = ROOT / 'YIELD-HUB'
OUT      = YHUB / 'dashboard' / 'data'
OUT.mkdir(parents=True, exist_ok=True)

ARCH_CSV   = YHUB / 'results' / 'benchmarkArchitectureResults.csv'
RES_CSV    = YHUB / 'results' / 'benchmarkResolutionResults.csv'
ADMIN_JSON = YHUB / 'dashboard' / '_extra' / 'admin_level.json'
KOPPEN_CSV = ROOT / 'cybench' / 'data' / 'koppen_labels' / 'koppen_labels.csv'

SK3  = 'skill_rmse_vs_rolling3'
SKLM = 'skill_rmse_vs_climatology'
SKOL = 'skill_rmse_vs_ols'

arch = pd.read_csv(ARCH_CSV)
res  = pd.read_csv(RES_CSV)

# ── build unified long table ──────────────────────────────────────────────────
# CNN-LSTM only from ARCH (not in RES)
cnn = arch[arch['model_name'] == 'CNN-LSTM'].copy()
cnn_rows = cnn[['crop','country','nrmse',SK3,SKLM,SKOL]].copy()
cnn_rows['model'] = 'CNN-LSTM'
cnn_rows['resolution'] = 'daily'
cnn_rows['strategy'] = 'default'
cnn_rows['combo'] = 'CNN-LSTM.daily'
cnn_rows['family'] = 'cnn'

# Everything else from RES
def _combo(r):
    s = f"{r['model_name']}.{r['resolution']}"
    if r.get('strategy','') == 'finetuned':
        s += '.finetuned'
    return s

def _family(mn):
    if mn in ('tabpfn','tabicl','tabdpt'): return 'tabular_fm'
    if mn in ('patchtst','tst','tsmixer','timesnet','timexer',
              'informer','itransformer','autoformer'): return 'transformer'
    if mn in ('xlinear','olinear','nlinear','rlinear','dlinear'): return 'linear'
    if 'lpjml' in mn: return 'process'
    if mn == 'mitra': return 'agent'
    return 'other'

res_rows = res[['crop','country','model_name','resolution','strategy',
                'nrmse',SK3,SKLM,SKOL]].copy()
res_rows['model'] = res_rows['model_name']
res_rows['combo'] = res_rows.apply(_combo, axis=1)
res_rows['family'] = res_rows['model_name'].apply(_family)
res_rows = res_rows.drop(columns=['model_name'])

df = pd.concat([cnn_rows, res_rows], ignore_index=True)
df[SK3]  = pd.to_numeric(df[SK3],  errors='coerce')
df[SKLM] = pd.to_numeric(df[SKLM], errors='coerce')
df[SKOL] = pd.to_numeric(df[SKOL], errors='coerce')
df['nrmse'] = pd.to_numeric(df['nrmse'], errors='coerce')

PROC = {'lpjml_rainfed','lpjml_irrigated'}
SKIP = {'mitra'}   # exclude agent-based from ranking

print(f"Loaded {len(df)} rows, {df['combo'].nunique()} combos, {df['country'].nunique()} countries")

# ── A1: Model summary + baseline beats ───────────────────────────────────────
# Friedman rank per country (exclude process + mitra)
def friedman_ranks(sub):
    """Average rank (1=best, higher=worse) over countries on SK3 desc."""
    sub = sub.dropna(subset=[SK3])
    sub = sub[~sub['model'].str.lower().isin(PROC | SKIP)]
    ranked = sub.copy()
    ranked['rk'] = ranked.groupby('country')[SK3].rank(
        method='average', ascending=False)
    return ranked.groupby('combo')['rk'].mean().round(2)

# Canonical 5 family leaders (one per family)
# Best resolution per model selected by skill within compute_findings
FAMILY_LEADERS = {
    'CNN-LSTM.daily':          ('CNN-LSTM',       'cnn'),
    'patchtst.daily':          ('PatchTST',        'transformer'),
    'xlinear.daily':           ('XLinear',         'linear'),
    'tabpfn.dekad.finetuned':  ('TabPFN (TBT)',    'tabular_fm'),
    'lpjml_rainfed.daily':     ('LPJmL rainfed',   'process'),
}

rows_summary = []
for crop in ['maize', 'wheat', 'both']:
    sub = df if crop == 'both' else df[df['crop'] == crop]
    # Global rank: exclude mitra, include all others including LPJmL
    fr_global = friedman_ranks(sub)

    # Rep-rank: Friedman rank WITHIN the 5 family leaders (including LPJmL)
    sub_rep = sub[sub['combo'].isin(FAMILY_LEADERS.keys())]
    sub_rep_ranked = sub_rep.dropna(subset=[SK3]).copy()
    sub_rep_ranked['rk_rep'] = sub_rep_ranked.groupby(['country','crop'])[SK3].rank(
        method='min', ascending=False)  # method='min' avoids tie artifacts
    fr_rep = sub_rep_ranked.groupby('combo')['rk_rep'].mean().round(2).to_dict()

    for combo, grp in sub.groupby('combo'):
        m = grp['model'].iloc[0]
        res_v = grp['resolution'].iloc[0] if 'resolution' in grp else 'daily'
        strat = grp['strategy'].iloc[0] if 'strategy' in grp else 'default'
        fam   = grp['family'].iloc[0]
        n     = grp['country'].nunique()
        nrmse_vals = grp['nrmse'].dropna()
        sk3   = grp[SK3].dropna()
        sklm  = grp[SKLM].dropna()
        skol  = grp[SKOL].dropna()
        rows_summary.append({
            'crop': crop,
            'combo': combo,
            'model': m,
            'resolution': res_v,
            'strategy': strat,
            'family': fam,
            # For 'both' crops, n_tasks = total country-crop pairs (not unique countries)
            'n_tasks': len(grp),
            'nrmse_med': round(float(nrmse_vals.median()), 4) if len(nrmse_vals) else None,
            'nrmse_min': round(float(nrmse_vals.min()),    4) if len(nrmse_vals) else None,
            'nrmse_max': round(float(nrmse_vals.max()),    4) if len(nrmse_vals) else None,
            'beats_clim_n':    int((sklm > 0).sum()),
            'beats_ols_n':     int((skol > 0).sum()),
            'beats_rolling3_n':int((sk3  > 0).sum()),
            'beats_clim_pct':    round(float((sklm > 0).mean() * 100), 1) if len(sklm) else None,
            'beats_ols_pct':     round(float((skol > 0).mean() * 100), 1) if len(skol) else None,
            'beats_rolling3_pct':round(float((sk3  > 0).mean() * 100), 1) if len(sk3)  else None,
            'skill_roll3_med': round(float(sk3.median()), 4) if len(sk3) else None,
            'friedman_rank': float(fr_global.get(combo, np.nan)),
            'is_rep': combo in FAMILY_LEADERS,
            'rep_rank': float(fr_rep.get(combo, np.nan)) if combo in FAMILY_LEADERS else None,
        })

summary_df = pd.DataFrame(rows_summary)
summary_df.to_csv(OUT / 'findings_summary.csv', index=False)
print(f"findings_summary.csv: {len(summary_df)} rows")

# ── A2: ANOVA variance decomposition ─────────────────────────────────────────
# Use log(nrmse) ~ country + crop + model; compute SS for each factor
anova_rows = []
for crop in ['maize', 'wheat', 'both']:
    sub = df if crop == 'both' else df[df['crop'] == crop]
    sub = sub.dropna(subset=['nrmse','combo','country'])
    sub = sub[~sub['model'].str.lower().isin(SKIP)]
    sub = sub.copy()
    sub['log_nrmse'] = np.log(sub['nrmse'].clip(lower=1e-6))
    grand_mean = sub['log_nrmse'].mean()
    SS_total = ((sub['log_nrmse'] - grand_mean) ** 2).sum()
    if SS_total == 0:
        continue
    # Country SS
    country_means = sub.groupby('country')['log_nrmse'].mean()
    SS_country = sum(
        len(sub[sub['country'] == c]) * (m - grand_mean)**2
        for c, m in country_means.items()
    )
    # Crop SS (only meaningful for 'both')
    if crop == 'both':
        crop_means = sub.groupby('crop')['log_nrmse'].mean()
        SS_crop = sum(
            len(sub[sub['crop'] == c]) * (m - grand_mean)**2
            for c, m in crop_means.items()
        )
    else:
        SS_crop = 0.0
    # Model SS
    model_means = sub.groupby('combo')['log_nrmse'].mean()
    SS_model = sum(
        len(sub[sub['combo'] == c]) * (m - grand_mean)**2
        for c, m in model_means.items()
    )
    SS_resid = SS_total - SS_country - SS_crop - SS_model
    for factor, ss in [('country', SS_country), ('crop', SS_crop),
                        ('model', SS_model), ('residual', max(0, SS_resid))]:
        anova_rows.append({
            'crop': crop, 'factor': factor,
            'ss': round(ss, 4),
            'pct_variance': round(ss / SS_total * 100, 2)
        })

anova_df = pd.DataFrame(anova_rows)
anova_df.to_csv(OUT / 'findings_anova.csv', index=False)
print(f"findings_anova.csv: {len(anova_df)} rows")

# ── A3: Rank stability ────────────────────────────────────────────────────────
# Rank each combo per country, exclude process + mitra
rank_rows = []
for crop in ['maize', 'wheat', 'both']:
    sub = df if crop == 'both' else df[df['crop'] == crop]
    sub = sub.dropna(subset=[SK3])
    sub = sub[~sub['model'].str.lower().isin(PROC | SKIP)]
    ranked = sub.copy()
    ranked['rank'] = ranked.groupby('country')[SK3].rank(
        method='average', ascending=False)
    for _, r in ranked.iterrows():
        rank_rows.append({
            'crop': crop, 'country': r['country'],
            'combo': r['combo'], 'model': r['model'],
            'resolution': r['resolution'],
            'skill_roll3': round(float(r[SK3]), 4),
            'rank': round(float(r['rank']), 2),
        })

rankstab_df = pd.DataFrame(rank_rows)
rankstab_df.to_csv(OUT / 'findings_rankstab.csv', index=False)
print(f"findings_rankstab.csv: {len(rankstab_df)} rows")

# ── A4: LPJmL vs TabPFN ──────────────────────────────────────────────────────
# LPJmL rainfed vs tabpfn.dekad (inference) and tabpfn.dekad.finetuned
lpjml_rows = []
lpjml_df = df[df['combo'] == 'lpjml_rainfed.daily']
for comparator in ['tabpfn.dekad', 'tabpfn.dekad.finetuned']:
    comp_df = df[df['combo'] == comparator]
    for crop in ['maize', 'wheat', 'both']:
        lsub = lpjml_df if crop == 'both' else lpjml_df[lpjml_df['crop'] == crop]
        csub = comp_df if crop == 'both' else comp_df[comp_df['crop'] == crop]
        merged = lsub[['country','crop',SK3]].merge(
            csub[['country','crop',SK3]], on=['country','crop'],
            suffixes=('_lpjml','_comp'))
        for _, r in merged.iterrows():
            lpjml_rows.append({
                'crop': crop,
                'country': r['country'],
                'comparator': comparator,
                'skill_lpjml': round(float(r[f'{SK3}_lpjml']), 4),
                'skill_comp':  round(float(r[f'{SK3}_comp']),  4),
                'lpjml_advantage': round(float(r[f'{SK3}_lpjml'] - r[f'{SK3}_comp']), 4),
                'lpjml_wins': int(r[f'{SK3}_lpjml'] > r[f'{SK3}_comp']),
            })

lpjml_out = pd.DataFrame(lpjml_rows)
lpjml_out.to_csv(OUT / 'findings_lpjml.csv', index=False)
print(f"findings_lpjml.csv: {len(lpjml_out)} rows")

# ── A5: Who wins most often ───────────────────────────────────────────────────
# Per country (best combo by SK3, excluding process + mitra)
wins_rows = []
for crop in ['maize', 'wheat', 'both']:
    sub = df if crop == 'both' else df[df['crop'] == crop]
    sub = sub.dropna(subset=[SK3])
    sub = sub[~sub['model'].str.lower().isin(PROC | SKIP)]
    best = sub.loc[sub.groupby('country')[SK3].idxmax()]
    for _, r in best.iterrows():
        wins_rows.append({
            'crop': crop,
            'country': r['country'],
            'best_combo': r['combo'],
            'best_model': r['model'],
            'best_resolution': r['resolution'],
            'best_skill_roll3': round(float(r[SK3]), 4),
        })

wins_df = pd.DataFrame(wins_rows)
# Add win counts per combo per crop
win_counts = wins_df.groupby(['crop','best_combo']).size().reset_index(name='n_wins')
wins_df = wins_df.merge(win_counts, on=['crop','best_combo'], how='left')
wins_df.to_csv(OUT / 'findings_wins.csv', index=False)
print(f"findings_wins.csv: {len(wins_df)} rows")

# ── A6: Admin level ───────────────────────────────────────────────────────────
cl = json.loads(ADMIN_JSON.read_text()).get('country_level', {})
REP_COMBOS = {
    'CNN-LSTM':  'CNN-LSTM.daily',
    'PatchTST':  'patchtst.daily',
    'XLinear':   'xlinear.daily',
    'TabPFN':    'tabpfn.dekad.finetuned',
    'LPJmL':     'lpjml_rainfed.daily',
}
admin_rows = []
for lv_key, lv_name in [(1,'NUTS-1'),(2,'NUTS-2'),(3,'NUTS-3'),('others','Others')]:
    if lv_key == 'others':
        countries = [c for c,v in cl.items() if v is None]
    else:
        countries = [c for c,v in cl.items() if v == lv_key]
    for crop in ['maize','wheat','both']:
        sub = df if crop == 'both' else df[df['crop'] == crop]
        sub = sub[sub['country'].isin(countries)]
        for label, combo in REP_COMBOS.items():
            g = sub[sub['combo'] == combo].dropna(subset=['nrmse', SK3])
            if g.empty: continue
            admin_rows.append({
                'admin_level': lv_key,
                'admin_name': lv_name,
                'crop': crop,
                'model_label': label,
                'combo': combo,
                'countries': ','.join(sorted(g['country'].unique())),
                'n_tasks': int(g['country'].nunique()),
                'nrmse_med': round(float(g['nrmse'].median()), 4),
                'win_rate': round(float((g[SK3] > 0).mean() * 100)),
            })

admin_df = pd.DataFrame(admin_rows)
admin_df.to_csv(OUT / 'findings_admin.csv', index=False)
print(f"findings_admin.csv: {len(admin_df)} rows")

# ── A7: Koppen climate zones ──────────────────────────────────────────────────
kp = pd.read_csv(KOPPEN_CSV)
kp_abcde = kp[kp['koppen_main_group'].isin(list('ABCDE'))]
kop_country = (kp_abcde.groupby(['crop','country'])['koppen_main_group']
               .agg(lambda s: s.value_counts().idxmax()).to_dict())
MULTI = {'AR','AU','BR','CN','IN','US'}
kop_country = {k: ('M' if k[1] in MULTI else v) for k,v in kop_country.items()}
for c in MULTI:
    for cr in ['maize','wheat']:
        kop_country.setdefault((cr,c), 'M')

ZONE_ASSIGN = {
    'A': (['AO','BF','ML','MZ','SN','TD'], 'Tropical (A)'),
    'B': (['EL','ES','ET','LS','NE','ZA','ZM'], 'Arid/Semi-arid (B)'),
    'C': (['AT','BE','BG','DE','DK','FR','HR','HU','IT','NL','PT','RO','SK'], 'Temperate (C)'),
    'D': (['CZ','EE','FI','LT','LV','PL','SE'], 'Continental/Boreal (D)'),
    'M': (['AR','AU','BR','CN','IN','US'], 'Mixed/Multi-zone (M)'),
}

koppen_rows = []
for zone, (zone_cs, zone_name) in ZONE_ASSIGN.items():
    for crop in ['maize','wheat','both']:
        sub = df if crop == 'both' else df[df['crop'] == crop]
        present = sorted(c for c in zone_cs if c in sub['country'].unique())
        if not present: continue
        sub_z = sub[sub['country'].isin(present)]
        for label, combo in REP_COMBOS.items():
            g = sub_z[sub_z['combo'] == combo].dropna(subset=[SK3])
            if g.empty: continue
            koppen_rows.append({
                'zone': zone,
                'zone_name': zone_name,
                'crop': crop,
                'model_label': label,
                'combo': combo,
                'countries': ','.join(present),
                'n_countries': len(present),
                'n_tasks': int(g['country'].nunique()),
                'win_rate': round(float((g[SK3] > 0).mean() * 100)),
            })

koppen_df = pd.DataFrame(koppen_rows)
koppen_df.to_csv(OUT / 'findings_koppen.csv', index=False)
print(f"findings_koppen.csv: {len(koppen_df)} rows")

print(f"\nAll findings CSVs saved to {OUT}")
