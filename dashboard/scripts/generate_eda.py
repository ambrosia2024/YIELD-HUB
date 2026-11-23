#!/usr/bin/env python3
"""
Refresh pages/tab_EDA.html:
  1. Replaces the 5 data constants (RAW, RAW_QUANT, etc.) from data/*.json
  2. Replaces all 8 findings sections with values computed from data
Run:  python3 scripts/generate_eda.py
"""
import json, re, statistics
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent  # dashboard/

UNIT = {
    'yield':'t/ha', 'tmin':'deg C', 'tmax':'deg C', 'tavg':'deg C',
    'prec':'mm/d', 'rad':'J/m2', 'et0':'mm/d', 'vpd':'hPa',
    'cwb':'mm/d', 'ssm':'-', 'rsm':'mm', 'ndvi':'-', 'fpar':'-',
}
UNIT_HTML = {
    'yield':'t/ha', 'tmin':'&deg;C', 'tmax':'&deg;C', 'tavg':'&deg;C',
    'prec':'mm/d', 'rad':'J/m&sup2;', 'et0':'mm/d', 'vpd':'hPa',
    'cwb':'mm/d', 'ssm':'-', 'rsm':'mm', 'ndvi':'-', 'fpar':'-',
}
# HTML entities for emoji (avoids encoding issues in this file)
COND = {
    'heat_stress':        ('&#128293;','Extreme Heat',      'tmax_condition'),
    'frost_risk':         ('&#129482;','Frost Risk',         'tmin_condition'),
    'excess_rainfall':    ('&#9928;&#65039;','Excess Rainfall','prec_condition'),
    'warm_season':        ('&#9728;&#65039;','Warm Season',   'tavg_condition'),
    'cold_season':        ('&#129398;','Cold Season',        'tavg_condition'),
    'water_deficit':      ('&#127797;','Water Deficit',      'cwb_condition'),
    'water_surplus':      ('&#127754;','Water Surplus',      'cwb_condition'),
    'low_soil_moisture':  ('&#127580;&#65039;','Dry Soil',   'rsm_condition'),
    'high_soil_moisture': ('&#128167;','Waterlogged',        'rsm_condition'),
}

# ---- helpers -----------------------------------------------------------------

def fp(v, d=1):
    if v is None: return 'n/a'
    return ('+' if v >= 0 else '') + f'{v:.{d}f}%'

def fv(v, unit='', d=2):
    if v is None: return 'n/a'
    s = ('+' if v >= 0 else '') + f'{v:.{d}f}'
    return s + (' ' + unit if unit else '')

def var_stats(raw, variable, crop='both'):
    rows = [r for r in raw if r['variable']==variable and r['quantile']=='annual'
            and (crop=='both' or r['crop']==crop)]
    if not rows: return None
    shifts = [r['mean_test']-r['mean_train'] for r in rows
              if r.get('mean_test') is not None and r.get('mean_train') is not None]
    spreads = [(r['std_test']-r['std_train'])/r['std_train']*100 for r in rows
               if r.get('std_test') and r.get('std_train')]
    if not shifts: return None
    return {
        'median':  round(statistics.median(shifts), 3),
        'min':     round(min(shifts), 3),
        'max':     round(max(shifts), 3),
        'sig_pct': round(100*sum(1 for r in rows if r.get('sig')==1)/len(rows), 1),
        'up_pct':  round(100*sum(1 for r in rows if r.get('direction')==1)/len(rows), 1),
        'n':       len(rows),
        'spread_pct': round(statistics.median(spreads), 1) if spreads else None,
    }

# ---- section builders --------------------------------------------------------

def ef1(crop, raw, pyr_summary):
    s = var_stats(raw, 'yield', crop)
    if not s: return None
    u = UNIT_HTML['yield']
    pyr = sorted([r for r in pyr_summary if r['variable']=='yield' and r['crop']==crop],
                 key=lambda r: r['test_year_idx'])
    pyr_str = ' &middot; '.join(f'yr{r["test_year_idx"]}: {r["median_shift_pct"]:+.1f}%' for r in pyr)
    spread_note = ''
    if s['spread_pct'] is not None:
        spread_note = (f', variability {("widening" if s["spread_pct"]>0 else "narrowing")} '
                       f'{abs(s["spread_pct"]):.0f}%')
    h3 = (f'{crop.capitalize()} yield shifted {fv(s["median"],u,2)} (median); '
          f'{s["up_pct"]}% of countries trending up{spread_note}')
    para = (f'Across {s["n"]} countries, yield in test years shifted '
            f'{fv(s["median"],u,2)} (range {fv(s["min"],u,2)} to {fv(s["max"],u,2)}). '
            f'{s["up_pct"]}% show an upward trend; '
            f'{s["sig_pct"]}% are statistically significant (KS p&lt;0.05). '
            + (f'Per-year progression: {pyr_str}.' if pyr_str else ''))
    return ('<div class="fs-num" aria-hidden="true">F1</div>\n'
            '<div class="fs-body">\n'
            f'  <h3>{h3}</h3>\n'
            f'  <p class="ans">{para}</p>\n'
            '</div>')

def ef2(raw):
    vars_order = ['vpd','et0','tmax','tmin','tavg','prec','cwb']
    tbody = ''
    for var in vars_order:
        s = var_stats(raw, var, 'both')
        if not s: continue
        u = UNIT_HTML.get(var, '')
        tbody += (f'<tr><td>{var} ({u})</td>'
                  f'<td>{fv(s["median"],u,2)}</td>'
                  f'<td class="{"hi" if s["sig_pct"]>50 else ""}">{s["sig_pct"]}%</td>'
                  f'<td class="{"hi" if s["up_pct"]>60 else "lo" if s["up_pct"]<40 else ""}">{s["up_pct"]}%</td>'
                  '</tr>\n')
    vpd = var_stats(raw, 'vpd', 'both')
    et0 = var_stats(raw, 'et0', 'both')
    if vpd and et0:
        h3 = (f'VPD ({fv(vpd["median"],"hPa",2)}, {vpd["sig_pct"]}% sig) and '
              f'ET&#8320; ({fv(et0["median"],"mm/d",2)}, {et0["sig_pct"]}% sig) '
              f'are the most consistently shifted weather variables')
    else:
        h3 = 'Weather variable shift summary'
    return ('<div class="fs-num" aria-hidden="true">F2</div>\n'
            '<div class="fs-body">\n'
            f'  <h3>{h3}</h3>\n'
            '  <p class="ans">Median shift and significance across all countries (both crops combined).</p>\n'
            '  <table class="fs-tbl"><thead><tr>'
            '<th>Variable</th><th>Median shift</th><th>% Sig</th><th>% Countries &uarr;</th>'
            f'</tr></thead><tbody>{tbody}</tbody></table>\n'
            '</div>')

def ef3(raw):
    rsm  = var_stats(raw, 'rsm',  'both')
    cwb  = var_stats(raw, 'cwb',  'both')
    prec = var_stats(raw, 'prec', 'both')
    if not rsm: return None
    u = UNIT_HTML['rsm']
    down_pct = 100 - rsm['up_pct']
    direction = 'declining' if rsm['median'] < 0 else 'rising'
    h3 = f'Root-zone soil moisture is {direction} ({fv(rsm["median"],u,1)}) '
    if cwb and prec and abs(prec['median']) < abs(rsm['median']):
        h3 += 'despite stable precipitation &mdash; ET&#8320; is extracting more water'
    else:
        h3 += 'alongside changes in evaporative demand'
    para = (f'Median RSM shifted {fv(rsm["median"],u,2)} '
            f'(range {fv(rsm["min"],u,2)} to {fv(rsm["max"],u,2)}); '
            f'{down_pct:.0f}% of countries show a decline; {rsm["sig_pct"]}% significant. ')
    if prec:
        para += f'Precipitation: {fv(prec["median"],UNIT_HTML["prec"],3)} ({prec["sig_pct"]}% sig). '
    para += ('Rising ET&#8320; is extracting soil moisture faster than precipitation replenishes it, '
             'increasing structural agricultural drought risk in the test period.')
    return ('<div class="fs-num" aria-hidden="true">F3</div>\n'
            '<div class="fs-body">\n'
            f'  <h3>{h3}</h3>\n'
            f'  <p class="ans">{para}</p>\n'
            '</div>')

def ef4(crop_key, esummary):
    if crop_key == 'both':
        m, w = esummary.get('maize',{}), esummary.get('wheat',{})
        thead = ('<tr><th>Condition</th><th>Maize (% test years)</th>'
                 '<th>Wheat (% test years)</th></tr>')
        tbody = ''
        all_avgs = []
        for cond in ['heat_stress','frost_risk','excess_rainfall']:
            if cond not in COND: continue
            icon,label,_ = COND[cond]
            mp = m.get(cond,{}).get('pct')
            wp = w.get(cond,{}).get('pct')
            if mp is None and wp is None: continue
            avg = ((mp or 0)+(wp or 0))/2
            all_avgs.append(avg)
            hi_m = ' class="hi"' if mp and mp>15 else ''
            hi_w = ' class="hi"' if wp and wp>15 else ''
            tbody += (f'<tr><td>{icon} {label}</td>'
                      f'<td{hi_m}>{mp:.1f}%</td><td{hi_w}>{wp:.1f}%</td>'
                      f'</tr>\n')
        min_r = f'{min(all_avgs)/10:.0f}' if all_avgs else '2'
        max_r = f'{max(all_avgs)/10:.0f}' if all_avgs else '4'
        h3 = 'Frequency of extreme conditions (heat, frost, excess rainfall) in test years'
    else:
        s = esummary.get(crop_key, {})
        thead = '<tr><th>Condition</th><th>% test years</th></tr>'
        tbody = ''
        cond_pcts = [(k,s.get(k,{}).get('pct')) for k in ['heat_stress','frost_risk','excess_rainfall'] if s.get(k,{}).get('pct') is not None]
        for cond, pct in cond_pcts:
            icon, label, _ = COND[cond]
            hi = ' class="hi"' if pct>20 else ''
            tbody += (f'<tr><td>{icon} {label}</td><td{hi}>{pct:.1f}%</td></tr>\n')
        top2 = sorted(cond_pcts, key=lambda x: x[1] or 0, reverse=True)[:2]
        tc = ', '.join(f'{COND[k][1]} ({v:.0f}%)' for k,v in top2 if v)
        h3 = f'{crop_key.capitalize()} test years: {tc}'
    para = ('Using the 10th/90th percentile of each country\u2019s training distribution as thresholds, '
            'expected frequency per condition is ~10%. '
            'Test-period frequencies are systematically higher, confirming the test period '
            'is not a random draw from the training distribution.')
    return ('<div class="fs-num" aria-hidden="true">F4</div>\n'
            '<div class="fs-body">\n'
            f'  <h3>{h3}</h3>\n'
            f'  <p class="ans">{para}</p>\n'
            f'  <table class="fs-tbl"><thead>{thead}</thead><tbody>{tbody}</tbody></table>\n'
            '</div>')

def ef5(esummary):
    m, w = esummary.get('maize',{}), esummary.get('wheat',{})
    tbody = ''
    for cond in ['heat_stress','frost_risk','excess_rainfall']:
        if cond not in COND: continue
        icon,label,_ = COND[cond]
        mv = m.get(cond,{}).get('yield_vs_normal_pct')
        wv = w.get(cond,{}).get('yield_vs_normal_pct')
        if mv is None and wv is None: continue
        mc = ' class="lo"' if mv is not None and mv<-1 else (' class="hi"' if mv is not None and mv>1 else '')
        wc = ' class="lo"' if wv is not None and wv<-1 else (' class="hi"' if wv is not None and wv>1 else '')
        tbody += f'<tr><td>{icon} {label}</td><td{mc}>{fp(mv)}</td><td{wc}>{fp(wv)}</td></tr>\n'
    dm = m.get('drought',{}).get('yield_vs_normal_pct')
    dw = w.get('drought',{}).get('yield_vs_normal_pct')
    hm = m.get('heat_stress',{}).get('yield_vs_normal_pct')
    hw = w.get('heat_stress',{}).get('yield_vs_normal_pct')
    if None not in (dm, dw, hm, hw):
        h3 = (f'Drought hurts both crops ({fp(dm)}&ndash;{fp(dw)}); '
              f'warm seasons help maize ({fp(hm)}) but hurt wheat ({fp(hw)}) '
              f'&mdash; a fundamental C4/C3 physiology contrast')
    else:
        h3 = 'Yield response to extreme conditions varies by crop and condition type'
    para = ('Yield impact = mean deviation in condition years minus mean deviation in normal years, '
            'both relative to the country\u2019s test-period mean (removes structural yield trend).')
    return ('<div class="fs-num" aria-hidden="true">F5</div>\n'
            '<div class="fs-body">\n'
            f'  <h3>{h3}</h3>\n'
            f'  <p class="ans">{para}</p>\n'
            '  <table class="fs-tbl">\n'
            '    <thead><tr><th>Condition</th><th>Maize yield vs normal</th><th>Wheat yield vs normal</th></tr></thead>\n'
            f'    <tbody>{tbody}</tbody>\n'
            '  </table>\n'
            '  <div class="callout ok"><strong>C4 vs C3 physiology:</strong> Maize (C4) benefits from warm seasons; wheat (C3) is harmed by the same conditions during grain fill.</div>\n'
            '  <div class="callout warn"><strong>Forecasting implication:</strong> Dry soil and extreme heat each appear 2&ndash;4&times; more often in test years &mdash; models face systematic out-of-distribution weather, which partly explains the skill gap.</div>\n'
            '</div>')

# ---- injection ---------------------------------------------------------------

def replace_section(html, sec_id, inner):
    pat = rf'(<section[^>]*id="{sec_id}"[^>]*>)(.*?)(</section>)'
    repl = rf'\g<1>\n      {inner}\n    \g<3>'
    updated, n = re.subn(pat, repl, html, count=1, flags=re.DOTALL)
    return updated, n > 0

def _find_json_end(html, pos):
    """Return (open_pos, end_pos) for JSON value starting at pos (handles \[ or [ forms)."""
    if pos < len(html) and html[pos] == '\\': pos += 1
    if pos >= len(html) or html[pos] not in '[{': return None, None
    open_c, close_c = html[pos], (']' if html[pos] == '[' else '}')
    depth, i = 0, pos
    while i < len(html):
        c = html[i]
        if c == '\\' and i+1 < len(html):
            nc = html[i+1]; i += 2
            if nc == open_c: depth += 1
            elif nc == close_c:
                depth -= 1
                if depth == 0: return pos, i
        else:
            i += 1
            if c == open_c: depth += 1
            elif c == close_c:
                depth -= 1
                if depth == 0: return pos, i
    return None, None

def inject_constants(html, data):
    for name, value in data.items():
        m = re.search(rf'\bconst\s+{re.escape(name)}\s*=\s*', html)
        if not m:
            print(f'  WARNING: const {name} not found'); continue
        val_start = m.end()
        _, close_pos = _find_json_end(html, val_start)
        if close_pos is None:
            print(f'  WARNING: could not parse value for {name}'); continue
        semi = html.find(';', close_pos)
        if semi < 0:
            print(f'  WARNING: no semicolon after {name}'); continue
        new_val = json.dumps(value, separators=(',',':'))
        html = html[:m.end()] + new_val + ';' + html[semi+1:]
        print(f'  OK: {name:20s} -> {len(new_val)//1024} KB')
    return html

# ---- main --------------------------------------------------------------------

def main():
    target = ROOT / 'pages' / 'tab_EDA.html'
    if not target.exists():
        print(f'ERROR: {target}'); return 1

    print('Loading data...')
    vs  = json.loads((ROOT/'data'/'eda_variable_shift.json').read_text())
    pyr = json.loads((ROOT/'data'/'eda_peryear.json').read_text())
    raw, raw_quant = vs['annual'], vs['quantile']
    per_year, extreme, esummary = pyr['per_year_shift'], pyr['extreme_years'], pyr['extreme_summary']

    # pre-compute per-year summary for ef1 sections
    groups = defaultdict(list)
    for r in per_year:
        if r.get('variable')=='yield_detrended': continue
        if r.get('shift_pct') is not None:
            groups[(r['crop'],r['variable'],r['test_year_idx'])].append(r['shift_pct'])
    pyr_summary = [{'crop':c,'variable':v,'test_year_idx':yi,
                    'median_shift_pct':round(statistics.median(vals),3)}
                   for (c,v,yi),vals in groups.items()]

    print(f'  RAW={len(raw)}, PER_YEAR={len(per_year)}, EXTREME={len(extreme)}')

    html = target.read_text(encoding='utf-8')
    print(f'  Original: {len(html)//1024} KB\n')

    # Data loaded via fetch() — no data injection needed.

    # Inject COND_META + COND_ORDER so HTML stays in sync with Python COND dict
    print('\nInjecting COND_META...')
    # Build JS COND_META from Python COND dict
    # COND keys: (icon_html, label, col)
    # col mapping: tmax_condition, tavg_condition, tmin_condition, cwb_condition, prec_condition, rsm_condition
    COL_MAP = {
        'tmax_condition': 'tmax_condition',
        'tavg_condition': 'tavg_condition',
        'tmin_condition': 'tmin_condition',
        'cwb_condition':  'cwb_condition',
        'prec_condition': 'prec_condition',
        'rsm_condition':  'rsm_condition',
    }
    PILL_MAP = {
        'tmax_condition': 'pill-heat',
        'tavg_condition': 'pill-cold',
        'tmin_condition': 'pill-frost',
        'cwb_condition':  'pill-dry',
        'prec_condition': 'pill-wet',
        'rsm_condition':  'pill-soil-dry',
    }
    cond_entries = []
    for key, (icon, label, col) in COND.items():
        pill = PILL_MAP.get(col, 'pill-ns')
        cond_entries.append(
            f"  {key}: {{label:'{label}',icon:'{icon}',pill:'{pill}',col:'{col}'}}"
        )
    cond_order_absolute = ['heat_stress', 'frost_risk', 'excess_rainfall']
    cond_order_js = json.dumps([k for k in cond_order_absolute if k in COND])
    new_cond_block = (
        '/* GEN:cond_meta — replaced by generate_eda.py, do not edit manually */\n'
        'const COND_META={\n' + ',\n'.join(cond_entries) + '\n};\n'
        'const COND_ORDER=' + cond_order_js + ';\n'
        '/* /GEN:cond_meta */'
    )
    import re as _re
    old_block_pat = r'/\* GEN:cond_meta.*?/\* /GEN:cond_meta \*/'
    html, n = _re.subn(old_block_pat, new_cond_block, html, count=1, flags=_re.DOTALL)
    print(f'  {"OK" if n else "NOT FOUND"}: COND_META block')

    print('\nInjecting findings sections...')
    sections = {
        'ef1-maize': ef1('maize', raw, pyr_summary),
        'ef1-wheat': ef1('wheat', raw, pyr_summary),
        'ef2-both':  ef2(raw),
        'ef3-both':  ef3(raw),
        'ef4-both':  ef4('both',  esummary),
        'ef4-maize': ef4('maize', esummary),
        'ef4-wheat': ef4('wheat', esummary),
        'ef5-both':  ef5(esummary),
    }
    for sec_id, content in sections.items():
        if content is None:
            print(f'  SKIP: {sec_id} (no data)')
            continue
        html, ok = replace_section(html, sec_id, content)
        print(f'  {"OK" if ok else "NOT FOUND"}: {sec_id}')

    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    html = re.sub(r'<!-- Generated .* -->',
                  f'<!-- Generated {ts} by generate_eda.py -->', html, count=1)
    if '<!-- Generated' not in html:
        html = html.replace('<body>',
                            f'<body>\n<!-- Generated {ts} by generate_eda.py -->', 1)

    target.write_bytes(html.encode('utf-8'))
    print(f'\nDone. {target}  ({len(html)//1024} KB)  [{ts}]')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
