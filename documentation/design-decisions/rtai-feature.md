# RTAI — Radiation–Temperature Assimilation Index (design decisions)

The feature formerly labelled the "Farquhar photosynthesis proxy"
(`--use_farquhar`, `_compute_farquhar_series`) is renamed **RTAI — Radiation–
Temperature Assimilation Index** and given **separate maize (C4) and wheat (C3)
formulas**. It is a seasonal-scale, biophysically-motivated PROXY derived from
Farquhar–von Caemmerer–Berry (FvCB) style equations — NOT a mechanistic leaf
photosynthesis simulation.

Code: `process/featureEngineering.py` — `_compute_rtai_series`,
`_cardinal_temp_response`, `_RTAI_PARAMS`. Status date: 2026-10-19.
Backups: `process/.backup_2026-10-19/`, `setups/temporal/.backup_2026-10-19/`.

---

## Why renamed (not "scientifically wrong", but mislabelled)

The original carried the FvCB (a C3 model) name and C3 constants while being
applied to maize, a C4 crop. It is not a broken computation — it is a documented
proxy — but the "Farquhar / FvCB C3" label overclaims C3 Rubisco biochemistry on
a C4 crop and invites reviewer objection. The honest fix is a general, descriptive
name (what it is: an index driven by radiation and temperature, dialled down by
water shortage) with crop-specific parameters where the biology actually differs.

## What actually differs between the two formulas (and what is cosmetic)

The CO2 biochemistry constants (Γ, Cᵢ/Cₐ, Kc) enter the old formula only as fixed
multipliers, so after the network's input normalisation they mostly rescale — a
constants-only "C4 variant" would change almost nothing. The two differences that
genuinely change the index's SHAPE, and therefore make maize and wheat honestly
distinct, are:

1. **Temperature response** — crop-specific cardinal temperatures, replacing the
   old monotone Arrhenius term (which implied "hotter is always better" and never
   turned down under heat).
2. **Carboxylation term** — C3 (wheat) is Rubisco CO2-*limited* (Michaelis–Menten
   with photorespiration); C4 (maize) is Rubisco CO2-*saturated* via the
   CO2-concentrating mechanism, with photorespiration ≈ 0. This is a
   functional-form difference, not a constant swap.

The CO2 constants are still set per pathway for transparency/completeness, but
their numerical effect is small — stated here so the paper doesn't overclaim them.

---

## The formula

Shared skeleton (per timestep of the growing season):

```
PAR    = 0.48 · cum_rad
f_T    = cardinal-temperature response(tavg; Tmin, Topt, Tmax)   ∈ [0,1]
Vcmax  = 100 · f_T · (1 + 5n)          (Rubisco capacity; n = leaf-N proxy)
J      = 0.1 · PAR · f_T               (electron transport, light-driven)
ci     = (Cᵢ/Cₐ) · co2

  wheat (C3): Ac = Vcmax · (ci − Γ) / (ci + Kc)     [CO2-limited + photorespiration]
  maize (C4): Ac = Vcmax                             [CO2-saturated, Γ ≈ 0]

Aj      = J · (ci − Γ) / (4·ci + 8·Γ)
A       = 2·Ac·Aj / (Ac + Aj)          (harmonic-mean co-limitation)
RTAI    = A · PAR/(PAR+100) · cum_prec/(cum_prec + water_half)
```

`f_T` is the normalised beta response of Yan & Hunt (1999): 1.0 at Topt, 0 at or
below Tmin and at or above Tmax, declining smoothly on the hot side.

---

## Crop parameter table (every value sourced below)

| Parameter | Wheat (C3) | Maize (C4) | Decision / source key |
|---|---|---|---|
| Base temperature Tmin (°C) | 0 | 8 | cardinal temps [1],[3] |
| Optimum temperature Topt (°C) | 22 | 33 | cardinal temps + leaf-photosynthesis optima [1],[2],[3],[4] |
| Maximum temperature Tmax (°C) | 37 | 45 | cardinal temps [1],[3],[4] |
| Carboxylation | CO2-limited (Kc=404.9, Γ=42.75 µmol/mol) | CO2-saturated (Γ≈0) | C3 constants [6]; C4 CO2-concentration [5],[7] |
| Cᵢ/Cₐ ratio | 0.70 | 0.40 | [5] |
| Water half-sufficiency (mm) | 300 | 400 | FAO seasonal water need, maize > wheat [8] |

### Sources (links directly below the decisions they support)

Temperature — wheat (C3):
- [1] Porter, J.R. & Gawith, M. (1999). *Temperatures and the growth and
  development of wheat: a review.* European Journal of Agronomy 10:23–36.
  https://www.sciencedirect.com/science/article/pii/S1161030198000471
  (DOI: 10.1016/S1161-0301(98)00047-1) — base ≈0 °C, optimum ≈21–22 °C, max ≈37 °C.
- [2] Acevedo, Silva & Silva (FAO), *Wheat growth and physiology.*
  https://www.fao.org/4/Y4011E/y4011e06.htm — leaf photosynthesis optimum
  20–25 °C; declines above 25 °C, roughly halved at 45 °C.

Temperature — maize (C4):
- [3] Sánchez, B., Rasmussen, A. & Porter, J.R. (2014). *Temperatures and the
  growth and development of maize and rice: a review.* Global Change Biology
  20(2):408–417. https://doi.org/10.1111/gcb.12349
  (record: https://knowledgecenter.cimmyt.org/bib/30578) — base ≈8 °C,
  optimum ≈30–34 °C, upper/lethal ≈40–45 °C.
- [4] Crafts-Brandner, S.J. & Salvucci, M.E. (2002). *Sensitivity of
  photosynthesis in a C4 plant, maize, to heat stress.* Plant Physiology
  129:1773–1780. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC166765/ — broad
  photosynthesis optimum 28–37.5 °C, net photosynthesis inhibited above 38 °C.
  Supporting: Massad et al. (2007), Plant, Cell & Environment 30:1191–1204,
  https://onlinelibrary.wiley.com/doi/full/10.1111/j.1365-3040.2007.01691.x
  (maize leaf-photosynthesis optimum 33–38 °C).

Pathway biochemistry (C3 vs C4):
- [5] *Adapting C4 photosynthesis to atmospheric change…* PNAS (2025).
  https://www.pnas.org/doi/10.1073/pnas.2419943122 — Cᵢ/Cₐ ≈ 0.4 in C4 vs ≈ 0.7
  in C3 leaves (greater CO2 drawdown in C4).
- [6] Bernacchi, C.J. et al. (2002). Plant Physiology 130:1992–1998.
  https://doi.org/10.1104/pp.008250 — the C3 constants Kc = 404.9 and
  Γ = 42.75 µmol/mol used for wheat.
- [7] von Caemmerer, S. (2000). *Biochemical Models of Leaf Photosynthesis.*
  CSIRO Publishing. https://doi.org/10.1071/9780643103405 — C4
  CO2-concentrating mechanism saturates Rubisco and suppresses photorespiration
  (basis for the C4 CO2-saturated carboxylation term and Γ ≈ 0).

Water sufficiency:
- [8] FAO, *Crop water needs* (Irrigation Water Management, Training Manual 3,
  Ch. 3). https://www.fao.org/4/s2022e/s2022e07.htm — indicative seasonal crop
  water need: maize ≈ 500–800 mm vs wheat ≈ 450–650 mm (maize higher → higher
  half-sufficiency constant). Method: FAO-56, Allen et al. (1998),
  https://www.fao.org/4/x0490e/x0490e00.htm.

Temperature-response functional form:
- [9] Yan, W. & Hunt, L.A. (1999). *An equation for modelling the temperature
  response of plants.* Annals of Botany 84:607–614.
  https://doi.org/10.1006/anbo.1999.0955 — the normalised beta cardinal-
  temperature curve used for `f_T`.

---

## Correction logged (water direction)

An earlier proposal had maize reaching "enough water" *sooner* (lower constant)
on a water-use-efficiency argument. That was wrong for this term: maize is more
water-use-efficient (more biomass per mm) but has a **higher** absolute seasonal
water requirement than wheat ([8]). The water-sufficiency half-saturation is
therefore **higher** for maize (400 mm) than wheat (300 mm). Note `water_half` is
a sufficiency *scale*, informed by (not equal to) FAO seasonal ET; it is the
least-constrained parameter here and a reasonable candidate for sensitivity
analysis.

---

## Naming migration status (partial, by design)

- **Renamed now:** the user-facing CLI flag `--use_farquhar` → `--use_rtai` in the
  three temporal baselines (`cnnLSTMBaseline.py`, `linearBaselines.py`,
  `tstBaselines.py`), the feature/channel concept name, and the core function
  (`_compute_rtai_series`, with `_compute_farquhar_series` kept as an alias).
- **Still named `use_farquhar`:** the internal config attribute / builder kwarg /
  cache key, which threads through `featureEngineering.py` (builder params),
  `featureCache.py`, `metrics.py`, `validateModel.py`, and the untouched
  baselines (`multitaskBaselines.py`, the `walk-forward/` copies). Renaming that
  everywhere is a larger cross-cutting change held for a separate pass, so those
  scripts keep working unchanged. In the three temporal scripts the new flag
  bridges to the old kwarg: `use_farquhar=args.use_rtai`.

---

## Rejected / deferred

- **Constants-only C4 variant** — rejected: Γ/Cᵢ/Cₐ/Kc enter as fixed multipliers
  and wash out under input normalisation; it would look different on paper but
  behave identically. The real differences are temperature and carboxylation
  form (above).
- **Full internal rename of `use_farquhar`** — deferred to a dedicated pass to
  avoid breaking the untouched baselines and the cache/eval plumbing.
- **Keeping the monotone Arrhenius temperature term** — rejected: it never turned
  down under heat, which is wrong for both C3 and C4 and erases the maize/wheat
  distinction.
