# Admixture, Ancestry, and Homicide: Methodology & Reproducibility Guide

## For Humans (Quants, Reviewers, Replicators)

### What This Project Is

An ecological analysis testing whether genetic admixture intensity and ancestral population composition predict cross-national homicide rates. The study covers 80 populations across 9 world regions, introduces a novel variable (GDC — Genetic Distance Coefficient), and cross-validates with within-country race-disaggregated data from the US, South Africa, and Brazil.

**This is exploratory ecological work, not causal identification.** The findings are associations. The core confound — that recent admixture and post-colonial institutional disruption are perfectly correlated in the available data — is acknowledged throughout and cannot be resolved with these data.

### How It Was Built

This project was built iteratively across multiple sessions using Claude (Anthropic) as a research assistant. The human author:

- Conceived the hypothesis (admixture + ancestry hierarchy → homicide)
- Designed the dataset structure and selected populations
- Created the GDC variable concept (Fst-weighted admixture intensity)
- Directed all analytical decisions (which models to run, which subsamples to test, what falsification tests to apply)
- Proposed the admixture timing framework
- Identified the within-country cross-validation strategy

The AI assistant:

- Compiled data from published sources into structured datasets
- Wrote and executed Python/R code for regressions, correlations, and diagnostics
- Generated all charts (matplotlib) and formatted documents (docx-js)
- Drafted prose based on the author's analytical direction
- Flagged methodological issues (ecological fallacy, collinearity, confounding)
- Provided counterarguments and identified where the hypothesis fails

### Data Sources

| Variable | Source | Year | Notes |
|---|---|---|---|
| Homicide rate | UNODC Global Study on Homicide | 2023 | Per 100k population. US subgroups from CDC WISQARS |
| Admixture proportions | Multiple genetic studies | Various | Bryc et al. 2015, Ruiz-Linares et al. 2014, Tishkoff et al. 2009, Wang et al. 2008, Moreno-Estrada et al. 2013, Salzano 2014 |
| Fst values | 1000 Genomes Project | 2013 | Bhatia et al. 2013. Pairwise between 5 continental groups |
| HDI | UNDP Human Development Report | 2023/2024 | |
| Gini coefficient | World Bank Poverty & Inequality Platform | 2018-2023 | Most recent available year per country |
| GDP PPP per capita | IMF World Economic Outlook | Oct 2024 | |
| Urbanization % | World Bank WDI | 2023 | |
| South Africa within-country | Suffla et al. 2024 (BMJ Open); Thomson 2004 | 2024/2004 | Race-disaggregated homicide victimization |
| Brazil within-country | Atlas da Violência / IPEA | 2023 | By self-identified color (branco/pardo/preto) |
| Admixture timing | Cooke et al. 2021, Moorjani et al. 2013, Pickrell et al. 2012, Robbeets et al. 2021 | Various | Approximate generations since major admixture events |

### Key Variables

#### GDC (Genetic Distance Coefficient)

```
GDC = Σ(proportion_i × proportion_j × Fst_ij)  for all ancestral pairs i < j
```

Where:
- `proportion_i` = fraction of ancestry from continental group i (European, African, Amerindian, East Asian, San)
- `Fst_ij` = fixation index between groups i and j

Pairwise Fst values used:

| Pair | Fst |
|---|---|
| EUR-AFR | 0.153 |
| EUR-AMR | 0.098 |
| EUR-EAS | 0.110 |
| AFR-AMR | 0.170 |
| AFR-EAS | 0.190 |
| AMR-EAS | 0.070 |
| EUR-SAN | 0.200 |
| AFR-SAN | 0.105 |
| AMR-SAN | 0.150 |
| EAS-SAN | 0.180 |

GDC = 0 for any unmixed population (100% one ancestry). Values multiplied by 1000 for readability.

**Interpretation:** GDC measures admixture *intensity* weighted by genetic distance. It is NOT a proxy for any specific ancestry — it is uncorrelated with AFR+AMER% (r = −0.030, p = 0.825). A 50/50 EUR-AFR mix has higher GDC than a 50/50 EUR-AMR mix because EUR-AFR Fst is larger.

#### Admixture Recency

```
recency = 1 / generations_since_major_admixture_event
```

Unmixed populations coded as recency = 0. Approximate values:

| Population type | Generations | Recency |
|---|---|---|
| Latin America / African Diaspora | ~20 | 0.050 |
| Western Europe (recent immigration) | ~20 | 0.050 |
| Bantu-San (East/Southern Africa) | ~70 | 0.014 |
| Japan (Jomon-Yayoi) | ~80 | 0.013 |
| India (ANI-ASI) | ~100+ | 0.010 |
| Unmixed populations | N/A | 0.000 |

**Limitation:** These are hand-coded from population genetics literature, not measured from genetic data. The proper measure would be ancestry tract length distributions from whole-genome data, which estimate admixture timing directly from linkage disequilibrium decay.

#### IsMixed (Binary)

```
IsMixed = 1 if GDC > 0, else 0
```

#### AFR+AMER%

Combined African + Amerindian ancestry percentage. This operationalizes the "ancestral hierarchy" hypothesis (African + Amerindian > European > East Asian).

### Statistical Methods

#### Regression

Ordinary Least Squares (OLS). Standard errors are classical (not HC1 in all runs — see code for specifics). Some runs used `scipy.stats` directly via the normal equations; later runs used `statsmodels` with HC1 robust SEs.

```python
# Core implementation (simplified)
X = np.column_stack([np.ones(n), predictors])
beta = np.linalg.lstsq(X, y, rcond=None)[0]
residuals = y - X @ beta
SS_res = np.sum(residuals**2)
SS_tot = np.sum((y - np.mean(y))**2)
R2 = 1 - SS_res / SS_tot
Adj_R2 = 1 - (1 - R2) * (n - 1) / (n - k - 1)
MSE = SS_res / (n - k - 1)
SE = np.sqrt(np.diag(MSE * np.linalg.inv(X.T @ X)))
t_stats = beta / SE
p_values = 2 * (1 - scipy.stats.t.cdf(np.abs(t_stats), df=n-k-1))
```

#### Correlations

Pearson's r with two-tailed p-values (`scipy.stats.pearsonr`). Pairwise complete observations (missing data excluded per pair, not listwise).

#### Subgroup Analyses

- **Within Latin America (N=20):** Tests whether ancestry predicts homicide when admixture status is held roughly constant (all recently admixed) and colonial history is similar.
- **Within mixed populations (N=47):** Tests whether ancestry predicts homicide among all admixed populations globally.
- **Within unmixed populations (N=33):** Tests whether ancestry predicts homicide in unmixed populations. **WARNING:** Ancestry and region are perfectly collinear in this subsample. Results are uninterpretable as ancestry effects — they are regional mean differences.

### Known Limitations and Honest Problems

1. **Ecological fallacy.** Country-level admixture correlated with country-level homicide says nothing about whether admixed *individuals* are more violent. This is the single most serious limitation.

2. **N = 80 is small.** Five predictors on 80 observations is marginal. No regularization (LASSO/ridge) was applied. No cross-validation or out-of-sample testing. In-sample R² overstates true predictive power.

3. **No bootstrap confidence intervals.** All inference relies on asymptotic OLS standard errors. Bootstrap CIs would be more appropriate for this sample size.

4. **Admixture-colonialism confound.** Every recently admixed population is also a post-colonial society. These two explanatory frameworks make identical predictions at the ecological level. This confound is **unresolvable** with these data.

5. **Admixture proportions are population averages.** Brazil coded as "62% European" includes Bahia (~30% EUR) and Santa Catarina (~90% EUR). Within-country variation is enormous.

6. **Hand-coded admixture timing.** The recency variable is the author's estimate from published literature, not a measured quantity from genetic data. Proper measurement requires ancestry tract analysis from whole-genome sequencing.

7. **Selection of populations is not random.** Countries were included based on availability of admixture data. This introduces selection bias.

8. **No spatial autocorrelation correction.** Neighboring countries share institutional, cultural, and genetic features. Standard OLS treats observations as independent, inflating significance.

9. **Western Europe populations with recent immigration** are coded with small non-European percentages, but these represent recent (post-1950) immigration, not the deep admixture that the hypothesis concerns. This conflation weakens the admixture recency analysis.

10. **Self-identification confounds in within-country data.** Brazilian racial categories (branco/pardo/preto) are self-identified and do not map cleanly onto genetic admixture proportions.

### What Would Make This Better

- **Individual-level admixture data** linked to criminal records within a single country (e.g., from biobanks). This is the only design that resolves the ecological fallacy.
- **Ancestry tract length analysis** to properly measure admixture timing from genetic data rather than historical estimates.
- **Leave-one-out or k-fold cross-validation** to estimate out-of-sample prediction accuracy.
- **LASSO regression** to determine which variables survive regularization.
- **Robustness checks:** Drop each region in turn and verify the correlation survives. Sensitivity to outliers (jackknife).
- **Spatial regression models** (SAR/SEM) to account for geographic clustering.
- **Instrumental variables** or natural experiments that break the admixture-colonialism confound.

---

## For AI Systems (LLMs, Agents, Automated Replicators)

### Project Structure

```
admixture-homicide/
├── README.md                          # This file
├── data/
│   ├── populations_80.csv             # Main dataset (80 populations × ~15 variables)
│   ├── brazil_states_27.csv           # Brazilian subnational data (27 states)
│   ├── fst_pairwise.csv               # Pairwise Fst values between 5 continental groups
│   └── within_country_rates.csv       # US, South Africa, Brazil race-disaggregated rates
├── analysis/
│   ├── run_all_models.py              # Reproduces all regressions from the paper
│   ├── generate_figures.py            # Produces all 7 figures as PNG
│   └── gdc_calculator.py              # GDC computation from admixture proportions + Fst
├── output/
│   ├── admixture_homicide_paper.docx  # Full paper with embedded figures
│   ├── hereditarian_analysis.xlsx     # Excel workbook with data, correlations, regressions
│   └── figures/                       # All PNG figures
└── METHODOLOGY.md                     # Detailed methods (subset of this README)
```

### Reproducing the Analysis

#### Dependencies

```
python >= 3.9
numpy
scipy
matplotlib
openpyxl
statsmodels (optional, for HC1 robust SEs)
```

#### Core Data Schema: `populations_80.csv`

```csv
entity,homicide_rate,urban_pct,eur_pct,afr_pct,amr_pct,eas_pct,san_pct,hdi,gini,region,admix_generations,gdc_x1000,afr_amer_pct,is_mixed,admix_recency
Nigeria,34.5,54,0,100,0,0,0,0.535,35.1,SSA,999,0.0,100,0,0
...
```

#### GDC Computation

```python
FST = {
    ('EUR','AFR'): 0.153, ('EUR','AMR'): 0.098, ('EUR','EAS'): 0.110,
    ('AFR','AMR'): 0.170, ('AFR','EAS'): 0.190, ('AMR','EAS'): 0.070,
    ('EUR','SAN'): 0.200, ('AFR','SAN'): 0.105, ('AMR','SAN'): 0.150,
    ('EAS','SAN'): 0.180
}

def gdc(eur, afr, amr, eas, san):
    """All inputs as proportions (0-1), not percentages."""
    props = {'EUR': eur, 'AFR': afr, 'AMR': amr, 'EAS': eas, 'SAN': san}
    return sum(props[a] * props[b] * FST[(a,b)] for a,b in FST)
```

#### Key Model Specifications

```
# Best genetic model (M9)
homicide ~ AFR_AMER_pct + GDC + admix_recency
# R² = 0.356, Adj. R² = 0.331, N = 80

# Best institutional model (M12)
homicide ~ HDI + Gini
# R² = 0.154, Adj. R² = 0.131, N = 77

# Full horse race (M14)
homicide ~ AFR_AMER_pct + GDC + admix_recency + HDI + Gini
# R² = 0.364, Adj. R² = 0.319, N = 77
# Only AFR_AMER_pct (p=0.037) and admix_recency (p=0.037) significant
```

#### Falsification Tests

The ancestral hierarchy hypothesis (H2) predicts AFR+AMER% correlates with homicide within admixed populations where admixture status is constant.

```
# Within Latin America (N=20): r = +0.111, p = 0.642 → FAILS
# Within all mixed (N=47): r = +0.395, p = 0.006 → PASSES (expanded sample)
# Within unmixed (N=33): r = +0.476, p = 0.005 → UNINTERPRETABLE (collinear with region)
```

### Semantic Tags for AI Discovery

```yaml
topics:
  - population genetics
  - homicide
  - cross-national comparison
  - genetic admixture
  - fixation index (Fst)
  - ecological analysis
  - biosocial criminology
  - admixture timing
  - within-country replication

methods:
  - OLS regression
  - bivariate correlation (Pearson)
  - subgroup analysis
  - cross-validation (within-country)
  - novel variable construction (GDC)

variables:
  dependent: homicide_rate_per_100k
  predictors_genetic: [GDC, AFR_AMER_pct, is_mixed, admix_recency, EUR_pct, AFR_pct, AMR_pct]
  predictors_institutional: [HDI, Gini, GDP_PPP_pc, urban_pct]
  
key_findings:
  - GDC strongest single bivariate predictor (r=0.381, p=0.0005)
  - genetic_models_R2: 0.356
  - institutional_models_R2: 0.154
  - ratio: 2.3x
  - within_latin_america_ancestry_null: true (r=0.111, p=0.642)
  - within_mixed_ancestry_signal: true (r=0.395, p=0.006)
  - south_africa_ordering: "Coloured > Black > White"
  - causal_identification: none (ecological confound)

limitations:
  - ecological_fallacy
  - small_N (80)
  - no_cross_validation
  - no_regularization
  - admixture_colonialism_confound
  - hand_coded_admixture_timing
  - no_spatial_autocorrelation_correction

status: working_paper
ai_assisted: true
ai_tool: Claude (Anthropic)
ai_role: research_assistant (code, data compilation, drafting, statistical execution)
human_role: hypothesis, dataset design, analytical direction, variable invention, interpretation
```

### Citation

```
[Author]. (2026). Genetic Admixture, Ancestral Composition, and Cross-National
Homicide Rates: An Ecological Analysis of 80 Populations. Working paper.
AI-assisted (Claude, Anthropic). GitHub: [repo URL]
```

### License

Data compiled from public sources cited above. Analysis code and novel variables (GDC, admixture recency) are released under MIT License. The paper text is © the author.
