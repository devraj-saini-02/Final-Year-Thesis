# Diversity extension: what its clusters look like

Observations from 2026-09-15, kept for later work on the clustering extension
(`oalcf-main/run_dq.py`, `oalcf-main/utils/diverse_query.py`). **Nothing here has been acted on.**
All of it is descriptive: it uses the test stream's true language labels, so none of it may be used
to choose a threshold, a clusterer or a gamma (see section 5).

## How the numbers were made

- Each session's clusters were rebuilt from the saved ECAPA embeddings with the extension's own
  `cluster_session` (cosine distance, average linkage, threshold 0.25). The rebuild matches the
  cluster counts logged during the runs (`al_diversity.csv`) in **all 463 sessions** of the four
  corpora.
- Clusters see only the audio, never the target label or the model. They are therefore **identical
  for every target language, every gamma and both arms** of a corpus. Any explanation of why one
  language prefers another gamma must involve how that language sits inside the clusters, not the
  clusters themselves.
- Figures: `runs/analysis/cluster_analysis/<corpus>_stream.png`, `<corpus>_sessions.png` and
  `<corpus>_clusters_025_vs_015.png` (`tools/plot_session_clusters.py`, `tools/plot_cluster_languages.py`).
- Tables: `runs/analysis/cluster_analysis/threshold_sweep.txt` (`tools/cluster_threshold_sweep.py`).

## 1. At threshold 0.25, a session is one giant cluster, often one medium one, and crumbs

Share of the 720-clip session, mean over sessions:

| Corpus | Largest | 2nd | 3rd | Rest | Clusters >=10% | Clusters under 8 clips | Sessions with largest >=50% |
|---|---|---|---|---|---|---|---|
| AC | 88% | 8% | 2% | 2% | 1.4 | 7.8 | 98% |
| CR | 65% | 32% | 1% | 2% | 1.9 | 7.0 | 96% |
| SEA | 70% | 26% | 2% | 2% | 1.8 | 9.0 | 88% |
| SA | 79% | 18% | 1% | 2% | 1.6 | 7.4 | 100% |

The logged "9-11 clusters per session" overstates the structure: only 1-2 are real groups (95-97% of
the session). The rest are crumbs of 1-7 clips: outlying clips far from everything else.

## 2. The clusters mix languages

A session holds 2.2 (AC) to 4.5 (CR, SA) languages but only 1-2 real clusters, so the big cluster
nearly always spans several languages. Homogeneity at 0.25 is 0.20-0.21 for CR, SEA and SA (AC 0.60,
because its sessions hold about two languages). Examples: CR session 10, where one cluster merges
Basque (filler), Georgian and Abkhaz, although t-SNE draws them as three separate groups; AC sessions
31 and 74, where one cluster holds 93% and 98% of a session with 12 and 4 languages.

## 3. Languages separate only in small sub-clusters; no threshold gives one cluster per language

Same tree, cut tighter. Pooled over all 463 sessions of the four corpora
(`tools/cluster_threshold_sweep.py`, full per-corpus tables in
`runs/analysis/cluster_analysis/threshold_sweep.txt`):

| Threshold | Clusters | Mini (<8 clips) | Clips in mini | Biggest cluster | Homogeneity | Completeness | V | Clips in a >=90% pure cluster | q of a typical clip's cluster |
|---|---|---|---|---|---|---|---|---|---|
| **0.25 (used)** | 10 | 8 | 2% | 73% | 0.28 | 0.15 | 0.12 | 29% | 4.30 |
| 0.22 | 23 | 19 | 5% | 57% | 0.44 | 0.19 | 0.21 | 42% | 2.67 |
| 0.20 | 41 | 34 | 10% | 46% | 0.59 | 0.21 | 0.27 | 55% | 1.60 |
| **0.18** | 72 | 63 | 17% | 37% | 0.74 | **0.22** | **0.31** | 70% | 0.95 |
| 0.16 | 124 | 112 | 29% | 28% | 0.84 | 0.20 | 0.30 | 81% | 0.47 |
| 0.15 | 159 | 147 | 37% | 24% | 0.89 | 0.19 | 0.29 | 85% | 0.29 |
| 0.14 | 204 | 191 | 46% | 20% | 0.92 | 0.18 | 0.28 | 89% | 0.16 |
| 0.12 | 315 | 304 | 66% | 11% | 0.96 | 0.16 | 0.26 | 94% | 0.05 |
| 0.10 | 453 | 447 | 85% | 5% | 0.99 | 0.14 | 0.24 | 98% | 0.02 |

Homogeneity climbs all the way to 1.0, but only because clusters shatter: past 0.16 most clips sit in
clusters of fewer than 8. Completeness and V both peak at **0.18**, so that is where the partition
best matches the language structure. CR is the hardest corpus (homogeneity 0.62 at 0.18, 0.85 at
0.15); AC the easiest (0.77 at 0.18) but its sessions hold only ~2.2 languages. t-SNE overstates the
separation: groups that look clean in 2-D are not as tidy in 256-D cosine space.

**Tightening the threshold forces gamma up.** The quota of the cluster a typical clip sits in falls
from 4.30 (at 0.25) to 0.95 (0.18) to 0.29 (0.15), and a cluster's second pick is scored
`gamma**(1/q)`. So the same gamma bites far harder on smaller clusters: at 0.25 gamma = 0.8 multiplies
a second pick by 0.95 (nearly nothing, which is why the extension changes ~1 query in 8), but at 0.15
it multiplies by 0.46 (close to a hard one-pick-per-cluster rule). To hold the second-pick multiplier
at a chosen level `m`, set **gamma = m ** q**.

### 3a. Gamma vs the second-pick decay, by threshold (pooled)

`q` is the quota of the cluster a typical clip sits in; a cluster's second pick is multiplied by
`gamma**(1/q)`. The last column is the gamma that keeps that multiplier at 0.95, i.e. as gentle as
the current setting (threshold 0.25, gamma 0.8) actually is.

| Threshold | q | gamma=0.8 | gamma=0.9 | gamma=0.95 | gamma for a 0.95 multiplier |
|---|---|---|---|---|---|
| **0.25 (used)** | 4.30 | **0.95** | 0.98 | 0.99 | 0.80 |
| 0.22 | 2.67 | 0.92 | 0.96 | 0.98 | 0.87 |
| 0.20 | 1.60 | 0.87 | 0.94 | 0.97 | 0.92 |
| 0.18 | 0.95 | 0.79 | 0.89 | 0.95 | **0.95** |
| 0.16 | 0.47 | 0.62 | 0.80 | 0.90 | 0.98 |
| 0.15 | 0.29 | 0.46 | 0.69 | 0.84 | 0.985 |
| 0.14 | 0.16 | 0.24 | 0.51 | 0.72 | 0.99 |
| 0.12 | 0.05 | 0.01 | 0.11 | 0.34 | 0.997 |
| 0.10 | 0.02 | 0.00 | 0.01 | 0.09 | 0.999 |

Each corpus has its own `q` (column below), so gamma = m ** q can be set per corpus if wanted.

### 3b. Per corpus: homogeneity and mini clusters at each threshold

Mini cluster = fewer than 8 clips. "Clips in mini" is the share of the 720-clip session sitting in
them. "Pure clips" is the share in a cluster that is at least 90% one language.

**African Continent** (83 sessions, 2.2 languages per session)

| Threshold | Clusters | Mini | Clips in mini | Homog. | Compl. | V | Pure clips | q |
|---|---|---|---|---|---|---|---|---|
| 0.25 | 10.1 | 7.8 | 2% | 0.60 | 0.14 | 0.11 | 67% | 5.22 |
| 0.22 | 20.9 | 16.8 | 5% | 0.66 | 0.10 | 0.11 | 71% | 3.92 |
| 0.20 | 36.6 | 30.2 | 9% | 0.71 | 0.08 | 0.11 | 74% | 2.05 |
| 0.18 | 65.0 | 56.6 | 16% | 0.77 | 0.08 | 0.12 | 79% | 1.40 |
| 0.16 | 111.3 | 101.6 | 27% | 0.83 | 0.08 | 0.12 | 84% | 0.71 |
| 0.15 | 142.1 | 131.3 | 33% | 0.85 | 0.07 | 0.11 | 86% | 0.45 |
| 0.14 | 181.4 | 169.5 | 40% | 0.88 | 0.07 | 0.11 | 88% | 0.26 |
| 0.12 | 279.7 | 265.7 | 58% | 0.93 | 0.06 | 0.11 | 94% | 0.07 |
| 0.10 | 407.2 | 397.6 | 80% | 0.97 | 0.06 | 0.10 | 97% | 0.03 |
| 0.08 | 553.5 | 550.9 | 96% | 0.99 | 0.05 | 0.10 | 99% | 0.01 |

**Caucasus Region** (159 sessions, 4.5 languages per session) - the hardest corpus

| Threshold | Clusters | Mini | Clips in mini | Homog. | Compl. | V | Pure clips | q |
|---|---|---|---|---|---|---|---|---|
| 0.25 | 9.3 | 7.0 | 2% | 0.21 | 0.14 | 0.11 | 22% | 3.83 |
| 0.22 | 22.5 | 18.6 | 5% | 0.32 | 0.16 | 0.17 | 29% | 2.52 |
| 0.20 | 40.2 | 34.1 | 10% | 0.45 | 0.20 | 0.24 | 40% | 1.67 |
| 0.18 | 70.8 | 61.9 | 17% | 0.62 | 0.22 | 0.30 | 56% | 0.95 |
| 0.16 | 121.1 | 109.8 | 29% | 0.78 | 0.22 | 0.32 | 71% | 0.45 |
| 0.15 | 155.8 | 143.5 | 36% | 0.85 | 0.21 | 0.32 | 79% | 0.25 |
| 0.14 | 198.9 | 186.4 | 45% | 0.89 | 0.20 | 0.31 | 84% | 0.14 |
| 0.12 | 309.6 | 298.5 | 65% | 0.95 | 0.18 | 0.29 | 92% | 0.05 |
| 0.10 | 447.2 | 440.4 | 85% | 0.98 | 0.16 | 0.27 | 97% | 0.02 |
| 0.08 | 587.5 | 585.5 | 97% | 1.00 | 0.15 | 0.25 | 99% | 0.01 |

**Southeast Asia** (126 sessions, 3.7 languages per session)

| Threshold | Clusters | Mini | Clips in mini | Homog. | Compl. | V | Pure clips | q |
|---|---|---|---|---|---|---|---|---|
| 0.25 | 11.2 | 9.0 | 2% | 0.20 | 0.10 | 0.09 | 22% | 3.99 |
| 0.22 | 25.9 | 21.2 | 6% | 0.45 | 0.21 | 0.25 | 43% | 2.02 |
| 0.20 | 47.0 | 39.5 | 11% | 0.67 | 0.25 | 0.34 | 62% | 1.23 |
| 0.18 | 82.6 | 72.6 | 20% | 0.82 | 0.25 | 0.36 | 78% | 0.70 |
| 0.16 | 142.2 | 130.0 | 34% | 0.91 | 0.22 | 0.34 | 88% | 0.34 |
| 0.15 | 183.0 | 170.5 | 42% | 0.93 | 0.20 | 0.32 | 91% | 0.21 |
| 0.14 | 232.9 | 221.0 | 52% | 0.95 | 0.19 | 0.30 | 93% | 0.12 |
| 0.12 | 355.4 | 346.9 | 73% | 0.98 | 0.17 | 0.27 | 97% | 0.03 |
| 0.10 | 498.1 | 494.0 | 89% | 0.99 | 0.15 | 0.25 | 99% | 0.02 |
| 0.08 | 626.8 | 625.7 | 98% | 1.00 | 0.14 | 0.24 | 100% | 0.01 |

**South Asia** (95 sessions, 4.5 languages per session) - the best V of any corpus, 0.41 at 0.18

| Threshold | Clusters | Mini | Clips in mini | Homog. | Compl. | V | Pure clips | q |
|---|---|---|---|---|---|---|---|---|
| 0.25 | 9.5 | 7.4 | 2% | 0.20 | 0.23 | 0.16 | 18% | 4.69 |
| 0.22 | 21.3 | 17.2 | 5% | 0.43 | 0.28 | 0.30 | 37% | 2.72 |
| 0.20 | 37.3 | 30.5 | 9% | 0.60 | 0.29 | 0.36 | 54% | 1.60 |
| 0.18 | 65.7 | 55.7 | 15% | 0.78 | 0.29 | 0.41 | 73% | 0.89 |
| 0.16 | 114.6 | 101.2 | 27% | 0.88 | 0.26 | 0.38 | 84% | 0.44 |
| 0.15 | 148.4 | 133.8 | 34% | 0.91 | 0.24 | 0.37 | 88% | 0.29 |
| 0.14 | 191.3 | 176.6 | 44% | 0.94 | 0.22 | 0.35 | 91% | 0.15 |
| 0.12 | 302.5 | 290.5 | 66% | 0.97 | 0.19 | 0.31 | 96% | 0.05 |
| 0.10 | 443.7 | 437.0 | 85% | 0.99 | 0.17 | 0.29 | 99% | 0.02 |
| 0.08 | 584.6 | 582.7 | 96% | 1.00 | 0.16 | 0.27 | 100% | 0.01 |

**Mini clusters are the trap.** At 0.18 there are ~63 clusters under 8 clips holding 17% of the
session; at 0.15, ~147 holding 37%. Each gets its own (tiny) quota, and a cluster's FIRST pick is
never penalised, so a tight threshold quietly turns the penalty into "spend queries on outliers" -
the behaviour that wrecked AC/bas. Any tight-threshold variant should therefore also pool clusters
below some size into a single background group (one shared quota), or refuse quota to clusters under
a minimum size.

**AC close-up** (`runs/analysis/cluster_analysis/ac_clusters_025_vs_015.png`, sessions 31, 36, 50):
- At 0.25 one cluster holds 673, 502 and 683 of the 720 clips, with 10, 6 and 10 languages, and
  every target clip in the session.
- At 0.15 the target languages come apart into tight **single-language** clusters (Basaa in
  session 31: clusters of 58, 39, 25 and 19 clips, all pure; Hausa in session 36: 61, 43 and 28, all
  pure). Non-target languages behave differently: some stay mixed, and Kabyle in session 36
  scatters into crumbs (196 clusters under 8 clips hold 335 clips, mostly Kabyle).
- **Open question:** a language often appears as several separate tight islands (Hausa as three in
  session 36). These may be individual **speakers** rather than the language as a whole, since a
  Common Voice session can hold many clips from one prolific speaker and ECAPA embeddings still carry
  speaker and channel information. Checking needs `client_id` from each locale's `test.tsv`. If
  true, clustering finer would spread queries across speakers, not languages.

**CR, SEA and SA close-ups** (`<corpus>_clusters_025_vs_015.png`; sessions CR 46/111/120,
SEA 27/88/106, SA 51/74/75, chosen as the sessions with the most languages among those with 30+
target clips):
- At 0.25 the same picture as AC: one or two big clusters (160-690 clips) with 5-13 languages each.
- At 0.15 the targets separate **much less cleanly than in AC**, and it varies by language:
  - CR is the least clean. Its largest 0.15 clusters still mix 8-10 languages (session 46: 96 clips
    with 10 languages, 80 with 8), and Central Kurdish, Tatar and Kyrgyz end up mostly in crumbs. The
    related Turkic languages (Tatar, Kyrgyz, Bashkir, Kazakh) overlap.
  - SA: Marathi (86 clips, mostly Marathi) and Saraiki (pure clusters of 30, 26 and 13) separate;
    Dhivehi and Hindi stay mixed.
  - SEA: Cantonese (138 clips, nearly pure), Indonesian (30, 23) and the filler Basque (46, 28, pure)
    separate; Vietnamese is mostly crumbs.
- Crumbs (clusters under 8 clips) hold 305-422 of the 720 clips at 0.15 in these nine sessions, about
  half of each session.
- So how cleanly a language clusters is a property of the language and its neighbours. That is the
  kind of per-language descriptor (section 4, item 4) that could explain a per-language gamma
  effect, if seed tests show such effects are real.

### 3c. The two candidate thresholds at a glance (2026-09-16)

Homogeneity is a **score** (1 - H(language|cluster)/H(language)), not a share of clips. The shares are
"purity" (clips whose cluster is mostly their own language) and "pure clips" (clips in a cluster that
is >= 90% one language).

| Corpus | Homog. 0.18 | Homog. 0.16 | Purity 0.18 | Purity 0.16 | Pure clips 0.18 | Pure clips 0.16 | Clips in mini 0.18 | Clips in mini 0.16 | q 0.18 | q 0.16 |
|---|---|---|---|---|---|---|---|---|---|---|
| African Continent | 0.77 | 0.83 | 93% | 95% | 79% | 84% | 16% | 27% | 1.40 | 0.71 |
| **Caucasus** | **0.62** | 0.78 | 83% | 89% | **56%** | 71% | 17% | 29% | 0.95 | 0.45 |
| Southeast Asia | 0.82 | 0.91 | 92% | 96% | 78% | 88% | 20% | 34% | 0.70 | 0.34 |
| South Asia | 0.78 | 0.88 | 90% | 94% | 73% | 84% | 15% | 27% | 0.89 | 0.44 |
| **Pooled** | **0.74** | **0.84** | 89% | 93% | 70% | 81% | 17% | 29% | 0.95 | 0.47 |

0.18 reaches ~0.75-0.82 everywhere except Caucasus (0.62), whose Turkic languages (Tatar, Kyrgyz,
Bashkir, Kazakh) keep merging; 0.16 lifts Caucasus to 0.78 but puts 29% of each session into clusters
of fewer than 8 clips. Completeness peaks at 0.18, so that is the best overall match to language
structure; 0.16 buys homogeneity by shattering.

**First runs (started 2026-09-16):** OAL at threshold 0.18 with gamma 0.95, and threshold 0.16 with
gamma 0.975. Both give a second-pick multiplier m = gamma**(1/q) of about 0.95 - the same effective
strength as gamma 0.8 at threshold 0.25, the only setting that ever beat a baseline. Holding m fixed
isolates the change in cluster granularity from a change in penalty strength.

### 3d. A label-free rule for choosing the threshold (PROPOSED, not yet validated)

The thresholds above were explored post-hoc and scored against the **test stream's** true languages,
which is fine for understanding and unusable for choosing the method's hyper-parameter. This rule
fixes the provenance problem: it needs no labels at all, only cluster sizes and the query budget.

**The quota-unit rule: choose the threshold t such that the median clip's cluster quota q(t) = 1**,
where `q_c = k * sqrt(|c|) / sum_j sqrt(|c_j|)` is exactly the quantity the penalty already divides by.

Why q is the natural unit of this algorithm:
- **q >> 1** - a typical cluster is entitled to several of the k queries, so `gamma**(n_c/q_c)` almost
  never binds and the method degenerates to plain uncertainty sampling (threshold 0.25: q = 4.30, and
  only ~1 query in 8 actually changed).
- **q << 1** - every cluster is over quota on its FIRST pick, so the penalty becomes a hard
  one-pick-per-cluster rule and starts paying real uncertainty to reach crumbs (0.15: q = 0.29).
- **q ~ 1** - a typical cluster is entitled to about one query: the penalty is active but
  proportionate.

Where the rule lands (interpolated from the q column of section 3b), against the label-based V peak:

| Corpus | q = 1 falls at | V-measure peak (needs labels) |
|---|---|---|
| African Continent | ~0.17 | flat (~0.18) |
| Caucasus | ~0.18 (q = 0.95) | 0.15-0.16 |
| Southeast Asia | ~0.19 | 0.18 |
| South Asia | ~0.185 | 0.18 |
| **Pooled** | **0.18 (q = 0.95)** | **0.18** |

It reproduces the 0.18 choice on every corpus, and is **more stable across corpora** than the
label-based criterion, which wants 0.15-0.16 for Caucasus. It is also computable online, per session,
from the clusters the method already forms.

**Status: proposed, not validated.** To use it in a paper it must be computed on the **training
split**, not the test stream. Only the bootstrap heads of the train splits have ECAPA features today;
extracting a few thousand training clips per corpus is minutes of CPU and no downloads.

Two supporting options, also label-free:
  1. **Internal cluster validity** (silhouette or Calinski-Harabasz) over training-stream sessions,
     maximised over t - the conventional move, and an independent second opinion.
  2. **An explicit trade-off objective** `J(t) = coverage(t) - lambda * uncertainty_forgone(t)`. Both
     terms are already logged in `al_diversity.csv` (distinct clusters among the 8 picks; the measured
     uncertainty sacrifice, 0.4% at 0.18 vs 0.7% at 0.25). lambda would be fixed on training data.

**Empirical note (2026-09-16, final), recorded against the rule's own case.** With all four settings
complete on 20 languages, the rule's selection (0.18) is **not** the best setting. Ranked by mean IMLM
eq1 against baseline OAL: **0.16 / gamma 0.93 (-32%, 15/20, p = 0.0028)**, 0.16 / gamma 0.975 (-14%,
p = 0.083), 0.18 / gamma 0.95 (-10%, p = 0.26), 0.18 / gamma 0.86 (-4%, p = 0.65), 0.25 / gamma 0.8
(-1%, p = 0.45). So neither the V-measure peak (0.18) nor the q ~ 1 rule (0.18) picked the winner;
both select a *safer, weaker* operating point. Internal cluster-quality measures are a defensible way
to choose a threshold with clean provenance, **not** a proxy for downstream benefit - say so
explicitly if the rule is used in a paper.

**The q ~ 1 calibration also under-corrects.** `gamma = m**q` uses the median clip's quota, and the
*realised* disturbance it produces falls as clusters shrink: at 0.18 / gamma 0.95 the penalty changed
1.21 of 8 queries per session (0.4% of uncertainty given up), but at 0.16 / gamma 0.975 - matched on
paper - only 0.98 (0.2%). Restoring the disturbance at 0.16 (gamma 0.93: 1.30 queries, 0.7%) is what
produced the significant result. **Match the realised disturbance, not the nominal m**; both
quantities are in `al_diversity.csv` and need no labels.

No function can *prove* 0.18 optimal - the optimum depends on the embedding, corpus, budget k and the
penalty's form. What a paper needs is a selection rule with clean provenance plus a sensitivity curve
over thresholds (we have 0.25 / 0.20 / 0.18 / 0.16), showing the conclusion is not a knife-edge.

## 4. What this means for the extension (inference, untested)

- It explains the weak, noisy gamma results (`runs/reports/dq_oal__summary.txt`, `runs/reports/dq_oalcf__summary_*.txt`).
  The penalty mostly moves picks between the giant cluster, the medium one and the crumbs
  (outliers), not across languages, and changes only about 1 of 8 queries per session with OAL-CF.
- "Each language has its own best gamma" matches chance on CR OAL-CF (best of three beats baseline in
  6/7 languages; 6.1/7 expected if each gamma were a coin flip).
- A tighter threshold alone would change the mechanism: with 150 tiny clusters and 8 queries the
  penalty becomes "at most one query per fragment".
- Candidate directions:
  1. Group each session by **ECAPA's own predicted language** (the VoxLingua107 classifier head on the
     stored embeddings). Label-free, so fair online, and it matches the extension's intent.
  2. A different linkage (e.g. Ward on unit-normalised embeddings) or a fixed cluster count.
  3. **Adaptive gamma** per session from what is visible at query time, e.g. penalise only when the
     uncertain top-8 fall inside one cluster; no gamma to tune.
  4. Per-language descriptors worth measuring if gamma effects prove real: target concentration
     (share of a session's target clips in one cluster), confuser mixing (share of the target's
     cluster that is other languages), uncertainty spread (clusters among the plain top-8).

## 4b. How concentrated the plain top-8 already is, and when the penalty is wasted

Measured 2026-09-16 from the gamma = 1.0 control runs, whose logged picks *are* the plain
uncertainty top-8, and from the gamma runs of the same languages (`al_diversity.csv`).

| Run | Distinct clusters in the plain top-8 | All 8 in one cluster | Already 3+ clusters |
|---|---|---|---|
| cr/hy-AM (OAL) | 1.99 | 14% of sessions | 13% |
| ac/yo (OAL) | 1.65 | 59% | 18% |
| sa/dv (OAL) | 1.77 | 42% | 17% |
| cr/sr (OAL-CF) | 2.31 | 14% | 30% |

Sessions hold ~9-10 clusters, so plain uncertainty sampling normally draws its 8 queries from just
**2 of them**: the case the extension was built for is the common one.

**But the penalty is unconditional.** In sessions where the plain top-8 *already* spanned 3 or more
clusters, it still moved at least one pick in 60-100% of them: it fires on quota accounting, never on
whether diversity already exists. The price is small at a mild gamma and large at a harsh one - the
uncertainty given up, measured within each run (its own top-8 vs what it picked), pooled over the 20
languages with OAL-CF:

| gamma | Sessions where a pick moved | Distinct clusters | Uncertainty given up (those sessions / overall) |
|---|---|---|---|
| 0.8 | 38% | 2.51 | 1.2% / 0.5% |
| 0.5 | 58% | 2.89 | 3.2% / 2.0% |
| 0.2 | 73% | 3.47 | 6.0% / 4.6% |

That ordering matches the results exactly: gamma 0.8 (the cheapest intervention) is the only setting
that beat baseline OAL-CF, and gamma 0.2 (6% of uncertainty traded away) was the worst.

Four fixes follow, all of which keep the "don't disturb the top queries" intent explicit rather than
leaving it to gamma:
  1. **Fire conditionally**: skip the penalty when the plain top-k already spans >= m clusters.
  2. **Quota floor** `q_c = max(q_c, 1)`: every cluster's first two picks are free, so only genuine
     over-concentration is penalised.
  3. **Uncertainty guard**: never swap in a candidate whose u is below (1 - eps) times the u of the
     clip it displaces, which caps the sacrifice at eps (e.g. 5%) by construction.
  4. **Pool mini clusters** into one background group (section 3), so "a fresh cluster" cannot mean
     "an outlier with a free first pick".

## 5. Rules for acting on this

- **Noise first.** Run the model-seed control (`--seed`, baseline vs gamma 0.8, CR) before
  interpreting any per-language gamma effect.
- **Choose on the training split, never the test stream.** Any threshold or clusterer must be
  selected on training clips, by cluster-language agreement, never by test labels or test IMLM.
  Only the bootstrap head of each training split has features today; the CR/SEA/SA archives in
  `data/CrSeaSa/raw` hold every split (`docs/other/DEVIATIONS.md`), so a sample of training clips can be
  extracted without downloading anything. AC's raw folder still has to be checked.
- Fix the hypothesis before the runs that test it, and keep the gamma = 1 bit-identity check for
  any change to the query code.

## 6. Upstream quirk found on the way

`src/dataset.py:317-321` wraps positions with `% (len - 1)`, so in every run the stream's very last
clip is represented by the embedding of the dataset's first entry, the first bootstrap (train-order)
clip. One clip in 57k-114k; upstream behaviour, not changed. The rebuild scripts replicate it.
