# Artificial Data Generation Study — Generator v2 (pilot)

## Scope
Five controlled **evaluation scenarios**, not five artificial-data generation methods. No SMOTE, CTGAN or oracle data are generated here. The original D4 findings remain exploratory and are not combined numerically with this pilot.

## Mechanisms
- **S1:** y ~ Bernoulli(0.1), z ~ N(0,I3), z0 <- z0+1.1y, z1 <- z1-0.8y.
- **S2a:** y ~ Bernoulli(0.1), theta ~ U(0,2pi), r = 2.3 for minority or 0.55 for majority + N(0,0.28^2). z0=r cos(theta), z1=r sin(theta), z2 <- z2+0.4y.
- **S2b:** same as S2a but radial noise SD = 0.85, a provisional difficulty setting. It is **not** an independently derived sample-size or difficulty optimum.
- **S3a:** same train mechanism as S1; test z2 <- z2+1.5 before feature transforms (nuisance shift).
- **S3b:** same train mechanism as S1; test z0 <- z0+1.5 before feature transforms (informative-feature intervention). This is **not automatically strict covariate shift**: changing the class-conditional feature mechanism may change P(Y|X).

Observed features for all scenarios: [z0,z1,z2,z0*z1,z0^2,sin(z2)] + independent N(0,0.08^2) measurement noise per feature. Latent labels are drawn first. Generated sample arrays are independent across train/test, even when using the same numeric seed.

## Reproducibility
Requires Python, numpy, pandas, scikit-learn. Run:

    python -m unittest -v test_generators.py
    python pilot_v2.py --seeds 10 --rf-trees 80

Output: pilot_v2_per_seed.csv and pilot_v2_summary.csv. Seeds 0..9 are pre-specified for the diagnostic pilot. Independent SeedSequence child streams separate labels, latent features, geometry, and measurement noise. Production runs should record dependency versions, Git commit, CPU/GPU settings and actual seed list.

## Audit results
Six unit tests passed. Diagnostic baseline-only pilot found S2a near-ceiling RF performance (AP 0.9993), while S2b had RF AP 0.5878. S3b RF AP 0.2661 vs S1 0.3357. This is not a significance test and not evidence for any augmentation method.

## Required before main experiments
1. Verify S3a/S3b shift taxonomy mathematically; use 'feature intervention' unless conditional invariance is demonstrated.
2. Freeze scenario parameters *without inspecting SMOTE/CTGAN outcomes*; report that S2b=0.85 was chosen after baseline-only pilot.
3. Determine training sample-size and minority-rate grid with a budget/power or precision rationale.
4. Define paired replication design, test distribution, oracle's privileged information, and independent RNG streams for augmentation methods.
5. Define distribution fidelity, structural dependence, coverage/diversity, AP and F1 measurement plan. Specify MMD kernel/bandwidth from a reference-only protocol.
6. Distinguish the oracle simulator's full DGP access from any real-data simulation method.
7. Pre-register primary comparisons and multiplicity family; avoid treating seeds as independent datasets.
