# Artificial Data Study — Generator Audit v1.0
Date: 2026-10-26
Status: pilot code audit; not a validated confirmatory protocol.

## Source files
`D4_extension/experiment_d1.py`, `experiment_d2_d3.py`, `experiment_d4.py` extracted from the previously supplied D4 extension package. Reproducible test: `python audit_generators.py`.

## Common setup
Labels y ~ Bernoulli(0.1), except oracle minority-only generation where y=1. Latent z has three coordinates. Six observed features are [z0, z1, z2, z0*z1, z0^2, sin(z2)] plus independent Gaussian noise N(0,0.08^2) per observed feature. These six features are dependent by construction, despite added noise.

## S1 / D1
Before transformations: z0 ~ N(1.1*y,1), z1 ~ N(-0.8*y,1), z2 ~ N(0,1), independently conditional on y. Thus the labels are generated first, and the classes overlap; there is no deterministic label function. The class prior is 0.1 in expectation, not exactly 10% in every sample.

## S2 / D2
Labels are generated as above. For each row angle ~ Uniform(0,2*pi), radius = (2.3 if y=1 else 0.55) + N(0,0.28^2), then z0=radius*cos(angle), z1=radius*sin(angle), z2 ~ N(0.4*y,1). Transformed features and Gaussian measurement noise follow. This is a ring/center geometry, with substantial separability and a nonlinear boundary in the first two coordinates. Radius may occasionally be negative due to Gaussian noise (low probability), so this should be described as noisy signed radial sampling rather than an exact annulus.

## S3 / D3
The unshifted generator is IDENTICAL to S1/D1, including RNG call sequence. Audit: across seeds 0–29, training X and y arrays matched exactly, element-for-element. Test-time shift adds 1.5 to latent z2 before constructing sin(z2), while labels and z0/z1 mechanisms remain unchanged. This is a nuisance-feature covariate shift, not a demonstrated change in the discriminative mechanism. A separate informative-feature shift should be introduced as a distinct condition if robustness to predictive shift is an intended claim.

## Audit measurements
For seeds 0–29 and n=1000, minority count: min 69, mean 99.5, max 119. For S3, mean observed sine feature across these seeds: training 0.00383, shifted test 0.60791. This is a descriptive shift check, not a hypothesis test.

## Important validity risks
1. Oracle minority-only samples are drawn from the known data-generating mechanism, granting privileged information unavailable to SMOTE and CTGAN; do not present as equal-information competitors.
2. Generator parameter values (class prior 0.1, shifts 1.1/-0.8, ring radii 2.3/0.55, measurement noise 0.08, shift 1.5) are design choices, not empirically calibrated values. Run sensitivity experiments or justify them as controlled experimental conditions.
3. Current D4 training has 1000 observations before splitting, 750 fit and 250 validation; with 10% minority expected, only about 75 minority fit observations. CTGAN quality and SMOTE neighbor selection may be sensitive to this.
4. D4 adds enough minority examples to approximately balance the fit set. This controls total added samples across methods, but not information access or generator complexity.
5. CTGAN generator uses `ctgan_minority` from D1, with `cuda=False`, while another file uses `enable_gpu=False`; pin package versions and test compatibility. Conditional-sampling validity and duplicate rates need explicit checks.
6. D4's `d4_paired_differences.csv` compares each condition with default-threshold baseline, not same-threshold baseline. For fair tuned-vs-tuned contrasts recompute pairings from `d4_per_seed.csv`.
7. In D4 `pr_auc` is computed with `average_precision_score`; label it Average Precision (AP), not trapezoidal PR-AUC.
8. A seed is a Monte Carlo repetition of a single synthetic DGP, not an independent real-world dataset. Inference must be limited to simulated conditions.

## Recommended decisions before rerunning
A. Freeze S1 and S2 formulas and label generation as versioned generators.
B. Rename S3 as nuisance covariate shift, and optionally add S4 informative covariate shift with a preregistered mechanism.
C. Test sample size, imbalance ratio, and shift/noise intensity through a small factorial sensitivity design; do not pick favorable seeds post hoc.
D. Add unit tests: same-seed reproducibility, no overlap between train/validation/test streams, minority-only label validity, generator moment checks, shift magnitude, and generated-feature finite values.
E. Separate generator-fidelity measures (MMD and dependency preservation) from downstream utility (AP and minority F1), and preserve a common held-out test set per repetition.
F. Treat previous D1–D4 scores as pilot results pending audited reruns.
