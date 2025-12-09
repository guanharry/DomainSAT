import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from scipy.stats import ks_2samp, chi2_contingency

# For Mann-Whitney U Test (a.k.a. Wilcoxon Rank-Sum)
from scipy.stats import mannwhitneyu

# For Cramér–von Mises two-sample test
from scipy.stats import cramervonmises_2samp  

# For classifier-based Methods
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# For Autoencoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# For MMD
from sklearn.metrics.pairwise import rbf_kernel

# For Wasserstein_distance
from scipy.stats import wasserstein_distance

# For KL Divergence
from scipy.special import rel_entr

# For JS Divergence
from scipy.spatial.distance import jensenshannon  

# For PCA visualization in 2D space
from sklearn.decomposition import PCA

# For UMAP visualization in 2D space
try:
    import umap  # pip install umap-learn
except ImportError:
    umap = None



###############################################################################
####### Section 1: Statistic-Based Methods
###############################################################################

###############################################################################
# Kolmogorov–Smirnov two-sample test
###############################################################################
def run_ks_test(source_df, target_df, p_thresh=0.05):
    """
    Kolmogorov–Smirnov two-sample test run column-wise.
    - Uses only columns common to both dataframes.
    - Ensure values are numeric and drops NaNs.
    - Skips columns where either side has fewer than 2 valid samples.
    """
    results = []
    common_cols = sorted(set(source_df.columns) & set(target_df.columns))

    for col in common_cols:
        # Ensure numeric, drop NaNs
        source = pd.to_numeric(source_df[col], errors="coerce").dropna()
        target = pd.to_numeric(target_df[col], errors="coerce").dropna()

        # Need at least a few samples on each side for a meaningful test
        if len(source) < 2 or len(target) < 2:
            continue

        try:
            stat, pval = ks_2samp(source, target)
        except Exception:
            # If something wrong happens for this column, just skip it
            continue

        shift = "Yes" if pval < p_thresh else "No"
        results.append({
            "Feature": col,
            "Method": "KS Test",
            "Statistic": stat,
            "P-Value": pval,
            "Shift Detected": shift
        })

    return pd.DataFrame(results)



###############################################################################
# Mann-Whitney U Test (a.k.a. Wilcoxon Rank-Sum)
###############################################################################
def run_mannwhitney_test(source_df, target_df, p_thresh=0.05):
    """
    Mann-Whitney U Test (a.k.a. Wilcoxon Rank-Sum Test).
    Tests whether two distributions differ in location (median shift).
    Run column-wise on numeric features shared by both dataframes.
    """
    results = []
    common_cols = sorted(set(source_df.columns) & set(target_df.columns))

    for col in common_cols:
        # Ensure numeric and drop NaNs
        x = pd.to_numeric(source_df[col], errors="coerce").dropna()
        y = pd.to_numeric(target_df[col], errors="coerce").dropna()

        # Need at least a few samples on each side
        if len(x) < 2 or len(y) < 2:
            continue

        try:
            stat, pval = mannwhitneyu(x, y, alternative="two-sided")
        except Exception:
            # If something wrong happens for this column, just skip it
            continue

        results.append({
            "Feature": col,
            "Method": "Mann-Whitney U",
            "Statistic": stat,
            "P-Value": pval,
            "Shift Detected": "Yes" if pval < p_thresh else "No"
        })

    return pd.DataFrame(results)


###############################################################################
# Cramér–von Mises two-sample test
###############################################################################
def run_cramervonmises_test(source_df, target_df, p_thresh=0.05):
    """
    Cramér–von Mises two-sample test.
    Run column-wise on numeric features shared by both dataframes.
    """
    results = []
    common_cols = sorted(set(source_df.columns) & set(target_df.columns))

    for col in common_cols:
        # Ensure numeric and drop NaNs
        x = pd.to_numeric(source_df[col], errors="coerce").dropna()
        y = pd.to_numeric(target_df[col], errors="coerce").dropna()

        # Need at least a few samples on each side
        if len(x) < 2 or len(y) < 2:
            continue

        try:
            result = cramervonmises_2samp(x, y)
        except Exception:
            # If someting wrong happens, skip it
            continue

        stat = result.statistic
        pval = result.pvalue

        results.append({
            "Feature": col,
            "Method": "Cramer-von Mises",
            "Statistic": stat,
            "P-Value": pval,
            "Shift Detected": "Yes" if pval < p_thresh else "No"
        })

    return pd.DataFrame(results)


###############################################################################
# Chi-square Test (categorical data)
###############################################################################
def run_chi2_test(source_df, target_df, p_thresh=0.05):
    """
    Chi-square test on categorical features.
    - Uses columns common to both dataframes.
    - Treats each column as categorical (string).
    - Builds a 2-column contingency table: Source vs Target.
    """
    results = []
    common_cols = sorted(set(source_df.columns) & set(target_df.columns))

    for col in common_cols:
        # Cast to string to treat values as categories
        source = source_df[col].astype(str)
        target = target_df[col].astype(str)

        # Build contingency table: rows = categories, columns = {Source, Target}
        contingency = (
            pd.crosstab(index=source, columns="Source")
            .join(pd.crosstab(index=target, columns="Target"), how="outer")
            .fillna(0)
        )

        # If there are fewer than 2 categories, or total count is 0, skip
        if contingency.shape[0] < 2 or contingency.values.sum() == 0:
            continue

        try:
            stat, pval, _, _ = chi2_contingency(contingency)
        except Exception:
            # e.g., if scipy complains about degenerate table
            continue

        results.append({
            "Feature": col,
            "Method": "Chi-square Test",
            "Statistic": stat,
            "P-Value": pval,
            "Shift Detected": "Yes" if pval < p_thresh else "No"
        })

    return pd.DataFrame(results)




###############################################################################
####### Section 2: Distance-Based Methods
###############################################################################

###############################################################################
# MMD (Maximum Mean Discrepancy), single feature
###############################################################################
def run_mmd_test(source_df, target_df, mmd_thresh=0.01, kernel="rbf", gamma=1.0):
    """
    Maximum Mean Discrepancy (MMD) using an RBF kernel (per-feature, 1D).
    Measures the difference in distributions between source and target.

    Parameters
    ----------
    source_df, target_df : pd.DataFrame
        DataFrames to compare.
    mmd_thresh : float
        Threshold above which shift is considered detected.
    kernel : str
        Currently only 'rbf' is supported (kept for future extensibility).
    gamma : float
        RBF kernel width parameter.
    """
    results = []
    common_cols = sorted(set(source_df.columns) & set(target_df.columns))

    for col in common_cols:
        # Ensure numeric and drop NaNs
        x = pd.to_numeric(source_df[col], errors="coerce").dropna().values.reshape(-1, 1)
        y = pd.to_numeric(target_df[col], errors="coerce").dropna().values.reshape(-1, 1)

        # Need at least a few samples on each side
        if x.shape[0] < 2 or y.shape[0] < 2:
            continue

        try:
            if kernel != "rbf":
                # For now we only support RBF
                continue

            K_xx = rbf_kernel(x, x, gamma=gamma)
            K_yy = rbf_kernel(y, y, gamma=gamma)
            K_xy = rbf_kernel(x, y, gamma=gamma)

            #mmd = K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean()
            mmd2 = K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean()
            mmd = float(np.sqrt(max(mmd2, 0.0)))

        except Exception:
            # If kernel computation fails for this column, skip it
            continue

        results.append({
            "Feature": col,
            "Method": "MMD",
            "Statistic": mmd,
            "P-Value": None,
            "Shift Detected": "Yes" if mmd > mmd_thresh else "No"
        })

    return pd.DataFrame(results)


###############################################################################
# MMD (Maximum Mean Discrepancy), multivariant
###############################################################################
def run_mmd_multivar_test(source_df, target_df, mmd_thresh=0.01, gamma=1.0):
    """
    Multivariate MMD using an RBF kernel over all common numeric features.
    Returns a single-row DataFrame with Feature='All'.
    """
    results = []

    # 1. Select common columns
    common_cols = sorted(set(source_df.columns) & set(target_df.columns))
    if not common_cols:
        return pd.DataFrame(results)

    # 2. Ensure numeric and drop rows with NaNs
    X_src = source_df[common_cols].apply(pd.to_numeric, errors="coerce")
    X_tgt = target_df[common_cols].apply(pd.to_numeric, errors="coerce")

    X_src = X_src.dropna(axis=0, how="any")
    X_tgt = X_tgt.dropna(axis=0, how="any")

    if X_src.shape[0] < 2 or X_tgt.shape[0] < 2:
        return pd.DataFrame(results)

    try:
        K_xx = rbf_kernel(X_src.values, X_src.values, gamma=gamma)
        K_yy = rbf_kernel(X_tgt.values, X_tgt.values, gamma=gamma)
        K_xy = rbf_kernel(X_src.values, X_tgt.values, gamma=gamma)

        #mmd = K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean()
        mmd2 = K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean()
        mmd = float(np.sqrt(max(mmd2, 0.0)))

    except Exception:
        return pd.DataFrame(results)

    results.append({
        "Feature": "All",
        "Method": "MMD (Multivariate)",
        "Statistic": mmd,
        "P-Value": None,
        "Shift Detected": "Yes" if mmd > mmd_thresh else "No"
    })

    return pd.DataFrame(results)


###############################################################################
# Multivariate MMD (using Random Fourier Features (RFF)), good for large data
###############################################################################
# Use for the RFF-based MMD
def _median_bandwidth_gamma_matrix(X, Y, max_pairs=50_000, random_state=42):
    """
    Median heuristic for RBF gamma using random cross-pairs
    from two matrices X and Y (rows = samples, cols = features).
    """
    rng = np.random.default_rng(random_state)
    nX, nY = X.shape[0], Y.shape[0]
    if nX == 0 or nY == 0:
        return 1.0

    max_possible = nX * nY
    k = min(max_pairs, max_possible)
    ix = rng.integers(0, nX, size=k, endpoint=False)
    iy = rng.integers(0, nY, size=k, endpoint=False)

    # pairwise distances for the sampled pairs
    d = np.linalg.norm(X[ix] - Y[iy], axis=1)
    d = d[d > 0]

    if d.size == 0:
        med = 1.0
    else:
        med = np.median(d)

    if not np.isfinite(med) or med <= 0:
        med = 1.0

    sigma2 = med ** 2
    return 1.0 / (2.0 * sigma2)


def _mmd_rff_matrix(X, Y, gamma, n_features=2048, random_state=42):
    """
    RFF-based MMD estimate on matrices X, Y (rows = samples, cols = features).
    Returns MMD (sqrt(MMD^2); non-negative scalar).
    """
    rng = np.random.default_rng(random_state)
    n_dim = X.shape[1]

    # Random Fourier feature parameters
    W = rng.normal(loc=0.0, scale=np.sqrt(2 * gamma), size=(n_dim, n_features)).astype(np.float32)
    b = rng.uniform(0, 2 * np.pi, size=(n_features,)).astype(np.float32)

    def rff(Z):
        proj = Z @ W + b  # [n, m]
        return np.sqrt(2.0 / n_features) * np.cos(proj)

    phiX = rff(X)
    phiY = rff(Y)

    diff = phiX.mean(axis=0) - phiY.mean(axis=0)
    mmd2 = float(np.dot(diff, diff))
    return float(np.sqrt(max(mmd2, 0.0)))


def run_mmd_rff_multivar(source_df, target_df, dist_thresh=0.01, n_features=2048, max_pairs=50_000, random_state=42):
    """
    Multivariate MMD using Random Fourier Features (RFF),

    - Uses all common numeric features.
    - Computes gamma via median heuristic on cross-pairs.
    - Returns a single-row DataFrame with Feature="All".

    Parameters
    ----------
    source_df, target_df : pd.DataFrame
        Input tables (rows = samples, columns = features).
    dist_thresh : float
        Threshold for shift detection.
    n_features : int
        Number of random Fourier features.
    max_pairs : int
        Max number of cross-pairs used for median heuristic.
    random_state : int
        Seed for reproducibility.
    """
    results = []

    # Common columns
    common_cols = sorted(set(source_df.columns) & set(target_df.columns))
    if not common_cols:
        return pd.DataFrame(results)

    # Ensure numeric
    X_src = source_df[common_cols].apply(pd.to_numeric, errors="coerce")
    X_tgt = target_df[common_cols].apply(pd.to_numeric, errors="coerce")

    # Drop columns that are all NaN in BOTH
    cols_to_keep = [
        c for c in common_cols
        if not (X_src[c].isna().all() and X_tgt[c].isna().all())
    ]
    if not cols_to_keep:
        return pd.DataFrame(results)

    X_src = X_src[cols_to_keep]
    X_tgt = X_tgt[cols_to_keep]

    # Drop rows with NaN
    X_src = X_src.dropna(axis=0, how="any")
    X_tgt = X_tgt.dropna(axis=0, how="any")

    # Need at least a few samples and at least 1 feature
    if X_src.shape[0] < 2 or X_tgt.shape[0] < 2 or X_src.shape[1] == 0:
        return pd.DataFrame(results)

    # standardize (optionally, but often helpful for embeddings)
    scaler = StandardScaler()
    X_src_z = scaler.fit_transform(X_src)
    X_tgt_z = scaler.transform(X_tgt)

    # Median heuristic for gamma
    gamma = _median_bandwidth_gamma_matrix(
        X_src_z,
        X_tgt_z,
        max_pairs=max_pairs,
        random_state=random_state,
    )

    # RFF-based MMD
    try:
        mmd_val = _mmd_rff_matrix(
            X_src_z,
            X_tgt_z,
            gamma=gamma,
            n_features=n_features,
            random_state=random_state,
        )
    except Exception:
        return pd.DataFrame(results)

    results.append({
        "Feature": "All",
        "Method": "MMD (Multivariate)",
        "Statistic": float(mmd_val),
        "P-Value": None,
        "Shift Detected": "Yes" if mmd_val > dist_thresh else "No",
    })

    return pd.DataFrame(results)




###############################################################################
# Mahalanobis Distance (multivariate, global)
###############################################################################
def run_mahalanobis_test(source_df, target_df, dist_thresh=1.0):
    """
    Global Mahalanobis distance between source and target means
    using a pooled covariance matrix over all common numeric features.

    Returns a single-row DataFrame with Feature='All'.
    """
    results = []

    try:
        # Common columns (sorted)
        common_cols = sorted(set(source_df.columns) & set(target_df.columns))
        if not common_cols:
            return pd.DataFrame(results)

        # Ensure numeric
        X_src = source_df[common_cols].apply(pd.to_numeric, errors="coerce")
        X_tgt = target_df[common_cols].apply(pd.to_numeric, errors="coerce")

        # Drop columns that are all NaN in BOTH source and target
        cols_to_keep = [
            c for c in common_cols
            if not (X_src[c].isna().all() and X_tgt[c].isna().all())
        ]
        if not cols_to_keep:
            return pd.DataFrame(results)

        X_src = X_src[cols_to_keep]
        X_tgt = X_tgt[cols_to_keep]

        # Drop rows with any NaN
        X_src = X_src.dropna(axis=0, how="any")
        X_tgt = X_tgt.dropna(axis=0, how="any")

        # Need at least 2 samples in each and at least 1 feature
        if X_src.shape[0] < 2 or X_tgt.shape[0] < 2 or X_src.shape[1] == 0:
            return pd.DataFrame(results)

        # Means
        mu_src = X_src.mean(axis=0).values
        mu_tgt = X_tgt.mean(axis=0).values
        diff = mu_src - mu_tgt  # shape (d,)

        # Pooled covariance from combined data
        X_combined = np.vstack([X_src.values, X_tgt.values])  # shape (n_src+n_tgt, d)
        # rowvar=False -> variables are columns
        Sigma = np.cov(X_combined, rowvar=False)

        # Mahalanobis distance using pseudo-inverse for stability
        Sigma_inv = np.linalg.pinv(Sigma)
        dist_sq = float(diff.T @ Sigma_inv @ diff)
        if dist_sq < 0:
            # Numerical guard: small negative due to precision
            dist_sq = 0.0
        dist = float(np.sqrt(dist_sq))

        results.append({
            "Feature": "All",
            "Method": "Mahalanobis Distance",
            "Statistic": dist,
            "P-Value": None,
            "Shift Detected": "Yes" if dist > dist_thresh else "No"
        })

    except Exception:
        pass

    return pd.DataFrame(results)




###############################################################################
# Wasserstein distance (per-feature, 1D)
###############################################################################
def run_wasserstein_test(source_df, target_df, dist_thresh=0.1):
    """
    1D Wasserstein Distance between source and target, computed per feature.
    Measures the 'cost' of transforming one marginal distribution into another.

    - Uses numeric values only (coerces to numeric, drops NaNs).
    - Runs column-wise over common columns.
    """
    results = []
    common_cols = sorted(set(source_df.columns) & set(target_df.columns))

    for col in common_cols:
        # Coerce to numeric and drop NaNs
        x = pd.to_numeric(source_df[col], errors="coerce").dropna()
        y = pd.to_numeric(target_df[col], errors="coerce").dropna()

        # Need at least a few samples on each side
        if len(x) < 2 or len(y) < 2:
            continue

        try:
            wd = wasserstein_distance(x, y)
        except Exception:
            # If fails for some reason on this feature, skip it
            continue

        results.append({
            "Feature": col,
            "Method": "Wasserstein",
            "Statistic": wd,
            "P-Value": None,
            "Shift Detected": "Yes" if wd > dist_thresh else "No"
        })

    return pd.DataFrame(results)


###############################################################################
# Wasserstein distance (Sliced_wasserstein, multivariant)
###############################################################################
def _sliced_wasserstein_matrix(X, Y, n_proj=256, random_state=42):
    """
    Sliced Wasserstein-1 distance between two point clouds X, Y.
    - X, Y: [n_samples, n_features]
    - n_proj: number of random 1D projections
    """
    rng = np.random.default_rng(random_state)
    d = X.shape[1]

    # Random directions on the unit sphere
    W = rng.normal(size=(d, n_proj)).astype(np.float32)
    W /= (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)

    Xp = X @ W   # [nX, n_proj]
    Yp = Y @ W   # [nY, n_proj]

    Wsum = 0.0
    for j in range(n_proj):
        xj = np.sort(Xp[:, j])
        yj = np.sort(Yp[:, j])
        n = min(len(xj), len(yj))

        # Align the shorter one by quantiles if lengths differ
        if len(xj) != n:
            q = np.linspace(0, 1, n, endpoint=False) + 0.5 / n
            xj = np.quantile(xj, q)
        if len(yj) != n:
            q = np.linspace(0, 1, n, endpoint=False) + 0.5 / n
            yj = np.quantile(yj, q)

        Wsum += np.mean(np.abs(xj - yj))

    return float(Wsum / n_proj)


def run_sliced_wasserstein_multivar(source_df, target_df, dist_thresh=0.1, n_proj=256, random_state=42):
    """
    Multivariate Sliced Wasserstein distance

    - Uses all common numeric features.
    - Projects data onto n_proj random 1D directions.
    - Computes 1D Wasserstein in each direction and averages.

    Returns a single-row DataFrame with Feature="All".
    """
    results = []

    # Common columns
    common_cols = sorted(set(source_df.columns) & set(target_df.columns))
    if not common_cols:
        return pd.DataFrame(results)

    # Ensure numeric
    X_src = source_df[common_cols].apply(pd.to_numeric, errors="coerce")
    X_tgt = target_df[common_cols].apply(pd.to_numeric, errors="coerce")

    # Drop columns that are all NaN in BOTH
    cols_to_keep = [
        c for c in common_cols
        if not (X_src[c].isna().all() and X_tgt[c].isna().all())
    ]
    if not cols_to_keep:
        return pd.DataFrame(results)

    X_src = X_src[cols_to_keep]
    X_tgt = X_tgt[cols_to_keep]

    # Drop rows with NaN
    X_src = X_src.dropna(axis=0, how="any")
    X_tgt = X_tgt.dropna(axis=0, how="any")

    # Need enough samples and at least 1 feature
    if X_src.shape[0] < 2 or X_tgt.shape[0] < 2 or X_src.shape[1] == 0:
        return pd.DataFrame(results)

    # 5. Standardize (optionally, but often stabilizes distances)
    scaler = StandardScaler()
    X_src_z = scaler.fit_transform(X_src)
    X_tgt_z = scaler.transform(X_tgt)

    # 6. Sliced Wasserstein
    try:
        swd_val = _sliced_wasserstein_matrix(
            X_src_z,
            X_tgt_z,
            n_proj=n_proj,
            random_state=random_state,
        )
    except Exception:
        return pd.DataFrame(results)

    results.append({
        "Feature": "All",
        "Method": "Wasserstein (Multivariate)",
        "Statistic": float(swd_val),
        "P-Value": None,
        "Shift Detected": "Yes" if swd_val > dist_thresh else "No",
    })

    return pd.DataFrame(results)

###############################################################################
# KL Divergence (per-feature, via histograms)
###############################################################################
def run_kl_test(source_df, target_df, kl_thresh=0.01, bins=30):
    """
    Per-feature KL Divergence using histogram approximation.

    KL(P || Q) is asymmetric; here we treat:
      P = source distribution
      Q = target distribution

    Implementation details:
    - Uses a shared histogram range [min(source, target), max(source, target)].
    - Adds small smoothing to avoid zeros before computing KL.
    """
    results = []
    common_cols = sorted(set(source_df.columns) & set(target_df.columns))

    for col in common_cols:
        # Ensure numeric and drop NaNs
        x = pd.to_numeric(source_df[col], errors="coerce").dropna()
        y = pd.to_numeric(target_df[col], errors="coerce").dropna()

        # Need at least a few points in each
        if len(x) < 2 or len(y) < 2:
            continue

        range_min = min(x.min(), y.min())
        range_max = max(x.max(), y.max())

        # If all values are identical, histogram is degenerate -> skip
        if range_min == range_max:
            continue

        try:
            # Histograms with shared range
            p_hist, _ = np.histogram(x, bins=bins, range=(range_min, range_max), density=True)
            q_hist, _ = np.histogram(y, bins=bins, range=(range_min, range_max), density=True)

            # Smoothing to avoid log(0) and zero probabilities
            eps = 1e-6
            p_hist = p_hist + eps
            q_hist = q_hist + eps

            # Normalize to sum to 1 (distributions)
            p_hist = p_hist / p_hist.sum()
            q_hist = q_hist / q_hist.sum()

            kl = float(np.sum(rel_entr(p_hist, q_hist)))
        except Exception:
            continue

        results.append({
            "Feature": col,
            "Method": "KL Divergence",
            "Statistic": kl,
            "P-Value": None,
            "Shift Detected": "Yes" if kl > kl_thresh else "No"
        })

    return pd.DataFrame(results)




###############################################################################
# JS Divergence (per-feature)
###############################################################################
def run_js_test(source_df, target_df, js_thresh=0.01, bins=30):
    """
    Per-feature Jensen–Shannon (JS) Divergence using histogram approximation.

    - Symmetric and bounded (0 to log(2) depending on base).
    - Here we square jensenshannon distance to get JS divergence.
    - Uses common columns, numeric-only values, and shared histogram range.
    """
    results = []
    common_cols = sorted(set(source_df.columns) & set(target_df.columns))

    for col in common_cols:
        # Coerce to numeric and drop NaNs
        source = pd.to_numeric(source_df[col], errors="coerce").dropna()
        target = pd.to_numeric(target_df[col], errors="coerce").dropna()

        # Need at least a few samples on each side
        if len(source) < 2 or len(target) < 2:
            continue

        hist_range = (min(source.min(), target.min()),
                      max(source.max(), target.max()))

        # Degenerate case: all values identical
        if hist_range[0] == hist_range[1]:
            continue

        try:
            # Histograms with shared range
            source_hist, _ = np.histogram(
                source, bins=bins, range=hist_range, density=True
            )
            target_hist, _ = np.histogram(
                target, bins=bins, range=hist_range, density=True
            )

            # Convert to proper probability distributions
            eps = 1e-6
            source_hist = source_hist + eps
            target_hist = target_hist + eps

            source_hist = source_hist / source_hist.sum()
            target_hist = target_hist / target_hist.sum()

            # jensenshannon returns the distance; square to get divergence
            stat = float(jensenshannon(source_hist, target_hist) ** 2)
        except Exception:
            continue

        results.append({
            "Feature": col,
            "Method": "JS Divergence",
            "Statistic": stat,
            "P-Value": None,
            "Shift Detected": "Yes" if stat > js_thresh else "No"
        })

    return pd.DataFrame(results)




##################################################################################
########## Section 3: Classifier-Based Methods
##################################################################################

###############################################################################
# Domain Classifier Shift Test (based on accuracy)
###############################################################################
def run_domain_classifier_acc(source_df, target_df, acc_thresh=0.6, random_state=42):
    """
    Train a classifier to distinguish source vs. target samples.
    High classification accuracy → high likelihood of domain shift.

    Steps:
    - Select common columns.
    - Convert all to numeric (drop non-numeric/empty columns).
    - Drop rows with NaN values.
    - Standardize features.
    - Run 5-fold CV Logistic Regression.
    """
    results = []

    try:
        # 1. Common columns (sorted for reproducibility)
        common_cols = sorted(set(source_df.columns) & set(target_df.columns))
        if not common_cols:
            return pd.DataFrame(results)

        # 2. Build combined dataset
        X = pd.concat([source_df[common_cols], target_df[common_cols]], axis=0)

        # Convert to numeric (invalid parses -> NaN)
        X = X.apply(pd.to_numeric, errors="coerce")

        # Drop columns that are entirely NaN
        X = X.dropna(axis=1, how="all")
        if X.shape[1] == 0:
            return pd.DataFrame(results)

        # 3. Drop rows containing any NaN
        mask = X.notna().all(axis=1)
        X = X[mask]

        # Create domain labels aligned with the mask
        y = np.array([0] * len(source_df) + [1] * len(target_df))
        y = y[mask.values]

        # Need at least two classes (rare corner case)
        if len(np.unique(y)) < 2:
            return pd.DataFrame(results)

        # 4. Normalize features
        X_scaled = StandardScaler().fit_transform(X)

        # 5. 5-fold CV domain classifier
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
        acc = cross_val_score(clf, X_scaled, y, cv=5, scoring="accuracy").mean()

        # 6. Append result
        results.append({
            "Feature": "All",
            "Method": "Domain Classifier",
            "Statistic": float(acc),
            "P-Value": None,
            "Shift Detected": "Yes" if acc > acc_thresh else "No"
        })

    except Exception:
        # Silent fail -> empty result
        pass

    return pd.DataFrame(results)



###############################################################################
# Domain Classifier Shift Test (based on AUC)
###############################################################################
def run_domain_classifier(source_df, target_df, auc_thresh=0.6, random_state=42):
    """
    Domain classifier using ROC-AUC as the statistic.

    - Labels: 0 = source, 1 = target
    - Statistic: 5-fold cross-validated ROC-AUC
    - Shift decision: AUC > auc_thresh
    """

    results = []

    try:
        # Common columns
        common_cols = sorted(set(source_df.columns) & set(target_df.columns))
        if not common_cols:
            return pd.DataFrame(results)

        # Combine data and coerce to numeric
        X = pd.concat([source_df[common_cols], target_df[common_cols]], axis=0)
        X = X.apply(pd.to_numeric, errors="coerce")

        # Drop columns that are all NaN
        X = X.dropna(axis=1, how="all")
        if X.shape[1] == 0:
            return pd.DataFrame(results)

        # Drop rows with any NaN
        mask = X.notna().all(axis=1)
        X = X[mask]

        # Labels: 0 = source, 1 = target
        y = np.array([0] * len(source_df) + [1] * len(target_df))
        y = y[mask.values]

        # Need both classes present
        if len(np.unique(y)) < 2:
            return pd.DataFrame(results)

        # Standardize features
        X_scaled = StandardScaler().fit_transform(X)

        # 5-fold CV ROC-AUC
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
        auc = float(
            cross_val_score(
                clf,
                X_scaled,
                y,
                cv=5,
                scoring="roc_auc",
            ).mean()
        )

        results.append({
            "Feature": "All",
            "Method": "Domain Classifier",
            "Statistic": auc,
            "P-Value": None,
            "Shift Detected": "Yes" if auc > auc_thresh else "No",
        })

    except Exception:
        pass

    return pd.DataFrame(results)




###############################################################################
# C2ST with Logistic Regression
###############################################################################

def run_c2st_logistic_classifier(
    source_df,
    target_df,
    auc_thresh=0.6,
    random_state=42,
):
    """
    Classifier Two-Sample Test (C2ST) using Logistic Regression.

    - Labels: 0 = source, 1 = target
    - Statistic: 5-fold cross-validated ROC-AUC (AUC)
    - Shift decision: based on AUC > auc_thresh
    """
    results = []

    try:
        # Common columns
        common_cols = sorted(set(source_df.columns) & set(target_df.columns))
        if not common_cols:
            return pd.DataFrame(results)

        # Build combined dataset and coerce to numeric
        X = pd.concat([source_df[common_cols], target_df[common_cols]], axis=0)
        X = X.apply(pd.to_numeric, errors="coerce")

        # Drop columns that are entirely NaN
        X = X.dropna(axis=1, how="all")
        if X.shape[1] == 0:
            return pd.DataFrame(results)

        # Drop rows with any NaN
        mask = X.notna().all(axis=1)
        X = X[mask]

        # Domain labels: 0 = source, 1 = target
        y = np.array([0] * len(source_df) + [1] * len(target_df))
        y = y[mask.values]

        # Need at least two classes
        if len(np.unique(y)) < 2:
            return pd.DataFrame(results)

        # Normalize features
        X_scaled = StandardScaler().fit_transform(X)

        # 5-fold CV ROC-AUC
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
        auc = float(
            cross_val_score(
                clf,
                X_scaled,
                y,
                cv=5,
                scoring="roc_auc",
            ).mean()
        )

        results.append({
            "Feature": "All",
            "Method": "C2ST (Logistic Regression)",
            "Statistic": auc,
            "P-Value": None,
            "Shift Detected": "Yes" if auc > auc_thresh else "No",
        })

    except Exception:
        pass

    return pd.DataFrame(results)



###############################################################################
# C2ST with Random Forest
###############################################################################

def run_c2st_forest_classifier(
    source_df,
    target_df,
    auc_thresh=0.6,
    random_state=42,
    n_estimators=200,
    max_depth=None,
):
    """
    Classifier Two-Sample Test (C2ST) using Random Forest.

    - Labels: 0 = source, 1 = target
    - Statistic: cross-validated ROC-AUC (AUC)
    - Shift decision: based on AUC > auc_thresh
    """
    results = []

    try:
        # Common columns
        common_cols = sorted(set(source_df.columns) & set(target_df.columns))
        if not common_cols:
            return pd.DataFrame(results)

        # Build combined dataset and coerce to numeric
        X = pd.concat([source_df[common_cols], target_df[common_cols]], axis=0)
        X = X.apply(pd.to_numeric, errors="coerce")

        # Drop columns that are entirely NaN
        X = X.dropna(axis=1, how="all")
        if X.shape[1] == 0:
            return pd.DataFrame(results)

        # Drop rows with any NaN
        mask = X.notna().all(axis=1)
        X = X[mask]

        # Domain labels: 0 = source, 1 = target
        y = np.array([0] * len(source_df) + [1] * len(target_df))
        y = y[mask.values]

        # Need at least two classes
        if len(np.unique(y)) < 2:
            return pd.DataFrame(results)

        # Normalize features (optional for trees, but keeps consistency)
        X_scaled = StandardScaler().fit_transform(X)

        # 5-fold CV ROC-AUC
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        auc = float(
            cross_val_score(
                clf,
                X_scaled,
                y,
                cv=5,
                scoring="roc_auc",
            ).mean()
        )

        results.append({
            "Feature": "All",
            "Method": "C2ST (Random Forest)",
            "Statistic": auc,
            "P-Value": None,
            "Shift Detected": "Yes" if auc > auc_thresh else "No",
        })

    except Exception:
        pass

    return pd.DataFrame(results)




##################################################################################
# Autoencoder-based Shift Detection
##################################################################################
def run_autoencoder_test(source_df, target_df, recon_thresh=0.01, random_state=42):
    """
    Train an autoencoder on source data and compare reconstruction error on target data.
    Higher increase in target reconstruction error → more likely data distribution has shifted.

    Steps:
    - Select common columns.
    - Coerce to numeric and clean NaNs.
    - Standardize using source statistics.
    - Train a small MLP autoencoder on source only.
    - Compare reconstruction error (we use relative error) on source vs target.
    """
    results = []

    try:
        # Common columns (sorted)
        common_cols = sorted(set(source_df.columns) & set(target_df.columns))
        if not common_cols:
            return pd.DataFrame(results)

        # Ensure numeric
        X_src = source_df[common_cols].apply(pd.to_numeric, errors="coerce")
        X_tgt = target_df[common_cols].apply(pd.to_numeric, errors="coerce")

        # Drop columns that are all NaN in BOTH source and target
        cols_to_keep = [
            c for c in common_cols
            if not (X_src[c].isna().all() and X_tgt[c].isna().all())
        ]
        if not cols_to_keep:
            return pd.DataFrame(results)

        X_src = X_src[cols_to_keep]
        X_tgt = X_tgt[cols_to_keep]

        # Drop rows with any NaN, separately for src and tgt
        X_src = X_src.dropna(axis=0, how="any")
        X_tgt = X_tgt.dropna(axis=0, how="any")

        # Need at least some samples and at least 1 feature
        if X_src.shape[0] < 2 or X_tgt.shape[0] < 2 or X_src.shape[1] == 0:
            return pd.DataFrame(results)

        # Normalize using source statistics only (to avoid target leakage)
        scaler = StandardScaler()
        X_src_scaled = scaler.fit_transform(X_src)
        X_tgt_scaled = scaler.transform(X_tgt)

        # Train a simple autoencoder (1 hidden layer MLP)
        ae = MLPRegressor(
            hidden_layer_sizes=(8,),
            activation="relu",
            max_iter=1000,
            random_state=random_state,
        )
        ae.fit(X_src_scaled, X_src_scaled)

        # Reconstruction errors
        err_src = mean_squared_error(X_src_scaled, ae.predict(X_src_scaled))
        err_tgt = mean_squared_error(X_tgt_scaled, ae.predict(X_tgt_scaled))

        # stat = float(abs(err_tgt - err_src))  # sbsolute reconstruction error
        # Instead of absolute error, use ralative error which is more interpretable. 
        # E.g., stat = 0.2 → target recon error is 20% higher than source
        eps = 1e-8
        stat = float(max(0.0, (err_tgt - err_src) / (err_src + eps)))  # relative change
        
        # Decide if Shift detected
        results.append({
            "Feature": "All",
            "Method": "Autoencoder",
            "Statistic": stat,
            "P-Value": None,
            "Shift Detected": "Yes" if stat > recon_thresh else "No"
        })

    except Exception:
        # On any failure, return empty result
        pass

    return pd.DataFrame(results)





###################################################################################
# Plot Histograms with KDE (Kernel Density Estimation) Overlay 
###################################################################################
def plot_histograms(source, target, col, bins="fd", add_kde=True):
    """
    Plot overlaid histograms (with automatic bin selection) and optional KDE curves
    for a single feature from source and target datasets.

    Parameters
    ----------
    source : pd.Series or array-like
        Source feature values.
    target : pd.Series or array-like
        Target feature values.
    col : str
        Feature name (for titles/labels).
    bins : str or int, default "fd"
        Bin rule or count. "fd" (Freedman–Diaconis) adapts to data spread & size.
    add_kde : bool, default True
        Whether to overlay KDE curves for source and target.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object ready for st.pyplot(fig).
    """
    # Coerce to numeric and drop NaNs (robust to mixed-type input)
    source = pd.to_numeric(source, errors="coerce").dropna()
    target = pd.to_numeric(target, errors="coerce").dropna()

    # If not enough data, return an empty-ish figure
    if len(source) < 1 or len(target) < 1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(f"Histogram: {col} (insufficient data)")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        return fig

    # Combine to determine a common binning strategy
    all_data = pd.concat([source, target])
    if all_data.nunique() <= 1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(f"Histogram: {col} (constant values)")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        return fig

    # Automatic bin edges using a common range for both distributions
    bin_edges = np.histogram_bin_edges(all_data, bins=bins)

    fig, ax = plt.subplots(figsize=(8, 4))

    # Histograms with density normalization for fair comparison
    ax.hist(source, bins=bin_edges, alpha=0.5, density=True, label="Source")
    ax.hist(target, bins=bin_edges, alpha=0.5, density=True, label="Target")

    # Optional KDE overlay
    if add_kde:
        try:
            xs = np.linspace(bin_edges[0], bin_edges[-1], 200)

            if len(source) > 1:
                kde_source = gaussian_kde(source)
                ax.plot(xs, kde_source(xs), linewidth=1.5, label="Source KDE")

            if len(target) > 1:
                kde_target = gaussian_kde(target)
                ax.plot(xs, kde_target(xs), linewidth=1.5, linestyle="--", label="Target KDE")
        except Exception:
            # If KDE fails (e.g., singular covariance for very degenerate data), just skip it
            pass

    ax.set_title(f"Histogram: {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()

    return fig


###############################################################################
# Plot Histograms for a Single Feature
###############################################################################
def plot_histograms_simple(source, target, col, bins=30):
    """
    Plot overlaid histograms of a feature from source and target datasets.

    Returns:
        fig (matplotlib.figure.Figure): ready for st.pyplot(fig)
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(source, bins=bins, alpha=0.5, label="Source")
    ax.hist(target, bins=bins, alpha=0.5, label="Target")

    ax.set_title(f"Histogram: {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()

    return fig



###################################################################################
# 2D Embedding Visualization with UMAP
###################################################################################
def plot_umap_2d(
    source_df,
    target_df,
    max_points=2000,
    random_state=42,
    n_neighbors=15,
    min_dist=0.1,
):
    """
    Visualize source and target samples in a shared 2D UMAP embedding space.

    Parameters
    ----------
    source_df, target_df : pd.DataFrame
        Input datasets.
    max_points : int, default 2000
        Maximum number of total points (source + target) to plot.
    random_state : int, default 42
        Random seed for reproducibility.
    n_neighbors : int, default 15
        UMAP n_neighbors parameter.
    min_dist : float, default 0.1
        UMAP min_dist parameter.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure ready for st.pyplot(fig).
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # Check UMAP availability
    if umap is None:
        ax.set_title("UMAP not installed (pip install umap-learn)")
        return fig

    # Common columns
    common_cols = sorted(set(source_df.columns) & set(target_df.columns))
    if not common_cols:
        ax.set_title("UMAP 2D Embedding (no common columns)")
        return fig

    # Ensure numeric
    X_src = source_df[common_cols].apply(pd.to_numeric, errors="coerce")
    X_tgt = target_df[common_cols].apply(pd.to_numeric, errors="coerce")

    # Drop columns that are all NaN in BOTH
    cols_to_keep = [
        c for c in common_cols
        if not (X_src[c].isna().all() and X_tgt[c].isna().all())
    ]
    if not cols_to_keep:
        ax.set_title("UMAP 2D Embedding (no usable numeric features)")
        return fig

    X_src = X_src[cols_to_keep]
    X_tgt = X_tgt[cols_to_keep]

    # Drop rows with any NaN
    X_src = X_src.dropna(axis=0, how="any")
    X_tgt = X_tgt.dropna(axis=0, how="any")

    if X_src.shape[0] < 2 or X_tgt.shape[0] < 2 or X_src.shape[1] == 0:
        ax.set_title("UMAP 2D Embedding (insufficient data)")
        return fig

    # Combine and create domain labels
    X = pd.concat([X_src, X_tgt], axis=0)
    y_domain = np.array([0] * X_src.shape[0] + [1] * X_tgt.shape[0])  # 0=source, 1=target

    # Optional subsampling for speed and visual clarity
    n_total = X.shape[0]
    rng = np.random.RandomState(random_state)
    if n_total > max_points:
        idx = rng.choice(n_total, size=max_points, replace=False)
        X = X.iloc[idx]
        y_domain = y_domain[idx]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # UMAP embedding
    try:
        embedder = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )
        Z = embedder.fit_transform(X_scaled)
    except Exception:
        ax.set_title("UMAP 2D Embedding (UMAP failed)")
        return fig

    # Split back into source / target
    Z_src = Z[y_domain == 0]
    Z_tgt = Z[y_domain == 1]

    # Scatter plot
    ax.scatter(Z_src[:, 0], Z_src[:, 1], s=15, alpha=0.7, label="Source")
    ax.scatter(Z_tgt[:, 0], Z_tgt[:, 1], s=15, alpha=0.7, label="Target", marker="x")

    ax.set_title("UMAP 2D Embedding")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend()
    fig.tight_layout()

    return fig


###################################################################################
# 2D Embedding Visualization with PCA
###################################################################################
def plot_pca_2d(
    source_df,
    target_df,
    max_points=5000,
    random_state=42,
):
    """
    Visualize source and target samples in a shared 2D PCA space.

    Parameters
    ----------
    source_df, target_df : pd.DataFrame
        Input datasets.
    max_points : int, default 5000
        Maximum number of total points (source + target) to plot.
    random_state : int, default 42
        Random seed for reproducible subsampling.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure ready for st.pyplot(fig).
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # Common columns
    common_cols = sorted(set(source_df.columns) & set(target_df.columns))
    if not common_cols:
        ax.set_title("PCA 2D Embedding (no common columns)")
        return fig

    # Ensure numeric
    X_src = source_df[common_cols].apply(pd.to_numeric, errors="coerce")
    X_tgt = target_df[common_cols].apply(pd.to_numeric, errors="coerce")

    # Drop columns that are all NaN in BOTH
    cols_to_keep = [
        c for c in common_cols
        if not (X_src[c].isna().all() and X_tgt[c].isna().all())
    ]
    if not cols_to_keep:
        ax.set_title("PCA 2D Embedding (no usable numeric features)")
        return fig

    X_src = X_src[cols_to_keep]
    X_tgt = X_tgt[cols_to_keep]

    # Drop rows with any NaN
    X_src = X_src.dropna(axis=0, how="any")
    X_tgt = X_tgt.dropna(axis=0, how="any")

    if X_src.shape[0] < 2 or X_tgt.shape[0] < 2 or X_src.shape[1] == 0:
        ax.set_title("PCA 2D Embedding (insufficient data)")
        return fig

    # Combine and create domain labels
    X = pd.concat([X_src, X_tgt], axis=0)
    y_domain = np.array([0] * X_src.shape[0] + [1] * X_tgt.shape[0])  # 0=source, 1=target

    # Optional subsampling for speed and visual clarity
    n_total = X.shape[0]
    rng = np.random.RandomState(random_state)
    if n_total > max_points:
        idx = rng.choice(n_total, size=max_points, replace=False)
        X = X.iloc[idx]
        y_domain = y_domain[idx]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA embedding
    try:
        pca = PCA(n_components=2, random_state=random_state)
        Z = pca.fit_transform(X_scaled)
    except Exception:
        ax.set_title("PCA 2D Embedding (PCA failed)")
        return fig

    # Split back into source / target
    Z_src = Z[y_domain == 0]
    Z_tgt = Z[y_domain == 1]

    # Scatter plot
    ax.scatter(Z_src[:, 0], Z_src[:, 1], s=15, alpha=0.7, label="Source")
    ax.scatter(Z_tgt[:, 0], Z_tgt[:, 1], s=15, alpha=0.7, label="Target", marker="x")

    ax.set_title("PCA 2D Embedding")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend()
    fig.tight_layout()

    return fig

