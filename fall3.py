import time
import zipfile
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.isotonic import IsotonicRegression
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# ----------------------------
# USER KNOBS (speed vs closeness)
# ----------------------------
SEED = 42
EARLY_STOP = 60

TOTAL_CORES = 176
N_PARALLEL_POV = 19
CORES_PER_POV = max(1, TOTAL_CORES // N_PARALLEL_POV)  # ~9
CORES_FOR_HH  = 48   # keep moderate to avoid weird oversubscription; try 64 if stable

OOF_FOLDS_POV = 3    # <<< key speed lever (closer=5, faster=3)

HASH_BUCKETS = 2**17
MIX_GRID = [1.0, 0.995, 0.99, 0.985, 0.98, 0.97, 0.95, 0.92, 0.90]  # slightly wider

# HH k-grid (smaller than your full)
K_LO, K_HI, K_N = 0.55, 1.55, 121

# ----------------------------
# Files
# ----------------------------
TRAIN_FEATURES = "train_hh_features.csv"
TRAIN_LABELS   = "train_hh_gt.csv"
TEST_FEATURES  = "test_hh_features.csv"
LABEL_COL = "cons_ppp17"

OUT_HH_CSV  = "predicted_household_consumption.csv"
OUT_POV_CSV = "predicted_poverty_distribution.csv"
OUT_ZIP     = "submission_hedge_v2.zip"

THRESHOLDS = [
    3.17, 3.94, 4.60, 5.26, 5.88, 6.47, 7.06, 7.70, 8.40,
    9.13, 9.87, 10.70, 11.62, 12.69, 14.03, 15.64, 17.76,
    20.99, 27.37
]
THRESH_COLS = [f"pct_hh_below_{t:.2f}" for t in THRESHOLDS]

# ----------------------------
# Timer / logger
# ----------------------------
t0 = time.time()
def log(msg: str):
    print(f"[{time.time()-t0:,.1f}s] {msg}", flush=True)

# ----------------------------
# Helpers
# ----------------------------
def safe_log1p(y):
    return np.log1p(np.maximum(np.asarray(y, dtype=np.float64), 0.0))

def safe_expm1(ylog):
    return np.maximum(np.expm1(np.asarray(ylog, dtype=np.float64)), 0.01)

def weighted_mape(y_true, y_pred, w):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    w      = np.asarray(w, dtype=np.float64)
    ape = np.abs(y_pred - y_true) / np.maximum(y_true, 1e-12)
    return float(np.sum(w * ape) / np.maximum(np.sum(w), 1e-12))

def enforce_monotonic_probs(p):
    return np.maximum.accumulate(p, axis=1)

# ----------------------------
# Feature engineering (CROSS)
# ----------------------------
def _coerce_yesno_01_frame(frame: pd.DataFrame) -> pd.DataFrame:
    s = frame.astype(str).apply(lambda col: col.str.strip().str.lower())
    s = s.replace({
        "yes": 1, "y": 1, "true": 1, "t": 1, "1": 1,
        "no": 0,  "n": 0, "false": 0, "f": 0, "0": 0,
        "": 0, "nan": 0, "none": 0
    })
    s = s.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.uint8)
    return s

def fe_cross(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    consumed_cols = [c for c in df.columns if c.startswith("consumed")]
    if consumed_cols:
        df[consumed_cols] = _coerce_yesno_01_frame(df[consumed_cols])
        df["total_consumed"] = df[consumed_cols].sum(axis=1).astype(np.float32)
    else:
        df["total_consumed"] = 0.0

    if "hsize" in df.columns:
        hs = pd.to_numeric(df["hsize"], errors="coerce").fillna(1.0).astype(np.float32)
        hs = np.maximum(hs, 1.0)
        df["consumed_per_person"] = (df["total_consumed"] / hs).astype(np.float32)

        if "utl_exp_ppp17" in df.columns:
            utl = pd.to_numeric(df["utl_exp_ppp17"], errors="coerce").fillna(0.0).astype(np.float32)
            df["utl_exp_ppp17_per_person"] = (utl / hs).astype(np.float32)
    else:
        df["consumed_per_person"] = df["total_consumed"].astype(np.float32)

    for c in ["utl_exp_ppp17", "utl_exp_ppp17_per_person", "hsize"]:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)
            df[f"log1p_{c}"] = np.log1p(np.maximum(v, 0.0)).astype(np.float32)

    # cross features (your test 9)
    if {"log1p_utl_exp_ppp17_per_person", "log1p_hsize"}.issubset(df.columns):
        df["x_logutlpp_x_loghsize"] = (
            df["log1p_utl_exp_ppp17_per_person"].astype(np.float32) *
            df["log1p_hsize"].astype(np.float32)
        )
        df["x_logutlpp_minus_loghsize"] = (
            df["log1p_utl_exp_ppp17_per_person"].astype(np.float32) -
            df["log1p_hsize"].astype(np.float32)
        )
    if {"utl_exp_ppp17_per_person", "consumed_per_person"}.issubset(df.columns):
        a = pd.to_numeric(df["utl_exp_ppp17_per_person"], errors="coerce").fillna(0).astype(np.float32)
        b = pd.to_numeric(df["consumed_per_person"], errors="coerce").fillna(0).astype(np.float32)
        df["x_utlpp_over_consumedpp"] = (a / np.maximum(b, 1e-3)).astype(np.float32)

    df["missing_count"] = df.isna().sum(axis=1).astype(np.float32)
    return df

def split_X_meta(df: pd.DataFrame):
    meta_cols = [c for c in ["survey_id", "hhid", "com", "weight", "strata"] if c in df.columns]
    meta = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)
    X = df.drop(columns=meta_cols, errors="ignore").copy()
    return X, meta

# ----------------------------
# Preprocess: medians + hashed cats
# ----------------------------
def detect_categorical_cols(X: pd.DataFrame):
    cat_cols = []
    for c in X.columns:
        if pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_string_dtype(X[c]):
            cat_cols.append(c)
    return cat_cols

def fit_medians(X_train: pd.DataFrame) -> pd.Series:
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    return X_train[num_cols].replace([np.inf, -np.inf], np.nan).median()

def apply_medians(X: pd.DataFrame, med: pd.Series) -> pd.DataFrame:
    X = X.copy()
    if len(med) == 0:
        return X
    num_cols = med.index
    X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)
    X[num_cols] = X[num_cols].fillna(med)
    return X

def make_hashed_cats(X: pd.DataFrame, cat_cols, n_buckets: int, seed: int = 17) -> pd.DataFrame:
    if not cat_cols:
        return pd.DataFrame(index=X.index)
    H = np.zeros((len(X), len(cat_cols) * 2), dtype=np.float32)
    for j, c in enumerate(cat_cols):
        s = X[c].astype("string").fillna("__MISSING__")
        h1 = pd.util.hash_pandas_object(s, index=False).astype(np.uint64).values
        h2 = pd.util.hash_pandas_object(s + f"__{seed}", index=False).astype(np.uint64).values
        b1 = (h1 % n_buckets).astype(np.float32) / float(n_buckets - 1) * 2.0 - 1.0
        b2 = (h2 % n_buckets).astype(np.float32) / float(n_buckets - 1) * 2.0 - 1.0
        H[:, 2*j]   = b1
        H[:, 2*j+1] = b2
    cols = []
    for c in cat_cols:
        cols += [f"hash_{c}_a", f"hash_{c}_b"]
    return pd.DataFrame(H, columns=cols, index=X.index)

def finalize_matrix(X_raw: pd.DataFrame, med: pd.Series) -> np.ndarray:
    X = X_raw.copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]) and X[c].dtype != bool:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = apply_medians(X, med)
    return X.astype(np.float32).to_numpy(copy=False)

# ----------------------------
# LightGBM params (speed-capped but close)
# ----------------------------
CLF_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    n_estimators=7000,
    num_leaves=255,
    min_child_samples=70,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_lambda=3.0,
    reg_alpha=0.0,
    n_jobs=CORES_PER_POV,
    verbose=-1,
    random_state=SEED,
)

REG_LOG_HUBER = dict(
    objective="huber",
    learning_rate=0.04,
    n_estimators=9000,
    num_leaves=255,
    min_child_samples=70,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_lambda=3.0,
    reg_alpha=0.0,
    n_jobs=CORES_FOR_HH,
    verbose=-1,
    random_state=SEED,
)

REG_LOG_MED = dict(
    objective="quantile",
    alpha=0.5,
    learning_rate=0.05,
    n_estimators=7000,
    num_leaves=255,
    min_child_samples=90,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_lambda=4.0,
    reg_alpha=0.0,
    n_jobs=CORES_FOR_HH,
    verbose=-1,
    random_state=SEED,
)

# ----------------------------
# Poverty: train one threshold (OOF iso + mix)
# ----------------------------
def train_one_threshold(t, X_tr, y_raw, w):
    y_bin = (y_raw < t).astype(np.int32)
    base_rate = float(np.sum(w * y_bin) / max(np.sum(w), 1e-12))

    pos_w = float(w[y_bin == 1].sum())
    neg_w = float(w[y_bin == 0].sum())
    spw = (neg_w / max(pos_w, 1e-12)) if pos_w > 0 else 1.0

    skf = StratifiedKFold(n_splits=OOF_FOLDS_POV, shuffle=True, random_state=SEED)
    oof = np.zeros(X_tr.shape[0], dtype=np.float64)

    for tr_idx, va_idx in skf.split(X_tr, y_bin):
        Xa, ya, wa = X_tr[tr_idx], y_bin[tr_idx], w[tr_idx]
        Xb = X_tr[va_idx]

        Xa2, Xes, ya2, yes, wa2, wes = train_test_split(
            Xa, ya, wa, test_size=0.15, random_state=SEED, stratify=ya
        )

        clf = lgb.LGBMClassifier(**{**CLF_PARAMS, "scale_pos_weight": spw})
        clf.fit(
            Xa2, ya2,
            sample_weight=wa2,
            eval_set=[(Xes, yes)],
            eval_sample_weight=[wes],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)]
        )
        oof[va_idx] = clf.predict_proba(Xb)[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof, y_bin, sample_weight=w)
    oof_cal = iso.predict(oof)

    best_mix, best_brier = 1.0, np.inf
    for mix in MIX_GRID:
        p_mix = mix * oof_cal + (1.0 - mix) * base_rate
        brier = float(np.sum(w * (p_mix - y_bin) ** 2) / max(np.sum(w), 1e-12))
        if brier < best_brier:
            best_brier, best_mix = brier, float(mix)

    # final train
    Xa2, Xes, ya2, yes, wa2, wes = train_test_split(
        X_tr, y_bin, w, test_size=0.15, random_state=SEED, stratify=y_bin
    )
    clf_full = lgb.LGBMClassifier(**{**CLF_PARAMS, "scale_pos_weight": spw})
    clf_full.fit(
        Xa2, ya2,
        sample_weight=wa2,
        eval_set=[(Xes, yes)],
        eval_sample_weight=[wes],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)]
    )

    return dict(t=float(t), iso=iso, base_rate=base_rate, mix=best_mix, model=clf_full)

def predict_poverty_probs(models, X):
    models = sorted(models, key=lambda d: d["t"])
    out = np.zeros((X.shape[0], len(models)), dtype=np.float64)
    for j, m in enumerate(models):
        raw = m["model"].predict_proba(X)[:, 1]
        cal = m["iso"].predict(raw)
        cal = m["mix"] * cal + (1.0 - m["mix"]) * m["base_rate"]
        out[:, j] = cal
    return enforce_monotonic_probs(out)

# ----------------------------
# HH bundle (close but capped)
# ----------------------------
def train_hh_bundle(X_tr, y_raw, w):
    y_log = safe_log1p(y_raw)

    idx = np.arange(X_tr.shape[0])
    idx_fit, idx_tune = train_test_split(idx, test_size=0.20, random_state=SEED)

    X_fit, X_tune = X_tr[idx_fit], X_tr[idx_tune]
    w_fit, w_tune = w[idx_fit], w[idx_tune]
    y_fit = y_log[idx_fit]
    y_tune_raw = np.maximum(y_raw[idx_tune], 0.01)

    # train split for early stop
    tr_i, va_i = train_test_split(np.arange(X_fit.shape[0]), test_size=0.15, random_state=SEED)

    def fit_reg(params):
        m = lgb.LGBMRegressor(**params)
        m.fit(
            X_fit[tr_i], y_fit[tr_i],
            sample_weight=w_fit[tr_i],
            eval_set=[(X_fit[va_i], y_fit[va_i])],
            eval_sample_weight=[w_fit[va_i]],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)]
        )
        return m

    log("[HH] training huber...")
    m_h = fit_reg(REG_LOG_HUBER)
    log("[HH] training median...")
    m_m = fit_reg(REG_LOG_MED)

    p_h = safe_expm1(m_h.predict(X_tune))
    p_m = safe_expm1(m_m.predict(X_tune))
    p_g = np.sqrt(np.maximum(p_h, 0.01) * np.maximum(p_m, 0.01))

    cand = {"huber": np.maximum(p_h, 0.01), "median": np.maximum(p_m, 0.01), "geom": np.maximum(p_g, 0.01)}
    best_name, best_s = None, np.inf
    for name, p in cand.items():
        s = weighted_mape(y_tune_raw, p, w_tune)
        if s < best_s:
            best_s, best_name = s, name
    log(f"[HH] best base on tune: {best_name} (wmape={best_s:.6f})")

    # retrain on full once each (close to your running script, but still capped)
    def fit_full(params):
        X_a2, X_es, y_a2, y_es, w_a2, w_es = train_test_split(
            X_tr, y_log, w, test_size=0.15, random_state=SEED
        )
        m = lgb.LGBMRegressor(**params)
        m.fit(
            X_a2, y_a2,
            sample_weight=w_a2,
            eval_set=[(X_es, y_es)],
            eval_sample_weight=[w_es],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)]
        )
        return m

    log("[HH] retraining full huber + median...")
    mh = fit_full(REG_LOG_HUBER)
    mm = fit_full(REG_LOG_MED)

    # calibration on tune (using the full models)
    def base_pred(Xp):
        a = np.maximum(safe_expm1(mh.predict(Xp)), 0.01)
        b = np.maximum(safe_expm1(mm.predict(Xp)), 0.01)
        if best_name == "huber":
            return a
        if best_name == "median":
            return b
        return np.sqrt(a * b)

    p_tune = base_pred(X_tune)
    lp = np.log(np.maximum(p_tune, 1e-12))
    ly = np.log(np.maximum(y_tune_raw, 1e-12))

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(lp, ly, sample_weight=w_tune)

    p_iso_tune = np.exp(iso.predict(lp))

    ks = np.exp(np.linspace(np.log(K_LO), np.log(K_HI), int(K_N)))
    best_k, best_k_s = 1.0, np.inf
    for kk in ks:
        s = weighted_mape(y_tune_raw, np.maximum(kk * p_iso_tune, 0.01), w_tune)
        if s < best_k_s:
            best_k_s, best_k = s, float(kk)
    log(f"[HH] best k={best_k:.6f} (wmape={best_k_s:.6f})")

    return dict(mh=mh, mm=mm, best_name=best_name, iso=iso, k=best_k)

def predict_hh(bundle, X):
    mh, mm = bundle["mh"], bundle["mm"]
    best_name = bundle["best_name"]
    iso, k = bundle["iso"], bundle["k"]

    a = np.maximum(safe_expm1(mh.predict(X)), 0.01)
    b = np.maximum(safe_expm1(mm.predict(X)), 0.01)

    if best_name == "huber":
        p = a
    elif best_name == "median":
        p = b
    else:
        p = np.sqrt(a * b)

    lp = np.log(np.maximum(p, 1e-12))
    p_iso = np.exp(iso.predict(lp))
    return np.maximum(k * p_iso, 0.01)

# ----------------------------
# MAIN
# ----------------------------
def main():
    log("Loading train/test...")
    train_feat = pd.read_csv(TRAIN_FEATURES)
    train_lbl  = pd.read_csv(TRAIN_LABELS)
    test_feat  = pd.read_csv(TEST_FEATURES)

    train = train_feat.merge(train_lbl[["survey_id", "hhid", LABEL_COL]], on=["survey_id", "hhid"], how="inner")

    if "weight" not in train.columns or "weight" not in test_feat.columns:
        raise ValueError("Expected 'weight' in both train and test features")

    log("Feature engineering (cross)...")
    train = fe_cross(train)
    test_feat = fe_cross(test_feat)

    Xtr_df, meta_tr = split_X_meta(train.drop(columns=[LABEL_COL]))
    Xte_df, meta_te = split_X_meta(test_feat)

    y = train[LABEL_COL].values.astype(np.float64)
    w = train["weight"].values.astype(np.float64)

    log("Preprocessing: medians + hashed cats...")
    cat_cols = detect_categorical_cols(Xtr_df)
    med = fit_medians(Xtr_df.select_dtypes(include=[np.number]))

    Htr = make_hashed_cats(Xtr_df, cat_cols, n_buckets=HASH_BUCKETS, seed=17)
    Hte = make_hashed_cats(Xte_df, cat_cols, n_buckets=HASH_BUCKETS, seed=17)

    Xtr = finalize_matrix(pd.concat([Xtr_df.drop(columns=cat_cols, errors="ignore"), Htr], axis=1), med)
    Xte = finalize_matrix(pd.concat([Xte_df.drop(columns=cat_cols, errors="ignore"), Hte], axis=1), med)

    # POVERTY MODELS
    log(f"Training 19 poverty thresholds in parallel (OOF={OOF_FOLDS_POV}) ...")
    pov_models = Parallel(n_jobs=N_PARALLEL_POV, backend="loky", verbose=10)(
        delayed(train_one_threshold)(t, Xtr, y, w) for t in THRESHOLDS
    )

    log("Predicting poverty probs on test...")
    pov_probs = predict_poverty_probs(pov_models, Xte)

    log("Aggregating poverty rates per survey (weighted)...")
    test_sids = meta_te["survey_id"].values.astype(int)
    test_w    = meta_te["weight"].values.astype(np.float64)

    pov_rows = []
    for sid in sorted(np.unique(test_sids)):
        msk = (test_sids == sid)
        ww = test_w[msk]
        wsum = float(ww.sum())
        rates = (pov_probs[msk] * ww.reshape(-1, 1)).sum(axis=0) / max(wsum, 1e-12)
        row = {"survey_id": int(sid)}
        for j, col in enumerate(THRESH_COLS):
            row[col] = float(np.clip(rates[j], 0.0, 1.0))
        pov_rows.append(row)

    pov_df = pd.DataFrame(pov_rows, columns=["survey_id"] + THRESH_COLS).sort_values("survey_id").reset_index(drop=True)

    # HH MODEL
    log("Training HH bundle (huber+median + iso + k-grid)...")
    hh_bundle = train_hh_bundle(Xtr, y, w)

    log("Predicting HH on test...")
    hh_pred = predict_hh(hh_bundle, Xte)

    hh_sub = pd.DataFrame({
        "survey_id": meta_te["survey_id"].values.astype(int),
        "household_id": meta_te["hhid"].values.astype(int),
        "cons_ppp17": hh_pred.astype(np.float64),
    }).sort_values(["survey_id", "household_id"]).reset_index(drop=True)

    log("Writing CSVs...")
    hh_sub.to_csv(OUT_HH_CSV, index=False)
    pov_df.to_csv(OUT_POV_CSV, index=False)

    log("Zipping submission...")
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(OUT_HH_CSV, arcname=OUT_HH_CSV)
        z.write(OUT_POV_CSV, arcname=OUT_POV_CSV)

    log(f"DONE: {OUT_ZIP}")
    log(f"  {OUT_HH_CSV}: rows={len(hh_sub):,}")
    log(f"  {OUT_POV_CSV}: rows={len(pov_df):,}, cols={len(pov_df.columns)}")

if __name__ == "__main__":
    main()

