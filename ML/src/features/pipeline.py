import numpy as np

# NOTE convert to tabular, with group identifiaction to then create full mask segment in order
def create_tabular_from_time_series(
    X_list,
    Y_list,
    add_lags: bool = False,
    lags: tuple[int, ...] = (1, 2, 4, 8),
):
    """Convert list-of-recordings into a 2D tabular matrix for boosting.

    Expected input format (per recording r):
      - X_list[r] is a list/tuple of N feature time series, each length T_r
        OR a numpy array shaped (N, T_r) or (T_r, N)
      - Y_list[r] is the mask array length T_r (0/1 or ints)

    Output:
      - X_tab: (sum_r T_r, num_features)
      - y_tab: (sum_r T_r,)
      - group_id: (sum_r T_r,) integer recording id
      - time_id: (sum_r T_r,) time index within recording
      - feature_names: list[str]
    """
    X_rows = []
    y_rows = []
    g_rows = []
    t_rows = []
    feature_names = ['mean', 'med_mean',
            'q_std', 'med_q_std',
            'q_linelen', 'med_linelen',
            'q_env', 'med_q_env',
            'q_std_env', 'med_q_std_env',
            'ef', 'ef_recovery','med_ef','med_ef_recovery',
            'prop_delta', 'prop_alpha', 'prop_beta', 'prop_gamma',
            'q_P_tot', 'med_q_P_tot', 'slopes'
            ]

    for g, (Xr, yr) in enumerate(zip(X_list, Y_list)):
        yr = np.asarray(yr).astype(int).ravel()
        T = len(yr)

        # Convert Xr to array shape (T, N)
        if isinstance(Xr, (list, tuple)):
            # list of N arrays length T
            Xr_arr = np.vstack([np.asarray(f).ravel() for f in Xr]).T  # (T, N)
        else:
            Xr_arr = np.asarray(Xr)
            if Xr_arr.ndim != 2:
                raise ValueError(f"X[{g}] must be 2D-like. Got shape {Xr_arr.shape}")
            # If (N, T) convert to (T, N)
            if Xr_arr.shape[0] != T and Xr_arr.shape[1] == T:
                Xr_arr = Xr_arr.T
            if Xr_arr.shape[0] != T:
                raise ValueError(f"X[{g}] time length mismatch: mask T={T}, X shape={Xr_arr.shape}")

        N = Xr_arr.shape[1]
        if feature_names is None:
            feature_names = [f"f{i:02d}" for i in range(N)]

        cols = [Xr_arr]
        names = feature_names.copy()

        if add_lags:
            for L in lags:
                lagged = np.full_like(Xr_arr, np.nan, dtype=float)
                if L < T:
                    lagged[L:, :] = Xr_arr[:-L, :]
                cols.append(lagged)
                names.extend([f"{fn}_lag{L}" for fn in feature_names])

        Xr_feat = np.concatenate(cols, axis=1).astype(float)  # (T, N*(1+len(lags)))
        X_rows.append(Xr_feat)
        y_rows.append(yr)
        g_rows.append(np.full(T, g, dtype=int))
        t_rows.append(np.arange(T, dtype=int))

    X_tab = np.concatenate(X_rows, axis=0)
    y_tab = np.concatenate(y_rows, axis=0)
    group_id = np.concatenate(g_rows, axis=0)
    time_id = np.concatenate(t_rows, axis=0)
    return X_tab, y_tab, group_id, time_id, names



def build_tabular(X, Y, config):
    add_lags = config.get(add_lags, False)
    lags = config.get("lags", (1, 2))
    return create_tabular_from_time_series(X, Y, add_lags, lags)