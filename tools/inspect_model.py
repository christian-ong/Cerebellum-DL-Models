import sys
import os
import numpy as np


def inspect_npz(path):
    print(f"Loading: {path}")
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())
    print("Keys:", keys)

    def show_arr(k):
        a = data[k]
        print(f"- {k}: shape={getattr(a,'shape',None)} dtype={getattr(a,'dtype',None)} min={np.nanmin(a) if a.size else 'NA'} max={np.nanmax(a) if a.size else 'NA'} mean={np.nanmean(a) if a.size else 'NA'}")

    # Print common hankel keys
    hankel_keys = [k for k in ['hankel_mean','hankel_components','hankel_singular_values','history_scale'] if k in data]
    for k in hankel_keys:
        show_arr(k)

    # RBF keys
    rbf_keys = [k for k in ['rbf_centers','rbf_sigmas','state_scale'] if k in data]
    for k in rbf_keys:
        show_arr(k)

    # model matrices
    for k in ['K','C','K_tilde','U_r','W_reduced','Lambda','Phi_lift','Phi_state','Phi']:
        if k in data:
            show_arr(k)

    # scales
    for k in ['x_mean','x_scale','psi_scale']:
        if k in data:
            show_arr(k)

    # Basic hankel diagnostics
    if 'hankel_singular_values' in data:
        s = np.asarray(data['hankel_singular_values']).astype(float)
        s2 = s**2
        tot = s2.sum()
        print('singular values (first 10):', s[:10])
        print('energy ratios (first 10):', (s2 / tot)[:10])
        print('cum energy (first 10):', np.cumsum(s2) / tot)    

    if 'hankel_components' in data:
        C = np.asarray(data['hankel_components'])
        try:
            cond = np.linalg.cond(C)
        except Exception as e:
            cond = str(e)
        print('components shape:', C.shape, 'cond:', cond)

    # Try reconstructing a random H if possible using components+mean: generate a random z and decode
    if 'hankel_components' in data and 'hankel_mean' in data:
        comps = np.asarray(data['hankel_components'])
        mean = np.asarray(data['hankel_mean'])
        rank, history_dim = comps.shape
        z = np.random.randn(1, rank)
        H_hat_scaled = z @ comps + mean.reshape(1, -1)
        print('reconstructed head (first 5):', H_hat_scaled[0, :min(5, H_hat_scaled.shape[1])])

    data.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: inspect_model.py <path/to/model.npz>')
        sys.exit(1)
    inspect_npz(sys.argv[1])
