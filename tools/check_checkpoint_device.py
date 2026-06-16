import sys
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import torch
from src.eval.model_io import load_model
from src.eval.sweep_utils import compute_rollout_metrics

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/models/regression_dmd/closed_trig_medium/pmnwhae1/model.npz"

meta = np.load(MODEL_PATH, allow_pickle=True)
state_dim = int(meta['state_dim'].item()) if 'state_dim' in meta else int(np.asarray(meta['state_dim']).item())
system = meta.get('system', 'unknown')
print('Model file:', MODEL_PATH)
print('Saved state_dim:', state_dim, 'system:', system)

for device in ['cpu', 'cuda']:
    try:
        dev = 'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        print('\n--- Loading on', dev, '---')
        model, extras = load_model(model_name='regression_dmd', model_path=MODEL_PATH, data_path='.', state_dim=state_dim, system=system, device=dev)

        print('Model x_mean device/dtype:', getattr(model, 'x_mean', None).device if getattr(model,'x_mean',None) is not None else None, getattr(model, 'x_mean', None).dtype if getattr(model,'x_mean',None) is not None else None)
        print('x_scale device/dtype:', getattr(model, 'x_scale', None).device if getattr(model,'x_scale',None) is not None else None, getattr(model,'x_scale',None).dtype if getattr(model,'x_scale',None) is not None else None)
        print('psi_scale device/dtype:', getattr(model, 'psi_scale', None).device if getattr(model,'psi_scale',None) is not None else None, getattr(model,'psi_scale',None).dtype if getattr(model,'psi_scale',None) is not None else None)

        print('\nExpander named buffers:')
        if hasattr(model, 'expander') and model.expander is not None:
            for name, buf in model.expander.named_buffers():
                print(f'  {name}: device={buf.device}, dtype={buf.dtype}, shape={tuple(buf.shape)}')
        else:
            print('  (no expander)')

        print('\nModel main tensors:')
        for name in ['K_fitted','C_fitted','K_tilde_fitted','Phi_lift_fitted','Phi_state_fitted','Lambda_fitted']:
            val = getattr(model, name, None)
            if val is not None:
                try:
                    print(f'  {name}: device={val.device}, dtype={val.dtype}, shape={tuple(val.shape)}')
                except Exception:
                    print(f'  {name}: (non-tensor) type={type(val)}')

        # Try to run a quick rollout metrics if training data is available in the same dataset folder
        if 'data_path' in meta:
            data_path = str(meta['data_path'].item()) if hasattr(meta['data_path'], 'item') else str(meta['data_path'])
            print('\nAttempting rollout eval using data_path from checkpoint:', data_path)
            try:
                # load full dataset split npz path is expected to be a dataset directory; compute_rollout_metrics expects (T,N,d) array
                from src.data_generation.load_data import resolve_split_npz_path
                train_npz = resolve_split_npz_path(data_path, 'val')
                d = np.load(train_npz)
                X = d['X']
                # ensure shape (T,N,d)
                if X.ndim == 2:
                    X = X[:, None, :]
                res = compute_rollout_metrics(model, X, device=dev, eval_horizons=[10,20,100], max_trajs=5)
                print('Rollout metrics:', res)
            except Exception as e:
                print('Rollout eval failed:', e)

    except Exception as exc:
        print('Loading on', device, 'failed:', exc)

print('\nDone')
