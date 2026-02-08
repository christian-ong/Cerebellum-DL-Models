import torch

def rollout_ae_model(model, x0, steps, device="cpu"):
    """
    Roll out an AE-based model forward in time.

    Args:
        model: AELinearDynamics
        x0: torch.Tensor (state_dim,)
        steps: int

    Returns:
        X_hat: torch.Tensor (steps+1, state_dim)
    """
    model.eval()

    x = x0.to(device)
    X_hat = [x]

    with torch.no_grad():
        for _ in range(steps):
            x_in = x.unsqueeze(0)          # (1, state_dim)
            x_next, _, _ = model(x_in)    # (1, state_dim)
            x = x_next.squeeze(0)
            X_hat.append(x)

    return torch.stack(X_hat, dim=0)
