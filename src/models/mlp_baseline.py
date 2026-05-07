import torch
import torch.nn as nn

class MLP_BlackBox(nn.Module):
    def __init__(
        self,
        state_dim=2,
        hidden_dim=128,
        num_layers=4,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.rollout_horizon = 20

        # We use SiLU (Swish) activation. It is smooth and continuously differentiable, 
        # which performs much better on dynamical systems/physics than ReLU.
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, state_dim)
        
        # Initialize the final layer to zero. 
        # This ensures the network starts by predicting x_{t+1} = x_t
        # This provides a stable starting point (identity mapping) similar
        # to how you initialized Phi and Lambda near the identity.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        """
        Predicts the next state using a learned vector field.
        x_{t+1} = x_t + f_theta(x_t)
        """
        delta = self.head(self.net(x))
        x_next = x + delta
        return x_next

    def compute_loss(self, x, x_next_true, future_x=None):
        # 1. One-Step Prediction Loss
        x_next_pred = self.forward(x)
        loss_state = nn.MSELoss()(x_next_pred, x_next_true)

        # 2. Multi-step Rollout Loss
        loss_rollout = torch.tensor(0.0, device=x.device)
        
        if future_x is not None and future_x.ndim == 3:
            horizon = min(self.rollout_horizon, future_x.shape[1])
            
            if horizon >= 2:
                curr_pred = x_next_pred 
                
                for k in range(1, horizon):
                    curr_pred = self.forward(curr_pred)
                    x_true_k = future_x[:, k, :]
                    
                    loss_rollout += torch.mean((curr_pred - x_true_k)**2)
                
                loss_rollout = loss_rollout / float(horizon - 1)

        # Total Loss
        loss_total = loss_state + 0.1 * loss_rollout
        
        loss_dict = {
            "state": loss_state.item(),
            "rollout": loss_rollout.item(),
        }
        
        return (loss_total, loss_dict)

    def rollout(self, x0, steps):
        if not torch.is_tensor(x0):
            x0 = torch.tensor(
                x0,
                dtype=next(self.parameters()).dtype,
                device=next(self.parameters()).device,
            )

        if x0.ndim == 1:
            x = x0.unsqueeze(0)
        else:
            x = x0

        traj = [x.squeeze(0)]
        for _ in range(steps):
            x = self.forward(x)
            traj.append(x.squeeze(0))

        return torch.stack(traj)