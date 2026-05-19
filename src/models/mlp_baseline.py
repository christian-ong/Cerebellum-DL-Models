import torch
import torch.nn as nn

class MLP_BlackBox(nn.Module):
    """
    A standard Multi-Layer Perceptron (MLP) baseline for dynamical systems.
    It learns a discrete vector field (Euler step): x_{t+1} = x_t + f(x_t)
    """
    def __init__(
        self,
        state_dim=2,
        hidden_dim=64,
        num_layers=4,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.rollout_horizon = 20

        # --- NEW: Add a buffer for MaxAbs scaling ---
        self.register_buffer("state_scale", torch.ones(state_dim, dtype=torch.float32))

        # 1. Build the Neural Network (Hidden Layers)
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            
        self.net = nn.Sequential(*layers)
        
        # 2. Build the Output Head
        self.head = nn.Linear(hidden_dim, state_dim)
        
        # Initialize final layer to exactly zero
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    # --- NEW: Add the scaler fitting method ---
    def fit_state_scaler(self, x: torch.Tensor):
        """Calculates MaxAbs scale from training data."""
        max_abs = torch.max(torch.abs(x), dim=0)[0]
        max_abs[max_abs == 0] = 1.0 # Prevent division by zero
        self.state_scale.copy_(max_abs)

    def forward(self, x):
        """
        Input: Physical state x_t
        Output: Physical state x_{t+1}
        """
        # 1. Scale input down to roughly [-1.0, 1.0]
        x_norm = x / self.state_scale
        
        # 2. Network predicts the NORMALIZED change in state
        delta_norm = self.head(self.net(x_norm))
        
        # 3. Multiply the predicted change back up to physical scale, and add it
        x_next = x + (delta_norm * self.state_scale)
        
        return x_next

    # ------------------------------------------------
    # Training Loss Calculation
    # ------------------------------------------------
    def compute_loss(self, x, x_next_true, future_x=None):
        
        # --- 1. One-Step Prediction ---
        x_next_pred = self.forward(x)
        
        # Physical loss (for logging and absolute accuracy)
        loss_state = nn.MSELoss()(x_next_pred, x_next_true)
        
        # NEW: Normalized loss (for balanced, stable gradients!)
        x_next_pred_norm = x_next_pred / self.state_scale
        x_next_true_norm = x_next_true / self.state_scale
        loss_state_norm = nn.MSELoss()(x_next_pred_norm, x_next_true_norm)

        # --- 2. Multi-Step Rollout Loss ---
        loss_rollout = torch.tensor(0.0, device=x.device)
        loss_rollout_norm = torch.tensor(0.0, device=x.device) # Track normalized rollout too
        
        if future_x is not None and future_x.ndim == 3:
            horizon = min(self.rollout_horizon, future_x.shape[1])
            if horizon >= 2:
                curr_pred = x_next_pred 
                
                for k in range(1, horizon):
                    curr_pred = self.forward(curr_pred)
                    x_true_k = future_x[:, k, :]
                    
                    # Physical rollout
                    loss_rollout += torch.mean((curr_pred - x_true_k)**2)
                    
                    # Normalized rollout
                    curr_pred_norm = curr_pred / self.state_scale
                    x_true_k_norm = x_true_k / self.state_scale
                    loss_rollout_norm += torch.mean((curr_pred_norm - x_true_k_norm)**2)
                
                loss_rollout = loss_rollout / float(horizon - 1)
                loss_rollout_norm = loss_rollout_norm / float(horizon - 1)

        # --- 3. Total Loss ---
        # Add the normalized losses as the primary driver to stabilize the network!
        loss_total = (
              1.0 * loss_state_norm 
            + 1.0 * loss_rollout_norm
        )
        
        # We keep the physical losses in the dictionary ONLY for logging and printing to the console.
        loss_dict = {
            "state": loss_state.item(),
            "rollout": loss_rollout.item(),
            "state_norm": loss_state_norm.item(),
            "rollout_norm": loss_rollout_norm.item(),
        }
        
        return (loss_total, loss_dict)

    # ------------------------------------------------
    # Evaluation Rollout (For Plotting)
    # ------------------------------------------------
    def rollout(self, x0, steps):
        """ Used during evaluation to generate a long trajectory from an initial point. """
        
        # Ensure input is a tensor
        if not torch.is_tensor(x0):
            x0 = torch.tensor(
                x0,
                dtype=next(self.parameters()).dtype,
                device=next(self.parameters()).device,
            )

        # Ensure input has a batch dimension
        if x0.ndim == 1:
            x = x0.unsqueeze(0)
        else:
            x = x0

        # Store the starting point
        traj = [x.squeeze(0)]
        
        # Loop forward in time, saving the physical state at each step
        for _ in range(steps):
            x = self.forward(x)
            traj.append(x.squeeze(0))

        # Stack into a single tensor for plotting
        return torch.stack(traj)