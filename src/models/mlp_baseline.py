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

        # ------------------------------------------------
        # 1. Build the Neural Network (Hidden Layers)
        # ------------------------------------------------
        layers = []
        
        # Input layer
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.SiLU())
        
        # Intermediate hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            
        self.net = nn.Sequential(*layers)
        
        # ------------------------------------------------
        # 2. Build the Output Head
        # ------------------------------------------------
        # This layer maps the hidden features back to the 2D physical state
        self.head = nn.Linear(hidden_dim, state_dim)
        
        # INITIALIZATION TRICK: 
        # We initialize the final layer to exactly zero. 
        # This means at Epoch 0, the network predicts "0 change" (x_{t+1} = x_t).
        # This gives the network a highly stable starting point.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    # ------------------------------------------------
    # Forward Pass (One-Step Prediction)
    # ------------------------------------------------
    def forward(self, x):
        """ Predicts the next state: x_{t+1} """
        
        # 1. Pass the current state through the network to get the "change" (delta)
        delta = self.head(self.net(x))
        
        # 2. Add the change to the current state (Residual / Euler integration)
        x_next = x + delta
        
        return x_next

    # ------------------------------------------------
    # Training Loss Calculation
    # ------------------------------------------------
    def compute_loss(self, x, x_next_true, future_x=None):
        
        # --- 1. One-Step Prediction Loss ---
        x_next_pred = self.forward(x)
        loss_state = nn.MSELoss()(x_next_pred, x_next_true)

        # --- 2. Multi-Step Rollout Loss ---
        loss_rollout = torch.tensor(0.0, device=x.device)
        
        # If we provided future trajectory data, calculate how well the model 
        # predicts multiple steps into the future using its own predictions.
        if future_x is not None and future_x.ndim == 3:
            
            # Decide how far into the future to look (cap it at rollout_horizon)
            horizon = min(self.rollout_horizon, future_x.shape[1])
            
            if horizon >= 2:
                curr_pred = x_next_pred 
                
                # Loop through the future steps
                for k in range(1, horizon):
                    
                    # Feed the model's LAST prediction back into itself
                    curr_pred = self.forward(curr_pred)
                    
                    # Compare the new prediction to the true future state
                    x_true_k = future_x[:, k, :]
                    loss_rollout += torch.mean((curr_pred - x_true_k)**2)
                
                # Average the loss over the number of steps
                loss_rollout = loss_rollout / float(horizon - 1)

        # --- 3. Total Loss ---
        # Note: We lowered the state loss to 0.5 to match your Koopman models!
        loss_total = 0.5 * loss_state + 0.1 * loss_rollout
        
        loss_dict = {
            "state": loss_state.item(),
            "rollout": loss_rollout.item(),
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