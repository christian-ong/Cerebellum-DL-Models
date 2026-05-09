import torch
import torch.nn as nn

from src.models.expander import build_expander


class ML_DMD_BAND(nn.Module):
    def __init__(
        self,
        state_dim=2,
        expansion_degree=2,
        bias=True,
        sine_cosine_expansion=False,
        expansion_type="general",
        system=None,
        rbf_n_centers=50,
        rbf_center_selection="farthest",
        rbf_bandwidth_mode="knn",
        rbf_knn_k=5,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.expansion_type = expansion_type

        # ------------------------------------------------
        # Initialize basis expansion
        # ------------------------------------------------
        # This creates the lifted state representation z.
        # The expansion can now be either:
        #   - ManualExpansion  (general / specific)
        #   - RBFExpansion     (rbf)
        self.expander = build_expander(
            state_dim=state_dim,
            expansion_type=expansion_type,
            expansion_degree=expansion_degree,
            bias=bias,
            sine_cosine_expansion=sine_cosine_expansion,
            system=system,
            rbf_n_centers=rbf_n_centers,
            rbf_center_selection=rbf_center_selection,
            rbf_bandwidth_mode=rbf_bandwidth_mode,
            rbf_knn_k=rbf_knn_k,
        )

        # Public aliases used elsewhere in the model / training code
        self.expand_names = self.expander.expand_names
        self.state_indices = self.expander.state_indices
        self.expanded_dim = self.expander.expanded_dim

        self.latent_dim = self.expanded_dim

        # ------------------------------------------------
        # Structural penalty hyperparameters
        # ------------------------------------------------
        # Smooth activation for rotation-block penalties (act in [0,1])
        self.rotation_act_tol = 1e-3
        self.rotation_act_scale = 1e-3

        # ------------------------------------------------
        # Eigenvector matrix Φ
        # ------------------------------------------------
        self.Phi = nn.Parameter(
            torch.eye(self.latent_dim)
            + 0.001 * torch.randn(self.latent_dim, self.latent_dim)
        )

        # ------------------------------------------------
        # Tridiagonal Eigenvalue Matrix Λ
        # ------------------------------------------------
        self.eig_diag = nn.Parameter(torch.ones(self.latent_dim) + torch.randn(self.latent_dim) * 0.01)
        
        # Initialize cleanly into the rotation basin (perfectly anti-symmetric)
        super_init = torch.randn(self.latent_dim - 1) * 0.01
        self.eig_super = nn.Parameter(super_init)
        self.eig_sub = nn.Parameter(-super_init.clone().detach())

        # ------------------------------------------------
        # Buffers and Constraints
        # ------------------------------------------------
        self.max_abs_z_norm = 1e6
        self.rollout_horizon = 20

    # ------------------------------------------------
    # Matrix Accessors
    # ------------------------------------------------

    def get_Phi(self):
        """Returns Physical Phi for visualization and analysis."""
        return self.Phi

    def get_Phi_inv(self):
        """Returns Physical Phi_inv."""
        return torch.linalg.pinv(self.Phi, rcond=1e-6)

    def get_Lambda(self):
        Lambda = (
            torch.diag(self.eig_diag) + 
            torch.diag(self.eig_super, diagonal=1) + 
            torch.diag(self.eig_sub, diagonal=-1)
        )
        return Lambda

    def get_K(self):
        """Returns Physical K."""
        return self.Phi @ self.get_Lambda() @ self.get_Phi_inv()

    def get_eigenvalues(self):
        return torch.linalg.eigvals(self.get_Lambda())

    def _get_modal_coords(self, z):
        I_eps = 1e-6 * torch.eye(self.latent_dim, device=self.Phi.device, dtype=self.Phi.dtype)
        return torch.linalg.solve(self.Phi + I_eps, z.mT).mT

    def _step_modal(self, b):
        # FIX: Call the method get_Lambda()
        return b @ self.get_Lambda().mT

    def _modal_to_latent(self, b):
        z = b @ self.Phi.mT
        return torch.clamp(z, min=-self.max_abs_z_norm, max=self.max_abs_z_norm)
    
    # ------------------------------------------------
    # Forward pass
    # ------------------------------------------------
    def forward(self, x):
        z = self.expander.expand(x)
        b = self._get_modal_coords(z)
        b_next = self._step_modal(b)
        # Use your helper to get clamping for free!
        z_next = self._modal_to_latent(b_next) 
        return self.expander.de_expand(z_next)

    # ------------------------------------------------
    # Training loss
    # ------------------------------------------------
    def compute_loss(self, x, x_next_true, future_x=None):
        z = self.expander.expand(x)
        z_next_true = self.expander.expand(x_next_true)

        # Use the helpers!
        b_curr = self._get_modal_coords(z)
        b_next = self._step_modal(b_curr)
        z_next_pred = self._modal_to_latent(b_next)

        loss_lift = torch.mean((z_next_pred - self.expander.expand(x_next_true))**2)
        loss_state = torch.mean((self.expander.de_expand(z_next_pred) - x_next_true)**2)

        # 3. Rollout
        loss_rollout = torch.tensor(0.0, device=x.device)
        if future_x is not None and future_x.ndim == 3:
            horizon = min(self.rollout_horizon, future_x.shape[1])
            if horizon >= 2:
                # PRE-EXPAND FUTURE TARGETS (Massive speedup!)
                z_targets = self.expander.expand(future_x.reshape(-1, self.state_dim))
                z_targets = z_targets.reshape(x.shape[0], future_x.shape[1], -1)

                b_rollout = b_next
                for k in range(1, horizon):
                    b_rollout = self._step_modal(b_rollout)
                    z_pred_k = self._modal_to_latent(b_rollout)
                    
                    loss_rollout += torch.mean((z_pred_k - z_targets[:, k, :])**2)
                    loss_rollout += torch.mean((self.expander.de_expand(z_pred_k) - future_x[:, k, :])**2)
                loss_rollout = loss_rollout / float(horizon - 1)
        # 3. Structural Constraints (Phi)
        phi_phys = self.get_Phi()
        col_norms = torch.linalg.norm(phi_phys, dim=0)
        loss_unit_length = torch.mean((col_norms - 1.0) ** 2) # Use L2 for stable normalization

        # 4. Lambda Regularization (The "Softened" Physics Router)
        b = self.eig_super
        c = self.eig_sub
        d = self.eig_diag
        diff_diag = d[:-1] - d[1:]

        # Branch 1: Rotation Block (L2 for equality allows precision)
        dist_rot = (b + c)**2 + (diff_diag)**2          
        
        # Branch 2: Independent Modes (L1 for sparsity forces true zero)
        dist_indep = torch.abs(b) + torch.abs(c)

        dist_all = torch.min(torch.stack([dist_rot, dist_indep], dim=0), dim=0).values
        loss_manifold = torch.mean(dist_all)

        loss_same_sign = torch.mean(torch.relu(b * c))
        # Keep Overlap and Sparsity as L1 to force blocks to separate
        loss_overlap = torch.mean(torch.abs(b[:-1] * b[1:]) + torch.abs(c[:-1] * c[1:]))
        loss_sparsity = torch.mean(torch.abs(b)) + torch.mean(torch.abs(c))

        # 5. Annealing (Lower cap and slower ramp)
        start_weight = 0.0     
        end_weight = 0.05      # REDUCED: 0.5 -> 0.05 (Don't let structure drown out accuracy)
        warmup_epochs = 40.0   
        ramp_epochs = 200.0  

        if self.current_epoch < warmup_epochs:
            structural_weight = 0.0
        elif self.current_epoch < ramp_epochs:
            progress = (self.current_epoch - warmup_epochs) / (ramp_epochs - warmup_epochs)
            structural_weight = start_weight + (end_weight - start_weight) * progress
        else:
            structural_weight = end_weight

        # 6. Total Loss
        loss_total = (
            loss_lift
            + 0.5 * loss_state
            + 1.0 * loss_rollout                 
            + 1e-3 * loss_unit_length
            + structural_weight * loss_manifold      
            + structural_weight * loss_same_sign 
            + structural_weight * loss_overlap      
            + 1e-8 * loss_sparsity               # LOWERED: 1e-7 -> 1e-8
        )
        
        loss_dict = {
            "state": loss_state.item(),
            "lift": loss_lift.item(),
            "rollout": loss_rollout.item(),
            "unit": loss_unit_length.item(),
            "manifold": loss_manifold.item(),
            "overlap": loss_overlap.item(),
            "lam_sp": loss_sparsity.item(),
            "struct_weight": structural_weight,
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
        
        # 1. Expand the state to latent space exactly ONCE
        z = self.expander.expand(x)
        b = self._get_modal_coords(z)
        
        traj = [x.squeeze(0)]
        for _ in range(steps):
            b = self._step_modal(b)               # Clean loop
            z_next = self._modal_to_latent(b)     # Handles clamping automatically
            traj.append(self.expander.de_expand(z_next).squeeze(0))
        return torch.stack(traj)