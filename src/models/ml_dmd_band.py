import torch
import torch.nn as nn
from src.models.expander import ManualExpansion

class ML_DMD_BAND(ManualExpansion):
    def __init__(self, state_dim=2, expansion_degree=2, bias=True, 
                 sine_cosine_expansion=False, expansion_type="general", system=None):
        super().__init__(state_dim=state_dim, expansion_degree=expansion_degree, bias=bias,
                         sine_cosine_expansion=sine_cosine_expansion, expansion_type=expansion_type, system=system)

        self.latent_dim = self.expanded_dim

        # ------------------------------------------------
        # 1. Scaled Parameters (Optimized in Normalized Space)
        # ------------------------------------------------
        # self.Phi_scaled = nn.Parameter(
        #     torch.eye(self.latent_dim) + 0.2 * torch.randn(self.latent_dim, self.latent_dim)
        # )

        phi_init = torch.empty(self.latent_dim, self.latent_dim)
        nn.init.orthogonal_(phi_init)
        self.Phi_scaled = nn.Parameter(phi_init)

        self.eig_diag = nn.Parameter(torch.ones(self.latent_dim) + torch.randn(self.latent_dim) * 0.01)
        self.eig_super = nn.Parameter(torch.randn(self.latent_dim - 1) * 0.01)
        self.eig_sub = nn.Parameter(torch.randn(self.latent_dim - 1) * 0.01)

        self.register_buffer("x_mean", torch.zeros(state_dim))
        self.register_buffer("x_scale", torch.ones(state_dim))
        self.register_buffer("z_scale", torch.ones(self.latent_dim))
        self.max_abs_z_norm = 1e6
        self.coupling_gate_temperature = 1e-3
        self.lambda_drift_target = 1.0
        self.lambda_drift_weight = 1e-3

    def set_state_scale(self, x_mean, x_scale):
        self.x_mean.copy_(x_mean.to(self.x_mean.device))
        self.x_scale.copy_(torch.clamp(x_scale.to(self.x_scale.device), min=1e-6))

    def set_z_scale(self, z_scale):
        self.z_scale.copy_(z_scale.to(self.z_scale.device))

    def scale_state(self, x): return (x - self.x_mean) / self.x_scale    
    def unscale_state(self, x): return x * self.x_scale + self.x_mean

    def get_Lambda(self):
        return (torch.diag(self.eig_diag) + 
                torch.diag(self.eig_super, diagonal=1) + 
                torch.diag(self.eig_sub, diagonal=-1))

    def get_coupling_gate(self):
        """Smoothly emphasize rows/columns that participate in off-diagonal coupling.

        A hard threshold can freeze the coupling terms before they have a chance to
        organize into meaningful 2x2 blocks, so this keeps the gradient signal alive.
        """
        return torch.sigmoid((torch.abs(self.eig_sub) - self.coupling_gate_temperature) / self.coupling_gate_temperature)
    
    def get_Phi(self):
        """Returns descaled Physical Phi for visualization.[cite: 13]"""
        return torch.diag(self.z_scale) @ self.Phi_scaled

    def get_Phi_inv(self):
        """Returns descaled Physical Phi_inv.[cite: 13]"""
        return torch.linalg.pinv(self.Phi_scaled) @ torch.diag(1.0 / (self.z_scale + 1e-6))
    
    def get_K(self):
        """Returns descaled Physical K.[cite: 13]"""
        # K_phys = S @ K_scaled @ S_inv
        K_scaled = self.Phi_scaled @ self.get_Lambda() @ torch.linalg.pinv(self.Phi_scaled)
        S = torch.diag(self.z_scale)
        S_inv = torch.diag(1.0 / (self.z_scale + 1e-6))
        return S @ K_scaled @ S_inv

    def forward(self, x):
        # 1. Move to Scaled Feature Space
        z = self.expand(self.scale_state(x)) / self.z_scale
        z = torch.clamp(z, min=-self.max_abs_z_norm, max=self.max_abs_z_norm)
        
        # 2. Linear Evolution in Scaled Space
        Phi_reg = self.Phi_scaled + 1e-6 * torch.eye(self.latent_dim, device=self.Phi_scaled.device)
        b = torch.linalg.solve(Phi_reg, z.mT).mT
        z_next_scaled = (b @ self.get_Lambda().mT) @ self.Phi_scaled.mT

        # 3. Return to Physical State Space
        z_next_phys = z_next_scaled * self.z_scale
        return self.unscale_state(self.de_expand(z_next_phys))

    def compute_loss(self, x, x_next_true):
        # 1. Feature Scaling and Evolution
        z = self.expand(self.scale_state(x)) / self.z_scale
        z_next_true = self.expand(self.scale_state(x_next_true)) / self.z_scale
        
        Phi_reg = self.Phi_scaled + 1e-6 * torch.eye(self.latent_dim, device=self.Phi_scaled.device)
        b = torch.linalg.solve(Phi_reg, z.mT).mT
        z_next_pred_scaled = (b @ self.get_Lambda().mT) @ self.Phi_scaled.mT

        # 2. Accuracy Losses
        loss_lift = torch.mean((z_next_pred_scaled - z_next_true)**2)
        z_next_pred_phys = z_next_pred_scaled * self.z_scale
        loss_state = nn.MSELoss()(self.de_expand(z_next_pred_phys), self.scale_state(x_next_true))

        # 3. Geometry regularization
        # Keep Phi well-conditioned enough to be numerically usable, but do not
        # force sparsity on the physical operator. Sparsity is basis-dependent and
        # can fight the similarity transform the model is trying to learn.
        col_norms = torch.linalg.norm(self.get_Phi(), dim=0)
        loss_unit_length = torch.mean((col_norms - 1.0) ** 2)

        # 4. Lambda physics
        # Use a smooth gate so the coupling terms can gradually organize into
        # oscillatory 2x2 blocks instead of getting zeroed out too early.
        coupling_gate = self.get_coupling_gate()

        loss_antisym = torch.mean(coupling_gate * (self.eig_super + self.eig_sub)**2)
        diag_diff = self.eig_diag[:-1] - self.eig_diag[1:]
        loss_diag_match = torch.mean(coupling_gate * diag_diff**2)
        loss_lambda_balance = torch.mean((torch.abs(self.eig_super) - torch.abs(self.eig_sub))**2)
        # Encourage a consistent sign convention for the sub-diagonal couplings.
        loss_sign_sub = torch.mean(torch.relu(self.eig_sub))

        # Keep the diagonal eigenvalues near a tunable target, but with a very
        # light weight so growth/decay can still be learned when the data demands it.
        loss_lambda_drift = torch.mean((torch.abs(self.eig_diag) - self.lambda_drift_target) ** 2)

        # 5. Lean Loss
        loss_total = (
            loss_lift
            + 10.0 * loss_state         
            + 1e-3 * loss_unit_length   
            + 1.0 * loss_antisym        
            + 1.0 * loss_diag_match     
            + 0.5 * loss_lambda_balance 
            + 0.1 * loss_sign_sub       
            + self.lambda_drift_weight * loss_lambda_drift  
        )

        loss_dict = {
            "state": loss_state.item(),
            "lam_bal": loss_lambda_balance.item(),
            "lam_drift": loss_lambda_drift.item(),
            "unit": loss_unit_length.item(),
            "couple": coupling_gate.mean().item(),
        }
        
        return loss_total, loss_dict

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