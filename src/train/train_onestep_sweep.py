from src.train.train_onestep import train_onestep


def train_onestep_sweep(
    model,
    train_loader,
    val_loader,
    device="cpu",
    epochs=50,
    lr=1e-3,
    weight_decay=1e-6,
    log_phi_every=1,
    phi_print_max_dim=12,
    eval_callback=None,
    rollout_horizon=None,
):
    model, losses, best_checkpoint = train_onestep(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        log_phi_every=log_phi_every,
        phi_print_max_dim=phi_print_max_dim,
        eval_callback=eval_callback,
        rollout_horizon=rollout_horizon,
    )

    return model, losses, best_checkpoint
