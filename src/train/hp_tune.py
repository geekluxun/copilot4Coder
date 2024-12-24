from ray import tune


def hp_space(backend='optuna'):
    if backend == 'optuna':
        return _optuna_hp_space
    elif backend == 'ray':
        return _ray_hp_space
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _ray_hp_space(trail):
    return {
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "per_device_train_batch_size": tune.choice([16, 32, 64, 128]),
    }


def _optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 5e-4),
        "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [4, 8, 16, 24]),
    }
