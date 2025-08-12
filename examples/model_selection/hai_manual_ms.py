import os
import sys
sys.path.insert(0, "biotrainer")
import optuna
from biotrainer.protocols import Protocol
from biotrainer.utilities.cli import train

config = {
    "input_file": "data/sampled_3Dii.fasta",
    "protocol": Protocol.residue_to_class.name,
    "model_choice": "CNN",
    "device": "cuda",
    "optimizer_choice": "adam",
    "learning_rate": 1e-3,
    "loss_choice": "cross_entropy_loss",
    "num_epochs": 200,
    "batch_size": 1024,
    "patience": 5,
    "ignore_file_inconsistencies": True,
    "cross_validation_config": {
        "method": "hold_out"
    },
    "embeddings_file": "data/embeddings_file_ProstT5.h5",
    #"embedder_name": "RostLab/ProstT5",
    "model_params": {
            "dropout_rate": 0.15,
            "n_layers": 3,
            "kernel_sizes": [(7, 1), (7, 1), (5, 1)],
            "padding": [(3, 0), (3, 0), (2, 0)],
            "hidden_dims": [256, 32]
    },
}

def objective(trial):
    config["output_dir"] = f"optuna_trials/{trial.number}"

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    config["learning_rate"] = learning_rate

    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    kernel_sizes = [(trial.suggest_categorical(f"kernel_size_{i}", [3, 5, 7]), 1) for i in range(n_layers)]
    padding = [(k[0] // 2, 0) for k in kernel_sizes]
    hidden_dims = [trial.suggest_int(f"hidden_dim_{i}", 32, 512) for i in range(n_layers-1)]
    last_layer_FNN = trial.suggest_categorical("last_layer_FNN", [True, False])

    config["model_params"] = {
        "dropout_rate": dropout_rate,
        "kernel_sizes": kernel_sizes,
        "padding": padding,
        "hidden_dims": hidden_dims,
        "last_layer_FNN": last_layer_FNN
    }

    print(config)
    res = train(config)
    return res["training_results"]["hold_out"]["best_training_epoch_metrics"]["training"]["accuracy"]


os.makedirs("optuna_trials", exist_ok=True)
storage = f'sqlite:///optuna_trials/study.db'

sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(study_name="3Dii",
                            direction="maximize", 
                            sampler=sampler, 
                            storage=storage, 
                            load_if_exists=True)

study.optimize(objective, n_trials=10)

print("Best trial:")
print(study.best_trial)
print("Best value:", study.best_value)
print("Best params:", study.best_params)
print("All trials:")
for trial in study.trials:
    print(trial) 
    print("Value:", trial.value)