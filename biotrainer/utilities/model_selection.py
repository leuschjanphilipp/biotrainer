import os
import optuna
import datetime
import randomname

#from biotrainer.utilities.cli import train
from ..utilities.cli import train

def objective(trial, config):

    config["output_dir"] = f"optuna/{config['model_selection']['study_name']}/trial_{trial.number}"

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    config["learning_rate"] = learning_rate

    model_choice = trial.suggest_categorical("model_choice", ["CNN", "FNN"])
    config["model_choice"] = model_choice

    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_dims = [trial.suggest_int(f"hidden_dim_{i}", 32, 512) for i in range(n_layers-1)]

    config["model_params"] = {
        "dropout_rate": dropout_rate,
        "hidden_dims": hidden_dims,
    }

    if model_choice == "CNN":
        #additional CNN parameters
        kernel_sizes = [(trial.suggest_categorical(f"kernel_size_{i}", [5, 7, 9, 11]), 1) for i in range(n_layers)]
        padding = [(k[0] // 2, 0) for k in kernel_sizes]
        last_layer_FNN = False # trial.suggest_categorical("last_layer_FNN", [True, False])
        config["model_params"].update({
            "kernel_sizes": kernel_sizes,
            "padding": padding,
            "last_layer_FNN": last_layer_FNN
        })

    res = train(config)
    return res["training_results"]["hold_out"]["best_training_epoch_metrics"]["validation"][config["model_selection"]["objective_metric"]]

def model_selection(config: dict):
    """
    Run model selection using Optuna.

    @param config: Biotrainer configuration dictionary
    """
    assert "model_selection" in config, "Config must contain 'model_selection' dict in order to run model selection."

    if config["model_selection"]["study_name"] is None:
        config["model_selection"]["study_name"] = datetime.datetime.now().strftime("%Y-%m-%d") + "-" + randomname.generate()

    os.makedirs("optuna", exist_ok=True)
    os.makedirs(f"optuna/{config['model_selection']['study_name']}", exist_ok=True)
    storage = f'sqlite:///optuna/{config["model_selection"]["study_name"]}/study.db'

    sampler = optuna.samplers.TPESampler(seed=config["model_selection"]["seed"])  # since it supports conditional sampling

    study = optuna.create_study(study_name=config["model_selection"]["study_name"],
                                direction=config["model_selection"]["direction"],
                                sampler=sampler,
                                storage=storage,
                                load_if_exists=True)

    study.optimize(lambda trial: objective(trial, config), n_trials=config["model_selection"]["n_trials"])

    print("\nBest trial:")
    print(study.best_trial, "\n", study.best_trial.value)