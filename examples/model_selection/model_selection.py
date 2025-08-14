import sys
sys.path.insert(0, "biotrainer") # since we dont want to use the pip install but the repo

from biotrainer.utilities.model_selection import model_selection

config = {"input_file": "data/sampled_3Dii.fasta",
          "protocol": "residue_to_class",
          "model_choice": "CNN",
          "device": "mps",
          "optimizer_choice": "adam",
          "learning_rate": 1e-3,
          "loss_choice": "cross_entropy_loss",
          "num_epochs": 200,
          "batch_size": 256,
          "ignore_file_inconsistencies": True,
          "embeddings_file": "data/embeddings_file_ProstT5.h5",
          "cross_validation_config": {
              "method": "hold_out"
          },
          "model_selection": {
              "study_name": None, 
              "n_trials": 10, 
              "direction": "maximize", 
              "seed": 42,
              "objective_metric": "accuracy"
          }
      }

model_selection(config)