from config import Config

import os
import inspect

import matplotlib.pyplot as plt
import numpy as np

from get_model import get_model
from build_model import build_registration_network
from generators import train_generator_
from generators import test_generator
from validation import evaluate_validation_dice
from configure_losses import configure_losses

def train_model(config, similarity_metric, weak_supervision, experiment_name,
                last_trial=None, latest_weights=None, Verbose=False):
    """Fit the registration network trial by trial. One trial = 32 optimiser steps + full validation."""

    train_cache = config.train_cache
    test_cache = config.test_cache
    moving_image_shape = tuple(config.config_yaml["moving_image_shape"]) 
    fixed_image_shape = tuple(config.config_yaml["fixed_image_shape"]) 

    save_path_root = config.config_yaml["save_path_root"]

    model_save_path = f'{save_path_root}/{experiment_name}'
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    for lambda_param in [0.1]:  # incase you want to tune the regularization scalar

        # =============================================================================
        # Build the backbone model and wrap it into the registration network
        # =============================================================================

        backbone = get_model(config, with_label_inputs=False)
        registration_model, spatial_transformer = build_registration_network(config, backbone)

        print('\nBackbone model inputs and outputs:')
        print('    input shape: ', ', '.join(str(t.shape) for t in backbone.inputs))
        print('    output shape:', ', '.join(str(t.shape) for t in backbone.outputs))

        print('\nRegistration network inputs and outputs:')
        print('    input shape: ', ', '.join(str(t.shape) for t in registration_model.inputs))
        print('    output shape:', ', '.join(str(t.shape) for t in registration_model.outputs))

        # =============================================================================
        # Compile with the configured loss grid
        # =============================================================================

        loss_functions, loss_weights = configure_losses(similarity_metric, weak_supervision, lambda_param)

        registration_model.compile(optimizer='Adam', loss=loss_functions, loss_weights=loss_weights)

        # =========================================================================
        # Resume from last trial
        # =========================================================================

        if latest_weights is not None:
            print(f"Resuming from {latest_weights} (trial {last_trial})")
            registration_model.load_weights(latest_weights)
            start_trial = last_trial + 1

            # Load arrays
            val_dice = np.load(os.path.join(model_save_path, "val_dice.npy")).tolist()
            transformer_losses = np.load(os.path.join(model_save_path, "transformer_losses.npy")).tolist()
            losses = np.load(os.path.join(model_save_path, "losses.npy")).tolist()
            conv3d_losses = np.load(os.path.join(model_save_path, "conv3d_losses.npy")).tolist()

            # Index from 0 to trial_num
            val_dice = val_dice[:last_trial + 1]
            transformer_losses = transformer_losses[:last_trial + 1]
            losses = losses[:last_trial + 1]
            conv3d_losses = conv3d_losses[:last_trial + 1]

        else:
            print("No previous weights found, starting fresh")
            start_trial = 0
            val_dice = []
            losses = []
            transformer_losses = []
            conv3d_losses = []

        # =============================================================================
        # Trial loop
        # =============================================================================

        batch_size = 8 # Decrease this if you are running out of RAM - set to 4, 8, 16 ect.

        train_gen = train_generator_(config, batch_size, with_label_inputs=True)

        num_trials = 1024 # This may be way above what we require

        for trial in range(start_trial, num_trials):
            print(f'\nTrial {trial} / {num_trials-1}:')

            hist = registration_model.fit(train_gen, epochs=1, steps_per_epoch=32, verbose=1);

            dice_scores = evaluate_validation_dice(config,
                                                   registration_model, 
                                                   spatial_transformer,
                                                   trial=trial, 
                                                   model_save_path=model_save_path, 
                                                   plot_first_batch=True)

            values = [arr.item() for arr in dice_scores]
            print(values)

            losses.append(hist.history["loss"][0])
            transformer_losses.append(hist.history["transformer_loss"][0])          # similarity head
            conv_loss_key = next(
                (key for key in hist.history.keys() 
                 if key.startswith('conv3d_') and key.endswith('_loss')),
                None
            )               
            conv3d_losses.append(hist.history[conv_loss_key][0])                     # regularisation head
            val_dice.append(np.mean(dice_scores))

            # Training curves for this trial
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.subplots_adjust(wspace=0.5) 

            plt.plot(losses, label="Total Loss")
            plt.plot(transformer_losses, label="Similarity Loss")
            plt.xlabel('Trials')
            plt.ylabel('Losses')
            plt.legend()  

            plt.subplot(1, 3, 2)
            plt.plot(conv3d_losses, label="Regularization Loss")
            plt.xlabel('Trials')
            plt.ylabel('Losses')
            plt.legend()  

            plt.subplot(1, 3, 3)
            plt.plot(val_dice, 'r')
            plt.xlabel('Trials')
            plt.ylabel('Dice')

            # Save figure with unique filename
            save_name = os.path.join(model_save_path, f"training_curves_trial_{trial}.png")
            plt.savefig(save_name, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved plot to {save_name}")

            print('    Validation Dice: ', np.mean(dice_scores))

# if __name__ == '__main__':
#     config = Config()
#     moving_image_shape = tuple(config.config_yaml["moving_image_shape"])
#     fixed_image_shape = tuple(config.config_yaml["fixed_image_shape"])

#     print("train_model", inspect.signature(train_model))
#     print()
#     for fn in [get_model, build_registration_network, configure_losses, train_generator_, test_generator, evaluate_validation_dice]:
#         print(f"    {fn.__name__}{inspect.signature(fn)}")
