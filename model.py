"""
NumPy Multiple Linear Regression GD

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - shuffle_xy (not yet solved)
# TODO: implement

# Step 2 - split_train_val_test (not yet solved)
# TODO: implement

# Step 3 - compute_feature_stats (not yet solved)
# TODO: implement

# Step 4 - standardize_features (not yet solved)
# TODO: implement

# Step 5 - add_bias_column (not yet solved)
# TODO: implement

# Step 6 - prepare_design_matrix (not yet solved)
# TODO: implement

# Step 7 - predict_linear
def predict_linear(X, weights):
    """Compute linear predictions y_hat = X @ weights.

    Args:
        X: Design matrix of shape (n, d_in), often including a bias column.
        weights: Weight vector of shape (d_in,).

    Returns:
        Predicted targets of shape (n,).
    """
    # TODO: Return the predicted target vector from X and weights
    return X @ weights

# Step 8 - mse_loss (not yet solved)
# TODO: implement

# Step 9 - mse_gradient (not yet solved)
# TODO: implement

# Step 10 - normal_equation (not yet solved)
# TODO: implement

# Step 11 - initialize_weights (not yet solved)
# TODO: implement

# Step 12 - gd_step (not yet solved)
# TODO: implement

# Step 13 - epoch_train_val_losses (not yet solved)
# TODO: implement

# Step 14 - update_early_stop_state (not yet solved)
# TODO: implement

# Step 15 - init_training_state (not yet solved)
# TODO: implement

# Step 16 - run_one_epoch (not yet solved)
# TODO: implement

# Step 17 - train_batch_gd (not yet solved)
# TODO: implement

# Step 18 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: Compute the mean absolute error between true targets and predictions
    return np.mean(np.abs(y_true-y_pred))

# Step 19 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    # TODO: Return the root mean squared error between y_true and y_pred.
    return np.sqrt(np.mean(np.square(y_true-y_pred)))

# Step 20 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute the coefficient of determination R^2.
    ss_res = sum(np.square(y_true-y_pred))
    ss_tot = sum(np.square(y_true-np.mean(y_true)))
    if ss_tot == 0:
        return np.nan
    return 1-(ss_res/ss_tot)

# Step 21 - evaluate_regression
def evaluate_regression(y_true, y_pred):
    # TODO: Bundle MAE, RMSE, and R^2 into one metrics dictionary for test-set reporting.
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'r2': r_squared(y_true, y_pred)
    }

# Step 22 - learning_curve_data (not yet solved)
# TODO: implement

# Step 23 - weights_l2_distance (not yet solved)
# TODO: implement

# Step 24 - create_lr_model (not yet solved)
# TODO: implement

# Step 25 - fit_lr_model (not yet solved)
# TODO: implement

# Step 26 - predict_lr_model (not yet solved)
# TODO: implement

# Step 27 - score_lr_model (not yet solved)
# TODO: implement

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement

