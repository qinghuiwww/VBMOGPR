import argparse
import datetime
import os
import pickle
import time
import numpy as np
import gpflow
import pandas as pd
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

def save_ard_lengthscales(kernel, fold_idx, dataset_name, feature_names):
    # Extract the ARD lengthscales from the kernel
    lengthscales = kernel.lengthscales.numpy()

    # Compute the inverse of the lengthscales (1 / lengthscales)
    inverse_lengthscales = 1 / lengthscales

    # Combine the feature names and inverse lengthscales into a dictionary
    data = {
        "Feature_Name": feature_names,
        "Inverse_Lengthscale": inverse_lengthscales
    }

    # Convert the dictionary to a DataFrame
    inverse_lengthscales_df = pd.DataFrame(data)

    # Ensure the directory exists
    os.makedirs("inverse_lengthscales", exist_ok=True)

    # Save the inverse lengthscales to a CSV file
    inverse_lengthscale_file = os.path.join("inverse_lengthscales", f"{dataset_name}_Inverse_Lengthscales_Fold{fold_idx + 1}.csv")
    inverse_lengthscales_df.to_csv(inverse_lengthscale_file, index=False)

    print(f"Inverse lengthscales have been saved to {inverse_lengthscale_file}")


def main(args):
    start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    DATA_ROOT = args.data_root
    data_type = args.data_type
    data_path = os.path.join(DATA_ROOT, "{}_{}.pkl".format(args.variable_prefix, data_type))

    with open(data_path, "rb") as df:
        content = pickle.load(df)

    X1 = content[args.fea1]  # n x m1  bac_group_fea
    X2 = content[args.fea2]  # n x m2  met_group_fea
    Name1 = content[args.fea1.replace("fea", "ids")]
    Name2 = content[args.fea2.replace("fea", "ids")]

    diagnosis = content["diagnosis"]

    X = X1  # bac_group_fea
    Y = X2  # met_group_fea
    metabolite_names = Name2

    # Set up 5-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    spearman_corrs_all_folds = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        # Define the GPR model
        num_dimensions = X_train.shape[1]
        lengthscales = np.ones(num_dimensions) * args.scale
        # kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscales)
        # kernel = gpflow.kernels.RationalQuadratic(lengthscales=lengthscales)
        # kernel = gpflow.kernels.Exponential(lengthscales=lengthscales)
        # kernel = gpflow.kernels.Matern12(lengthscales=lengthscales)
        # kernel = gpflow.kernels.Matern32(lengthscales=lengthscales)
        kernel = gpflow.kernels.Matern52(lengthscales=lengthscales)
        model = gpflow.models.GPR(data=(X_train, Y_train), kernel=kernel)

        # Train the model
        optimizer = gpflow.optimizers.Scipy()
        optimizer.minimize(model.training_loss, model.trainable_variables)

        # Save the ARD lengthscales
        save_ard_lengthscales(model.kernel, fold_idx, args.dataset_name, Name1)

        # Predict
        mean, var = model.predict_f(X_test)
        pred_met = mean
        true_met = Y_test

        # Calculate Spearman correlations for each metabolite
        correlations = [spearmanr(pred_met[:, i], true_met[:, i])[0] for i in range(Y_test.shape[1])]
        spearman_corrs_all_folds.append(correlations)

        # Save predictions and Spearman correlation per fold
        fold_corr_file = f"{args.dataset_name}_Spearman_Fold{fold_idx + 1}.txt"
        with open(fold_corr_file, "w") as f:
            for i, corr in enumerate(correlations):
                f.write(f"{metabolite_names[i]}\t{corr}\n")

        # Get mean Spearman correlation for the top 10 metabolites (sorted by correlation)
        topk_corrs = sorted(correlations, reverse=True)[:args.topk]
        mean_corr_topk = np.mean(topk_corrs)
        fold_results.append(mean_corr_topk)

        print(f"Fold {fold_idx + 1} mean Spearman correlation of top {args.topk} metabolites: {mean_corr_topk:.4f}")

    # Calculate overall mean and std of Spearman correlations from the 5 folds
    overall_mean_top10 = np.mean(fold_results)
    overall_std_top10 = np.std(fold_results)

    # Save a combined Spearman correlation file for all folds
    summary_file_name = f"{args.dataset_name}_Combined_Spearman.txt"
    with open(summary_file_name, "w") as f:
        for i, corr_list in enumerate(spearman_corrs_all_folds):
            f.write(f"Fold {i + 1}:\n")
            for j, corr in enumerate(corr_list):
                f.write(f"{metabolite_names[j]}\t{corr}\n")

    # Print the formatted outputs
    print(f"Spearman correlation summary file saved as: {summary_file_name}")
    print(f"Overall mean of top 10 metabolites: {overall_mean_top10:.3f}±{overall_std_top10:.3f}")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='../genus_data/ADENOMAS')
    parser.add_argument("--variable_prefix", type=str, default='ADENOMAS')
    parser.add_argument("--dataset_name", type=str, default='ADENOMAS')
    parser.add_argument("--scale", type=float, default=150)
    parser.add_argument("--data_type", type=str, default='clr')  # clr  log  z-score  min-max
    parser.add_argument("--fea1", type=str, default='bac_group_fea')
    parser.add_argument("--fea2", type=str, default='met_group_fea')
    parser.add_argument("--topk", type=int, default=10)

    args = parser.parse_args()
    main(args)
