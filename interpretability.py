import argparse
import datetime
import os
import pickle
import time
import numpy as np
import gpflow
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler


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

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # 用于存储不同ratio下的总体均值和总体标准差结果
    ratio_results = []
    for ratio in np.arange(0.1, 1.1, 0.1):  # 从0.1每次增加0.1到0.9，共9次
        args.ratio = ratio
        fold_results = []
        spearman_corrs_all_folds = []
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
            print(f"Fold {fold_idx + 1} Training:")
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            diagnosis_train, diagnosis_test = diagnosis[train_idx], diagnosis[test_idx]

            num_dimensions = X_train.shape[1]
            lengthscales = np.ones(num_dimensions) * args.scale

            kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscales)
            model = gpflow.models.GPR(data=(X_train, Y_train), kernel=kernel)
            print("训练前")
            print(model.parameters)

            optimizer = gpflow.optimizers.Scipy()
            optimizer.minimize(model.training_loss, model.trainable_variables)
            print("训练后")
            print(model.parameters)

            mean, var = model.predict_f(X_test)
            print("Mean predictions:\n", mean)

            pred_met = mean
            true_met = Y_test

            n_metabolites = pred_met.shape[1]
            metabolite_names = Name2

            correlations = []
            with open(f"{DATA_ROOT}/GPFlow_Spearman_Fold{fold_idx + 1}.txt", "w") as f:
                for i in range(n_metabolites):
                    corr, _ = spearmanr(pred_met[:, i], true_met[:, i])
                    metabolite_name = metabolite_names[i]
                    f.write(f"{metabolite_name}\t{corr}\n")
                    correlations.append(corr)

            correlations_sorted = sorted(correlations, reverse=True)
            mean_corr_top10 = np.mean(correlations_sorted[:10])
            print(f"Fold {fold_idx + 1}: Spearman相关系数文件中前10个代谢物的均值是：", mean_corr_top10)

            # fold_results.append(mean_corr_top10)
            spearman_corrs_all_folds.append(correlations)

            inv_lengthscale = np.asarray(model.kernel.lengthscales) ** (-1)
            sorted_indices_inv_lengthscale = np.argsort(inv_lengthscale)[::-1]
            top_ratio_percentile = int(args.ratio * len(inv_lengthscale))
            top_ratio_percentile_feature_names = [Name1[i] for i in
                                                  sorted_indices_inv_lengthscale[:top_ratio_percentile]]
            top_ratio_percentile_original_indices = [np.where(np.arange(len(inv_lengthscale)) == idx)[0][0] for idx in
                                                     sorted_indices_inv_lengthscale[:top_ratio_percentile]]

            print(f"Fold {fold_idx + 1}: Top ratio original indices corresponding to highest inv_lengthscales:")
            for original_index in top_ratio_percentile_original_indices:
                print(original_index + 2)
            print(f"Fold {fold_idx + 1}: Top ratio feature names with highest inv_lengthscales:")
            for name in top_ratio_percentile_feature_names:
                print(name)

            X_train_selected = X_train[:, top_ratio_percentile_original_indices]
            X_test_selected = X_test[:, top_ratio_percentile_original_indices]

            kernel_selected = gpflow.kernels.SquaredExponential(lengthscales=np.ones(top_ratio_percentile) * args.scale)
            model_selected = gpflow.models.GPR(data=(X_train_selected, Y_train), kernel=kernel_selected)

            optimizer_selected = gpflow.optimizers.Scipy()
            optimizer_selected.minimize(model_selected.training_loss, model_selected.trainable_variables)

            mean_selected, var_selected = model_selected.predict_f(X_test_selected)

            pred_met = mean_selected
            true_met = Y_test
            correlations = []
            with open(f"{DATA_ROOT}/interpretability_Spearman_Fold{fold_idx + 1}.txt", "w") as f:
                for i in range(n_metabolites):
                    corr, _ = spearmanr(pred_met[:, i], true_met[:, i])
                    metabolite_name = metabolite_names[i]
                    f.write(f"{metabolite_name}\t{corr}\n")
                    correlations.append(corr)

            correlations_sorted = sorted(correlations, reverse=True)
            mean_corr_top10 = np.mean(correlations_sorted[:10])
            print(f"Fold {fold_idx + 1}: 筛选特征重新建模后Spearman相关系数文件中前10个代谢物的均值是：", mean_corr_top10)
            fold_results.append(mean_corr_top10)

            inv_lengthscale = np.asarray(model_selected.kernel.lengthscales) ** (-1)

            plt.subplot(132)
            plt.bar(np.arange(inv_lengthscale.size), height=inv_lengthscale)
            plt.title('Inverse Lengthscale with SE-ARD kernel', fontsize='small')
            plt.suptitle(f'Dataset Name: {args.dataset_name}, spearman_corrs_top10={mean_corr_top10}',
                         fontsize='medium')
            # plt.show()

        overall_mean_top10 = np.mean(fold_results)
        overall_std_top10 = np.std(fold_results)
        print(ratio)
        print(f"总体均值 (Top 10代谢物): {overall_mean_top10:.3f}±{overall_std_top10:.3f}")
        ratio_results.append((ratio, overall_mean_top10, overall_std_top10))

    # 可以在这里根据需求进一步处理ratio_results，比如保存到文件等
    print(ratio_results)

    end_time = time.time()
    print(f'代码运行时间：{end_time - start_time:.2f}秒')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='./data/ESRD')
    parser.add_argument("--variable_prefix", type=str, default='ESRD')
    parser.add_argument("--dataset_name", type=str, default='ESRD')
    parser.add_argument("--scale", type=float, default=100)
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--data_type", type=str, default='clr')  # clr  log  z-score  min-max
    parser.add_argument("--fea1", type=str, default='bac_group_fea')
    parser.add_argument("--fea2", type=str, default='met_group_fea')
    parser.add_argument("--topk", type=int, default=10)

    args = parser.parse_args()

    main(args)
