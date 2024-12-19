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

def main(args):
    # 记录开始时间
    start_time = time.time()
    # Get the current timestamp
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

    # 进行5折交叉验证
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []
    spearman_corrs_all_folds = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
        # 划分数据
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        diagnosis_train, diagnosis_test = diagnosis[train_idx], diagnosis[test_idx]

        num_dimensions = X_train.shape[1]
        lengthscales = np.ones(num_dimensions) * args.scale

        all_mean_preds = []
        all_var_preds = []
        for i in range(Y_train.shape[1]):
            Y_train_i = Y_train[:, i:i + 1]  # 选择单独的输出进行训练
            Y_test_i = Y_test[:, i:i + 1]  # 选择单独的输出进行测试

            # 创建高斯过程回归模型，并设置长度尺度参数
            kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscales)
            model = gpflow.models.GPR(data=(X_train, Y_train_i), kernel=kernel)

            # print(f"训练第 {i + 1} 个输出前")
            # 进行模型训练
            optimizer = gpflow.optimizers.Scipy()
            optimizer.minimize(model.training_loss, model.trainable_variables)

            print(f"训练第 {i + 1} 个输出后")

            # 使用模型进行预测
            mean_i, var_i = model.predict_f(X_test)

            all_mean_preds.append(mean_i)
            all_var_preds.append(var_i)

        # 将所有输出的预测结果沿着列方向合并
        all_mean_preds = np.concatenate(all_mean_preds, axis=1)
        all_var_preds = np.concatenate(all_var_preds, axis=1)

        # 打印预测结果
        print("Mean predictions:\n", all_mean_preds)

        pred_met = all_mean_preds
        true_met = Y_test

        n_metabolites = pred_met.shape[1]
        rmse_scores = []
        r2_scores = []
        metabolite_names = Name2

        topk = args.topk  # 需要选取的前k个代谢物

        # 获取排名前topk的代谢物
        topk_indices = np.argsort(r2_scores)[::-1][:topk]

        # 生成Spearman文件
        correlations = []
        with open(f"{DATA_ROOT}/SOGP_Spearman_Fold{fold_idx + 1}.txt", "w") as f:
            for i in range(n_metabolites):
                corr, _ = spearmanr(pred_met[:, i], true_met[:, i])
                metabolite_name = metabolite_names[i]
                f.write(f"{metabolite_name}\t{corr}\n")
                correlations.append(corr)

        # 根据Spearman相关系数从高到低排序
        correlations_sorted = sorted(correlations, reverse=True)

        # 获取前10个代谢物的均值
        mean_corr_top10 = np.mean(correlations_sorted[:10])
        print(f"Fold {fold_idx + 1}: Spearman相关系数文件中前10个代谢物的均值是：", mean_corr_top10)

        fold_results.append(mean_corr_top10)
        spearman_corrs_all_folds.append(correlations)

        # 绘制SE-ARD图
        inv_lengthscale = np.asarray(model.kernel.lengthscales) ** (-1)

        # 按照 inv_lengthscale 值进行排序，获取排序后的索引
        sorted_indices_inv_lengthscale = np.argsort(inv_lengthscale)[::-1]

        # 获取 inv_lengthscale 值最高的前10个特征名
        top_10_feature_names_inv_lengthscale = [Name1[i] for i in sorted_indices_inv_lengthscale[:10]]

        print(f"Fold {fold_idx + 1}: Top 10 feature names with highest inv_lengthscales:")
        for name in top_10_feature_names_inv_lengthscale:
            print(name)

    # 计算所有折叠的均值和标准差
    overall_mean_top10 = np.mean(fold_results)
    overall_std_top10 = np.std(fold_results)

    # 保存最终结果
    summary_file_name = f"{args.dataset_name}_Combined_Spearman.txt"
    with open(summary_file_name, "w") as f:
        for i, corr_list in enumerate(spearman_corrs_all_folds):
            f.write(f"Fold {i + 1}:\n")
            for j, corr in enumerate(corr_list):
                f.write(f"{metabolite_names[j]}\t{corr}\n")

    print(f"Spearman相关系数汇总文件已保存为: {summary_file_name}")
    print(f"总体均值 (Top 10代谢物): {overall_mean_top10:.3f}±{overall_std_top10:.3f}")

    # 保存SE-ARD图的数据
    inv_lengthscale_all_folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        kernel = gpflow.kernels.SquaredExponential(lengthscales=np.ones(X_train.shape[1]) * args.scale)
        model = gpflow.models.GPR(data=(X_train, Y[train_idx]), kernel=kernel)
        inv_lengthscale_all_folds.append(np.asarray(model.kernel.lengthscales) ** (-1))

    # 将所有fold的inverse lengthscales汇总并保存
    inv_lengthscale_all_folds = np.mean(np.array(inv_lengthscale_all_folds), axis=0)
    df = pd.DataFrame({"FeatureName": Name1, "InvLengthscale": inv_lengthscale_all_folds})
    save_path = os.path.join(DATA_ROOT, "SOGP-inv_lengthscale_all.csv")
    df.to_csv(save_path, index=False)

    # 打印运行时间
    end_time = time.time()
    print('代码运行时间：{:.2f}秒'.format(end_time - start_time))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='./data/ESRD')
    parser.add_argument("--variable_prefix", type=str, default='ESRD')
    parser.add_argument("--dataset_name", type=str, default='ESRD')
    parser.add_argument("--scale", type=float, default=100)
    parser.add_argument("--data_type", type=str, default='clr')
    parser.add_argument("--fea1", type=str, default='bac_group_fea')
    parser.add_argument("--fea2", type=str, default='met_group_fea')
    parser.add_argument("--topk", type=int, default=10)

    args = parser.parse_args()

    main(args)
