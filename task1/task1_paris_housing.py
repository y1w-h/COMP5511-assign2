import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_and_inspect(path="ParisHousing.csv"):
    # 读入数据
    df = pd.read_csv(path)
    print("=== Data Info ===")
    print(df.info())
    print("\n=== Data Description ===")
    print(df.describe())
    return df


def train_test_preprocess(df, target_col="price", test_size=0.2, random_state=42):
    # 特征和标签
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 训练/测试划分（保持索引，后面按 numPrevOwners 分组会用到）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 特征标准化（除了 price 以外所有特征）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 转回 DataFrame，保留列名和索引
    X_train_scaled = pd.DataFrame(
        X_train_scaled, columns=X.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled, columns=X.columns, index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R^2 : {r2:.4f}")
    return {
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "n_samples": len(y_true),
    }


def baseline_linear_regression(X_train, X_test, y_train, y_test):
    """B.1 基线线性回归：在完整训练集上训练一个模型"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_model("Baseline Linear Regression", y_test, y_pred)
    return model, metrics


def models_by_num_prev_owners(
    X_train, X_test, y_train, y_test, owner_col="numPrevOwners"
):
    """B.2 按 numPrevOwners（1~10）划分数据，每一组独立训练一个回归模型"""
    results = []
    owners = sorted(X_train[owner_col].unique())
    for owner in owners:
        train_mask = X_train[owner_col] == owner
        test_mask = X_test[owner_col] == owner

        X_train_grp = X_train[train_mask]
        y_train_grp = y_train[train_mask]
        X_test_grp = X_test[test_mask]
        y_test_grp = y_test[test_mask]

        # 跳过样本太少或测试集中没有该组的情况
        if len(X_train_grp) < 2 or len(X_test_grp) == 0:
            continue

        model = LinearRegression()
        model.fit(X_train_grp, y_train_grp)
        y_pred_grp = model.predict(X_test_grp)
        metrics = evaluate_model(
            f"LR by numPrevOwners = {owner}", y_test_grp, y_pred_grp
        )
        results.append(metrics)

    # 计算加权平均表现（按各组测试样本数量加权）
    if results:
        total_n = sum(r["n_samples"] for r in results)
        weighted_mae = sum(r["MAE"] * r["n_samples"] for r in results) / total_n
        weighted_rmse = sum(r["RMSE"] * r["n_samples"] for r in results) / total_n
        weighted_r2 = sum(r["R2"] * r["n_samples"] for r in results) / total_n
        print("\n=== Overall performance (grouped by numPrevOwners, weighted by test size) ===")
        print(f"MAE : {weighted_mae:.2f}")
        print(f"RMSE: {weighted_rmse:.2f}")
        print(f"R^2 : {weighted_r2:.4f}")
    return results


def models_by_kmeans_clusters(
    X_train, X_test, y_train, y_test, n_clusters=4, random_state=42
):
    """B.3 使用 KMeans 聚类，再在每个 cluster 里各自训练线性回归"""
    # 只用特征做聚类（这里用已经标准化后的特征）
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    train_clusters = kmeans.fit_predict(X_train)
    test_clusters = kmeans.predict(X_test)

    results = []
    for c in range(n_clusters):
        train_mask = train_clusters == c
        test_mask = test_clusters == c

        X_train_c = X_train[train_mask]
        y_train_c = y_train[train_mask]
        X_test_c = X_test[test_mask]
        y_test_c = y_test[test_mask]

        if len(X_train_c) < 2 or len(X_test_c) == 0:
            continue

        model = LinearRegression()
        model.fit(X_train_c, y_train_c)
        y_pred_c = model.predict(X_test_c)
        metrics = evaluate_model(
            f"LR in KMeans cluster {c}", y_test_c, y_pred_c
        )
        results.append(metrics)

    if results:
        total_n = sum(r["n_samples"] for r in results)
        weighted_mae = sum(r["MAE"] * r["n_samples"] for r in results) / total_n
        weighted_rmse = sum(r["RMSE"] * r["n_samples"] for r in results) / total_n
        weighted_r2 = sum(r["R2"] * r["n_samples"] for r in results) / total_n
        print("\n=== Overall performance (cluster-based LR, weighted by test size) ===")
        print(f"MAE : {weighted_mae:.2f}")
        print(f"RMSE: {weighted_rmse:.2f}")
        print(f"R^2 : {weighted_r2:.4f}")
    return results


def main():
    # A.1–A.2 读取并查看数据
    df = load_and_inspect("ParisHousing.csv")

    # A.3–A.6 特征/标签划分 + 预处理 + 训练/测试划分
    X_train, X_test, y_train, y_test, scaler = train_test_preprocess(df)

    # B.1 基线线性回归模型
    baseline_model, baseline_metrics = baseline_linear_regression(
        X_train, X_test, y_train, y_test
    )

    # B.2 按 numPrevOwners 分组分别训练线性回归
    owners_results = models_by_num_prev_owners(
        X_train, X_test, y_train, y_test, owner_col="numPrevOwners"
    )

    # B.3 使用 KMeans 聚类后，每个 cluster 训练线性回归
    # 你可以根据结果修改 n_clusters（比如 3~8 之间试）
    kmeans_results = models_by_kmeans_clusters(
        X_train, X_test, y_train, y_test, n_clusters=4
    )


if __name__ == "__main__":
    main()
