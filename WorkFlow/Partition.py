import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_covid_data(file_path, output_dir):
    # 读取数据
    df = pd.read_csv(file_path)

    # 检查缺失值
    missing_values = df.isnull().sum()
    print("Missing values per column:")
    print(missing_values)

    # 选择数值列进行标准化
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # 进行标准化
    scaler = StandardScaler()
    df_standardized = df.copy()
    df_standardized[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # 检查标准化后是否和原数据不同
    if not df_standardized.equals(df):
        standardized_file_path = "../Data/standardized_data.csv"
        df_standardized.to_csv(standardized_file_path, index=False)
        print(f"Standardized data saved to {standardized_file_path}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 按国家分割数据并存储
    country_groups = df.groupby("Entity")
    for country, data in country_groups:
        country_file_path = os.path.join(output_dir, f"{country}.csv")
        data.to_csv(country_file_path, index=False)
        print(f"Saved {country} data to {country_file_path}")


# 示例调用
file_path = "../Data/daily-new-confirmed-covid-19-cases-per-million-people.csv"
output_dir = "../Data/country_data"
preprocess_covid_data(file_path, output_dir)