import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

# ======================
# 目录配置
# ======================
BASE_DIR = "D:/python/Code/DPW_COVID_19"
DATA_DIR = os.path.join(BASE_DIR, "Data/country_data")  # 原始数据目录
RESULTS_DIR = os.path.join(BASE_DIR, "Analysis_Results2")  # 统一结果目录

# 创建标准化子目录结构
SUB_DIRS = {
    'stats': 'statistics',
    'trends': 'visualizations/trends',
    'models': 'model_outputs',
    'forecasts': 'forecasts',
    'anomalies': 'anomalies',
    'comparisons': 'visualizations/comparisons'
}


def create_directory_structure():
    """创建标准化的结果目录结构"""
    for category in SUB_DIRS.values():
        dir_path = os.path.join(RESULTS_DIR, category)
        os.makedirs(dir_path, exist_ok=True)


# ======================
# 核心分析模块
# ======================
class CountryAnalyzer:
    def __init__(self, country_name, data_path):
        self.country = country_name
        self.df = pd.read_csv(data_path)
        self.cases_col = "Daily new confirmed cases of COVID-19 per million people (rolling 7-day average, right-aligned)"
        self.prepare_data()

    def prepare_data(self):
        """数据预处理"""
        self.df['Day'] = pd.to_datetime(self.df['Day'])
        self.df.set_index('Day', inplace=True)
        self.df = self.df.asfreq('D').fillna(0)  # 处理缺失日期

    def save_result(self, data, category, filename, fmt='csv'):
        """统一保存结果到指定类别目录"""
        dir_path = os.path.join(RESULTS_DIR, SUB_DIRS[category])
        full_path = os.path.join(dir_path, f"{self.country}_{filename}")

        if fmt == 'csv':
            data.to_csv(full_path)
        elif fmt == 'txt':
            with open(full_path, 'w') as f:
                f.write(data)
        elif fmt == 'png':
            plt.savefig(full_path)
            plt.close()

    def basic_analysis(self):
        """基础统计分析"""
        stats = {
            'mean': self.df[self.cases_col].mean(),
            'std': self.df[self.cases_col].std(),
            'skewness': skew(self.df[self.cases_col]),
            'kurtosis': kurtosis(self.df[self.cases_col])
        }
        stats_str = "\n".join([f"{k}: {v:.4f}" for k, v in stats.items()])
        self.save_result(stats_str, 'stats', 'basic_stats.txt', 'txt')

    def trend_analysis(self):
        """趋势可视化分析"""
        # 原始趋势
        plt.figure(figsize=(12, 6))
        self.df[self.cases_col].plot(title=f"COVID-19 Trend - {self.country}")
        self.save_result(None, 'trends', 'raw_trend.png', 'png')

        # 移动平均
        ma7 = self.df[self.cases_col].rolling(7).mean()
        ma30 = self.df[self.cases_col].rolling(30).mean()
        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, self.df[self.cases_col], label='Daily', alpha=0.3)
        plt.plot(ma7.index, ma7, label='7-day MA', linewidth=2)
        plt.plot(ma30.index, ma30, label='30-day MA', linewidth=2)
        plt.title(f"Moving Average Comparison - {self.country}")
        plt.legend()
        self.save_result(None, 'trends', 'moving_average.png', 'png')

    def time_series_analysis(self):
        """时间序列分解"""
        decomposition = seasonal_decompose(self.df[self.cases_col], model='additive', period=30)
        decomposition.plot()
        self.save_result(None, 'trends', 'seasonal_decomposition.png', 'png')

    def arima_forecast(self):
        """ARIMA建模预测"""
        model = ARIMA(self.df[self.cases_col], order=(5, 1, 0))
        results = model.fit()
        forecast = results.get_forecast(steps=30)

        # 保存模型
        self.save_result(pd.Series(forecast.predicted_mean), 'forecasts', 'arima_forecast.csv')

        # 可视化
        plt.figure(figsize=(12, 6))
        self.df[self.cases_col].plot(label='Historical')
        forecast.predicted_mean.plot(label='Forecast')
        plt.title(f"ARIMA Forecast - {self.country}")
        self.save_result(None, 'forecasts', 'arima_forecast.png', 'png')

        return forecast.predicted_mean

    def holtwinters_forecast(self):
        """Holt-Winters预测"""
        try:
            model = ExponentialSmoothing(self.df[self.cases_col],
                                         seasonal='additive',
                                         seasonal_periods=7)
            results = model.fit()
            forecast = results.forecast(30)

            # 保存结果
            forecast_df = pd.DataFrame(
                {'Date': forecast.index, 'HW_Forecast': forecast.values}
            ).set_index('Date')
            self.save_result(forecast_df, 'forecasts', 'holtwinters_forecast.csv')

            # 可视化
            plt.figure(figsize=(12, 6))
            self.df[self.cases_col].plot(label='Historical')
            forecast.plot(label='Holt-Winters Forecast')
            plt.title(f"Holt-Winters Forecast - {self.country}")
            plt.legend()
            self.save_result(None, 'forecasts', 'holtwinters_forecast.png', 'png')

            return forecast.values
        except Exception as e:
            print(f"Holt-Winters failed for {self.country}: {str(e)}")
            return np.zeros(30)

    def random_forest_forecast(self):
        """随机森林预测"""
        try:
            # 创建滞后特征
            look_back = 14
            df = self.df[[self.cases_col]].copy()
            for i in range(1, look_back + 1):
                df[f'lag_{i}'] = df[self.cases_col].shift(i)

            df.dropna(inplace=True)

            X = df.drop(columns=[self.cases_col])
            y = df[self.cases_col]

            # 时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=5)
            model = RandomForestRegressor(n_estimators=100, random_state=42)

            # 训练最终模型
            model.fit(X, y)

            # 递归预测
            forecast = []
            last_window = X.iloc[-1].values

            for _ in range(30):
                pred = model.predict([last_window])[0]
                forecast.append(pred)
                last_window = np.roll(last_window, -1)
                last_window[-1] = pred

            forecast_dates = pd.date_range(self.df.index[-1], periods=31)[1:]

            # 保存结果
            forecast_df = pd.DataFrame(
                {'Date': forecast_dates, 'RF_Forecast': forecast}
            ).set_index('Date')
            self.save_result(forecast_df, 'forecasts', 'rf_forecast.csv')

            return np.array(forecast)
        except Exception as e:
            print(f"Random Forest forecast failed for {self.country}: {str(e)}")
            return np.zeros(30)

    def seir_modeling(self):
        """SEIR传染病模型拟合"""
        try:
            pop_density = 1e6  # 按百万标准化
            initial_infected = self.df[self.cases_col].iloc[0]

            def model(variables, t, beta, gamma):
                S, E, I, R = variables
                dSdt = -beta * S * I / pop_density
                dEdt = beta * S * I / pop_density - 0.2 * E
                dIdt = 0.2 * E - gamma * I
                dRdt = gamma * I
                return [dSdt, dEdt, dIdt, dRdt]

            # 参数优化
            def objective(params):
                beta, gamma = params
                solution = odeint(model,
                                  [pop_density - initial_infected, 0, initial_infected, 0],
                                  np.arange(len(self.df)),
                                  args=(beta, gamma))
                return np.mean((solution[:, 2] - self.df[self.cases_col]) ** 2)

            res = minimize(objective, [0.5, 0.1], bounds=[(0.001, 1), (0.001, 1)])
            beta_opt, gamma_opt = res.x

            # 生成拟合曲线
            solution = odeint(model,
                              [pop_density - initial_infected, 0, initial_infected, 0],
                              np.arange(len(self.df)),
                              args=(beta_opt, gamma_opt))

            # 可视化
            plt.figure(figsize=(12, 6))
            plt.plot(self.df.index, self.df[self.cases_col], label='Actual')
            plt.plot(self.df.index, solution[:, 2], label='SEIR Fit')
            plt.title(f"SEIR Model Fit - {self.country}")
            plt.legend()
            self.save_result(None, 'models', 'seir_fit.png', 'png')

            return beta_opt, gamma_opt
        except Exception as e:
            print(f"SEIR modeling failed for {self.country}: {str(e)}")
            return None, None

    def anomaly_detection(self):
        """异常值检测"""
        model = IsolationForest(contamination=0.05, random_state=42)
        anomalies = model.fit_predict(self.df[[self.cases_col]])

        # 可视化
        plt.figure(figsize=(12, 6))
        self.df[self.cases_col].plot(label='Normal')
        plt.scatter(self.df.index[anomalies == -1],
                    self.df[self.cases_col][anomalies == -1],
                    color='red', label='Anomaly')
        plt.title(f"Anomaly Detection - {self.country}")
        plt.legend()
        self.save_result(None, 'anomalies', 'anomalies.png', 'png')

    def run_full_analysis(self):
        """执行完整分析流程"""
        self.basic_analysis()
        self.trend_analysis()
        self.time_series_analysis()

        arima_fc = self.arima_forecast()
        hw_fc = self.holtwinters_forecast()
        rf_fc = self.random_forest_forecast()
        self.seir_modeling()
        self.anomaly_detection()

        # 多模型预测对比
        plt.figure(figsize=(15, 7))
        self.df[self.cases_col].plot(label='Historical')
        forecast_dates = pd.date_range(self.df.index[-1], periods=31)[1:]
        plt.plot(forecast_dates, arima_fc, label='ARIMA')
        plt.plot(forecast_dates, hw_fc, label='Holt-Winters')
        plt.plot(forecast_dates, rf_fc, label='Random Forest')
        plt.title(f"Forecast Comparison - {self.country}")
        plt.legend()
        self.save_result(None, 'comparisons', 'forecast_comparison.png', 'png')


# ======================
# 主执行程序
# ======================
def main():
    create_directory_structure()

    processed_countries = set(
        f.split('_')[0] for f in os.listdir(RESULTS_DIR) if '_stats2.txt' in f)

    for file in os.listdir(DATA_DIR):
        if file.endswith('.csv'):
            country_name = file[:-4]
            if country_name in processed_countries:
                continue

            print(f"Processing {country_name}...")
            try:
                analyzer = CountryAnalyzer(
                    country_name,
                    os.path.join(DATA_DIR, file))
                analyzer.run_full_analysis()
                print(f"Completed {country_name}")
            except Exception as e:
                print(f"Error processing {country_name}: {str(e)}")


if __name__ == "__main__":
    main()