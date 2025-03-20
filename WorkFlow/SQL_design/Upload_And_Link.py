import os
import mysql.connector
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import hashlib

# ======================
# 配置区
# ======================
DB_CONFIG = {
    "user": "root",
    "password": "",
    "host": "localhost",
    "port": 3306,
    "database": "covid19",  # 统一使用小写数据库名
    "raise_on_warnings": True,
    # "use_pure": True        # 避免连接器兼容性问题
}


# 使用原始字符串处理Windows路径
DATA_DIR = Path(r"D:\python\Code\DPW_COVID_19\Data\country_data")
RESULTS_DIR = Path(r"D:\python\Code\DPW_COVID_19\Analysis_Results2")
CHUNK_SIZE = 5000
VALID_COLUMNS = [
    'Day',
    'Daily new confirmed cases of COVID-19 per million people (rolling 7-day average, right-aligned)'
]

# 匹配分析代码中的目录结构
SUB_DIRS = {
    'stats': 'statistics',
    'trends': 'visualizations/trends',
    'models': 'model_outputs',
    'forecasts': 'forecasts',
    'anomalies': 'anomalies',
    'comparisons': 'visualizations/comparisons'
}

# ======================
# 日志配置
# ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("db_sync.log"),
        logging.StreamHandler()
    ]
)

class COVIDDatabase:
    def __init__(self):
        self.conn = None
        self._connect()
        self._init_db()

    def _connect(self):
        """建立数据库连接"""
        try:
            self.conn = mysql.connector.connect(**DB_CONFIG)
            logging.info(f"成功连接到数据库 {DB_CONFIG['database']}")
        except mysql.connector.Error as err:
            logging.error(f"数据库连接失败: {err}")
            raise

    def _init_db(self):
        """强化表创建逻辑"""
        schema = [
            # 国家表
            """CREATE TABLE IF NOT EXISTS `countries` (
                `country_id` INT AUTO_INCREMENT PRIMARY KEY,
                `country_name` VARCHAR(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci UNIQUE,
                `data_hash` CHAR(64),
                `last_updated` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",

            # 数据文件表（修正表名和字段）
            """CREATE TABLE IF NOT EXISTS `data_files` (
                `file_id` INT AUTO_INCREMENT PRIMARY KEY,
                `country_id` INT,
                `file_path` VARCHAR(511) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin UNIQUE,
                `file_size` BIGINT,
                `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (`country_id`) REFERENCES `countries`(`country_id`)
                    ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

            # 分析链接表
            """CREATE TABLE IF NOT EXISTS `analysis_links` (
                `link_id` INT AUTO_INCREMENT PRIMARY KEY,
                `country_id` INT,
                `analysis_type` ENUM('stats', 'trends', 'models', 'forecasts', 'anomalies', 'comparisons'),
                `file_path` VARCHAR(511) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin,
                `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY `unique_link` (`country_id`, `analysis_type`),
                FOREIGN KEY (`country_id`) REFERENCES `countries`(`country_id`)
                    ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""
        ]

        with self.conn.cursor() as cursor:
            for table_sql in schema:
                try:
                    cursor.execute(table_sql)
                    self.conn.commit()
                    logging.info(f"成功创建/验证表：{table_sql.split('(')[0].split()[-1]}")
                except mysql.connector.Error as err:
                    logging.critical(f"表创建失败: {err}\nSQL: {table_sql}")
                    self.conn.rollback()
                    raise RuntimeError("数据库初始化失败") from err

    def _get_country_id(self, country_name):
        """获取国家ID，不存在则创建"""
        query = "SELECT country_id FROM countries WHERE country_name = %s"
        insert = "INSERT INTO countries (country_name) VALUES (%s)"

        with self.conn.cursor() as cursor:
            cursor.execute(query, (country_name,))
            result = cursor.fetchone()
            if result:
                return result[0]

            cursor.execute(insert, (country_name,))
            self.conn.commit()
            return cursor.lastrowid

    def process_data_files(self):
        """处理数据文件更新（仅元数据）"""
        for file in DATA_DIR.glob("*.csv"):
            try:
                country_name = file.stem  # 直接使用文件名作为国家名
                file_hash = self._calculate_file_hash(file)

                if self._is_file_unchanged(country_name, file_hash):
                    logging.info(f"数据未变化: {country_name}")
                    continue

                self._update_country_metadata(country_name, file_hash)
                self._create_file_link(country_name, file)

            except Exception as e:
                logging.error(f"处理文件 {file.name} 失败: {str(e)}")

    def _calculate_file_hash(self, file_path):
        """计算文件内容哈希值"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(4096):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _is_file_unchanged(self, country_name, current_hash):
        """检查数据是否变化"""
        query = """SELECT data_hash FROM countries 
                   WHERE country_name = %s AND data_hash = %s"""
        with self.conn.cursor() as cursor:
            cursor.execute(query, (country_name, current_hash))
            return cursor.fetchone() is not None

    def _update_country_metadata(self, country_name, file_hash):
        """更新元数据"""
        update_sql = """
            UPDATE countries 
            SET data_hash = %s, last_updated = %s 
            WHERE country_name = %s
        """
        with self.conn.cursor() as cursor:
            cursor.execute(update_sql, (
                file_hash,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                country_name
            ))
            self.conn.commit()

    def _create_file_link(self, country_name, file_path):
        """最终修正的文件链接创建方法"""
        country_id = self._get_country_id(country_name)
        rel_path = os.path.relpath(file_path, start=DATA_DIR)

        # 修正后的SQL语句（明确指定字段来源）
        insert_sql = """
            INSERT INTO `data_files` (country_id, file_path, file_size)
            VALUES (%s, %s, %s) AS new_data
            ON DUPLICATE KEY UPDATE 
                file_size = new_data.file_size,
                created_at = IF(`data_files`.file_size != new_data.file_size, 
                               CURRENT_TIMESTAMP, 
                               `data_files`.created_at)
        """
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(insert_sql, (
                    country_id,
                    str(rel_path),
                    os.path.getsize(file_path)
                ))
                self.conn.commit()
        except mysql.connector.Error as err:
            logging.error(f"创建文件链接失败: {err}")
            raise

    def link_analysis_results(self):
        """最终修正的分析结果链接方法"""
        for analysis_type, sub_dir in SUB_DIRS.items():
            result_dir = RESULTS_DIR / sub_dir
            if not result_dir.exists():
                continue

            for result_file in result_dir.glob("*.*"):
                # 增强文件名解析逻辑
                filename = result_file.stem
                if '_forecast' in filename:
                    country_name = filename.rsplit('_forecast', 1)[0]
                elif '_comparison' in filename:
                    country_name = filename.rsplit('_comparison', 1)[0]
                else:
                    country_name = filename.split('_')[0]

                try:
                    country_id = self._get_country_id(country_name)
                    rel_path = result_file.relative_to(RESULTS_DIR)

                    # 修正后的SQL语句
                    insert_sql = """
                        INSERT INTO `analysis_links` 
                        (country_id, analysis_type, file_path)
                        VALUES (%s, %s, %s) AS new_link
                        ON DUPLICATE KEY UPDATE 
                            file_path = new_link.file_path,
                            updated_at = IF(`analysis_links`.file_path != new_link.file_path,
                                          CURRENT_TIMESTAMP,
                                          `analysis_links`.updated_at)
                    """
                    with self.conn.cursor() as cursor:
                        cursor.execute(insert_sql, (
                            country_id,
                            analysis_type,
                            str(rel_path)
                        ))
                        self.conn.commit()

                except Exception as e:
                    logging.error(f"链接分析结果失败 {result_file}: {str(e)}")

    def close(self):
        """显式关闭连接"""
        if self.conn and self.conn.is_connected():
            self.conn.close()
            logging.info("数据库连接已关闭")

    def __del__(self):
        try:
            if self.conn and self.conn.is_connected():
                self.conn.close()
        except AttributeError:
            pass

# ======================
# 执行主程序
# ======================
if __name__ == "__main__":
    db = None
    try:
        db = COVIDDatabase()
        db.process_data_files()
        db.link_analysis_results()
        logging.info("数据同步完成")
    except Exception as e:
        logging.critical(f"主程序错误: {str(e)}")
    finally:
        if db:
            db.close()