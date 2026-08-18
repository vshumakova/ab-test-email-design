"""
Модуль для загрузки данных проекта
"""
import pandas as pd
from pathlib import Path
import os

try:
    current_dir = Path(__file__).parent
    DATA_DIR = current_dir.parent / 'data'
    
    if not DATA_DIR.exists():
        DATA_DIR = current_dir / 'data'
        
    if not DATA_DIR.exists():
        DATA_DIR = Path(os.getcwd()) / 'data'
        
except NameError:
    # Если __file__ не определен
    DATA_DIR = Path('data')


def load_economic_data():
    """
    Загрузка экономических данных
    
    Returns:
    --------
    pd.DataFrame: Данные с колонками:
        - emails_sent: количество отправленных писем
        - open_rate: доля открытых писем
        - ctr: конверсия из открытия в клик
        - conversion_to_purchase: конверсия из клика в покупку
        - avg_order_value: средний чек
        - revenue: выручка
    """
    file_path = DATA_DIR / 'economic_data.csv'
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    df = pd.read_csv(file_path)
    return df


def load_monitoring_data():
    """
    Загрузка данных мониторинга (первый эксперимент)
    
    Returns:
    --------
    pd.DataFrame: Данные с колонками:
        - date: дата
        - user_id: ID пользователя
        - group: группа (control/treatment)
        - converted: совершил ли клик (0/1)
    """
    file_path = DATA_DIR / 'monitoring.csv'
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    return df


def load_results_data():
    """
    Загрузка результатов эксперимента (второй эксперимент)
    
    Returns:
    --------
    pd.DataFrame: Данные с колонками:
        - date: дата
        - user_id: ID пользователя
        - group: группа (control/treatment)
        - user_type: тип пользователя (new/old)
        - converted: совершил ли клик (0/1)
    """
    file_path = DATA_DIR / 'results.csv'
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    return df


def load_all_data():
    """
    Загрузка всех данных проекта
    
    Returns:
    --------
    dict: Словарь с тремя датафреймами
    """
    return {
        'economic': load_economic_data(),
        'monitoring': load_monitoring_data(),
        'results': load_results_data()
    }


def get_data_info():
    """
    Получение информации о загруженных данных
    """
    try:
        data = load_all_data()
        info = {}
        for name, df in data.items():
            info[name] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'missing': df.isnull().sum().sum(),
                'memory_usage': df.memory_usage(deep=True).sum() / 1024  # KB
            }
        return info
    except FileNotFoundError as e:
        return {'error': str(e)}


if __name__ == '__main__':
    # Тест загрузки
    print("=== Проверка загрузки данных ===\n")
    try:
        data = load_all_data()
        print(" Данные загружены:\n")
        for name, df in data.items():
            print(f"  {name}: {len(df):,} строк, {len(df.columns)} колонок")
            print(f"     Колонки: {', '.join(df.columns)}")
            print()
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        print(f"   Ищите файлы в: {DATA_DIR.absolute()}")
