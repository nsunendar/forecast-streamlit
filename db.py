
import mysql.connector
from mysql.connector import Error
from datetime import datetime

# Fungsi membuat koneksi (isi parameter sesuai konfigurasi Anda)
def create_connection():
    try:
        connection = mysql.connector.connect(
            host="your_host",
            port=3306,
            database="your_database",
            user="your_username",
            password="your_password"
        )
        return connection
    except Error as e:
        print(f"Koneksi gagal: {e}")
        return None

# Fungsi insert hasil forecast ke MySQL
def insert_forecast_results(connection, data):
    try:
        cursor = connection.cursor()
        sql = '''
        INSERT INTO forecast_results (area, model, forecast_date, forecast_value, rmse, r2, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        '''
        cursor.executemany(sql, data)
        connection.commit()
        return True
    except Error as e:
        print(f"Gagal insert: {e}")
        return False
