import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
import mplfinance as mpf
from BBDD import create_connection

def download_stock_data(symbol, start_date, end_date, conn):
    # Verificar si los datos ya existen en la base de datos
    cursor = conn.cursor()
    query = f"SELECT COUNT(*) FROM datos_financieros WHERE simbolo = '{symbol}' AND fecha >= '{start_date}' AND fecha <= '{end_date}'"
    cursor.execute(query)
    count = cursor.fetchone()[0]
    cursor.close()

    if count > 0:
        # Los datos ya existen en la base de datos, no es necesario descargarlos nuevamente
        # Obtener los datos almacenados de la base de datos ordenados por fecha de manera descendente
        cursor = conn.cursor()
        query = f"SELECT fecha, apertura, alto, bajo, cierre, volumen FROM datos_financieros WHERE simbolo = '{symbol}' AND fecha >= '{start_date}' AND fecha <= '{end_date}' ORDER BY fecha DESC"
        cursor.execute(query)
        data = cursor.fetchall()
        cursor.close()
    else:
        # Descargar datos de la API de Yahoo Finance
        df_yfinance = yf.download(symbol, start=start_date, end=end_date)

        # Almacenar datos en la base de datos
        cursor = conn.cursor()
        records = []
        for index, row in df_yfinance.iterrows():
            record = (
                symbol,
                index.date(),
                row['Open'],
                row['High'],
                row['Low'],
                row['Close'],
                row['Volume']
            )
            records.append(record)
        query = f"INSERT INTO datos_financieros (simbolo, fecha, apertura, alto, bajo, cierre, volumen) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        cursor.executemany(query, records)
        conn.commit()
        cursor.close()

        # Obtener los datos almacenados de la base de datos ordenados por fecha de manera descendente
        cursor = conn.cursor()
        query = f"SELECT fecha, apertura, alto, bajo, cierre, volumen FROM datos_financieros WHERE simbolo = '{symbol}' AND fecha >= '{start_date}' AND fecha <= '{end_date}' ORDER BY fecha DESC"
        cursor.execute(query)
        data = cursor.fetchall()
        cursor.close()

    # Crear DataFrame con los datos obtenidos
    df = pd.DataFrame(data, columns=['Fecha', 'Apertura', 'Alto', 'Bajo', 'Cierre', 'Volumen'])
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df.sort_values('Fecha', ascending=False, inplace=True)
    return df

def normalize_data(df):
    if df.empty:
        return df, None

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df.iloc[:, 1:])  # Excluir la columna de fecha en la normalización

    return df_scaled, scaler


def prepare_data(df_scaled, forecast_out):
    if len(df_scaled) == 0:
        return np.array([]), np.array([])

    X = []
    y = []

    for i in range(forecast_out, len(df_scaled)):
        X.append(df_scaled[i-forecast_out:i, :])
        y.append(df_scaled[i, 0])

    X = np.array(X)
    y = np.array(y)

    return X, y


def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def train_model(model, X_train, y_train, batch_size, epochs):
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

def make_predictions(model, X_test, scaler, df):
    predictions = model.predict(X_test)
    num_features = df.shape[1] - 1  # Número de características en el DataFrame

    # Ajustar las dimensiones de los datos concatenados
    if predictions.shape[1] != num_features:
        zeros = np.zeros((predictions.shape[0], num_features - 1))
        concatenated = np.concatenate((predictions, zeros), axis=1)
    else:
        concatenated = predictions

    # Realizar la inversión del escalado
    predictions = scaler.inverse_transform(concatenated)[:, 0]
    return predictions

def generate_plots(df, predictions, symbol):
    df_copy = df.copy()
    df_copy.set_index('Fecha', inplace=True)  # Establecer 'Fecha' como el índice

    rsi_indicator = RSIIndicator(df_copy['Cierre'], window=14)
    df_copy['RSI'] = rsi_indicator.rsi()
    df_copy['Buy_Signal'] = np.where(df_copy['RSI'] < 30, 1, 0)
    df_copy['Sell_Signal'] = np.where(df_copy['RSI'] > 70, 1, 0)

    predictions_df = pd.DataFrame(predictions, columns=["Prediction"])
    predictions_df.to_excel(f"predictions_{symbol}.xlsx", index=False)

    plt.figure(figsize=(14, 5))
    plt.plot(predictions, color='red', label='Predictions')
    plt.title(f'Predictions of the closing price for {symbol}')
    plt.legend()
    plt.show()

    mpf.plot(df_copy, type='candle', volume=True, style='yahoo', show_nontrading=True, warn_too_much_data=1000,
             columns=['Apertura', 'Alto', 'Bajo', 'Cierre', 'Volumen'])

def analyze_stocks(symbols, forecast_out, start_date, end_date):
    conn = create_connection()

    for symbol in symbols:
        df = download_stock_data(symbol, start_date, end_date, conn)

        if df.empty:
            print(f"No se encontraron datos para el símbolo {symbol}.")
            continue

        df_scaled, scaler = normalize_data(df)
        X, y = prepare_data(df_scaled, forecast_out)

        if len(X) == 0:
            print(f"No hay suficientes datos para el símbolo {symbol} después de la preparación.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = create_lstm_model(input_shape)
        train_model(model, X_train, y_train, batch_size=64, epochs=100)
        predictions = make_predictions(model, X_test, scaler, df)
        generate_plots(df, predictions, symbol)
        
        # Cálculo y evaluación del RSI
        df_copy = df.copy()
        rsi_indicator = RSIIndicator(df_copy['Cierre'], window=14)
        df_copy['RSI'] = rsi_indicator.rsi()
        last_rsi = df_copy['RSI'].iloc[-1]
        
        if last_rsi < 30:
            print(f"For {symbol}: BUY")
        elif last_rsi > 70:
            print(f"For {symbol}: SELL")
        else:
            print(f"For {symbol}: HOLD")

    conn.close()

# Configuración de parámetros
# symbols = ['AAPL', 'GOOGL', 'AMZN']
symbols = ['BTC']
forecast_out = 30
start_date = '2018-01-01'
end_date = '2023-06-22'

# Llamada a la función principal para analizar las acciones
analyze_stocks(symbols, forecast_out, start_date, end_date)
