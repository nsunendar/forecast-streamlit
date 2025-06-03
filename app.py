
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from models import models

st.set_page_config(layout="wide")
st.title("📦 Forecast Permintaan Barang per Produk/Area (6 Bulan ke Depan)")

uploaded_file = st.file_uploader("📁 Upload file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M').dt.to_timestamp()

    st.success("File berhasil dimuat. Kolom tersedia: " + ", ".join(df.columns))

    # Pilihan kolom area/produk
    area_col = st.selectbox("🧭 Pilih kolom untuk Area/Produk", options=[col for col in df.columns if col not in ['Date', 'Order_permintaan']])
    unique_areas = df[area_col].unique()
    selected_area = st.selectbox("🏷 Pilih Area/Produk", options=unique_areas)

    filtered = df[df[area_col] == selected_area]
    ts = filtered.groupby('Month')['Order_permintaan'].sum().asfreq('MS')

    st.subheader(f"📈 Data Historis - {selected_area}")
    st.line_chart(ts)

    st.subheader("🔮 Forecast 6 Bulan - Semua Model")
    model_funcs = {
        "SARIMA": models.forecast_sarima,
        "Prophet": models.forecast_prophet,
        "LSTM": lambda s: models.forecast_dl(s, model_type='lstm'),
        "GRU": lambda s: models.forecast_dl(s, model_type='gru'),
        "Transformer": models.forecast_transformer
    }

    results = {}
    for model_name, func in model_funcs.items():
        with st.spinner(f"Running {model_name}..."):
            try:
                pred, evals = func(ts)
                results[model_name] = (pred, evals)
                st.markdown(f"**{model_name}**  
RMSE: `{evals['RMSE']:.2f}` | R²: `{evals['R2']:.4f}`")
                fig, ax = plt.subplots()
                ax.plot(ts[-12:], label='Aktual')
                ax.plot(pred, label='Forecast')
                ax.set_title(model_name)
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"{model_name} gagal: {e}")
