import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import os

API_URL   = os.getenv('API_URL', 'http://localhost:8000')
DATA_PATH = 'data/forecast_test.csv'

st.set_page_config(page_title='Helios Power Forecast', layout='wide')
st.title('Helios — Power Forecast')

# Health check
try:
    res = requests.get(f'{API_URL}/health', timeout=3)
    st.success('API is online') if res.ok else st.error('API is reachable but unhealthy')
except requests.exceptions.ConnectionError:
    st.error('Cannot reach the API')
    st.stop()

st.markdown('---')

if st.button('Prediction', use_container_width=True):
    with st.spinner('Running predictions...'):
        try:
            df      = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
            payload = {'rows': df.assign(timestamp=df['timestamp'].astype(str)).to_dict(orient='records')}
            res     = requests.post(f'{API_URL}/predict', json=payload, timeout=30)

            if res.status_code == 200:
                df['predicted_mw'] = res.json()['predictions']

                fig, ax = plt.subplots(figsize=(14, 4))
                ax.plot(df['timestamp'], df['predicted_mw'], linewidth=0.9, color='steelblue', label='Predicted MW')
                ax.set_title('Power Forecast')
                ax.set_xlabel('Timestamp')
                ax.set_ylabel('MW')
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)

            else:
                st.error(f'API error {res.status_code}: {res.json().get("detail")}')

        except Exception as e:
            st.error(f'Something went wrong: {e}')
