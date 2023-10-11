import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from babel.numbers import format_currency

st.header('Proyek Akhir Data analyst with python')

def create_rfm_df(df):
    rfm_df = df.groupby('customer_id', as_index=False).agg({
        'order_purchase_timestamp': 'max',
        'order_id': 'nunique',
        'price': 'sum'
    })

    rfm_df.columns = ['customer_id', 'max_purchase_timestamp', 'frequency', 'monetary']
    analysis_date = pd.to_datetime('2023-10-02')
    rfm_df['max_purchase_timestamp'] = pd.to_datetime(rfm_df['max_purchase_timestamp']).dt.date
    rfm_df['recency'] = (analysis_date - pd.to_datetime(rfm_df['max_purchase_timestamp'])).dt.days
    rfm_df.drop('max_purchase_timestamp', axis=1, inplace=True)

    return rfm_df

def create_segmented_df(df):
    segmented_df = df.groupby('customer_id', as_index=False).agg({
        'order_purchase_timestamp': 'max',
        'order_id': 'nunique',
        'price': 'sum'
    })

    segmented_df.columns = ['customer_id', 'max_purchase_timestamp', 'frequency', 'monetary']
    analysis_date = pd.to_datetime('2023-10-02')
    segmented_df['max_purchase_timestamp'] = pd.to_datetime(segmented_df['max_purchase_timestamp']).dt.date
    segmented_df['recency'] = (analysis_date - pd.to_datetime(segmented_df['max_purchase_timestamp'])).dt.days
    segmented_df.drop('max_purchase_timestamp', axis=1, inplace=True)

    segmented_df['r_rank'] = segmented_df['recency'].rank(ascending=False)
    segmented_df['f_rank'] = segmented_df['frequency'].rank(ascending=True)
    segmented_df['m_rank'] = segmented_df['monetary'].rank(ascending=True)

    segmented_df['r_rank_norm'] = (segmented_df['r_rank'] / segmented_df['r_rank'].max()) * 100
    segmented_df['f_rank_norm'] = (segmented_df['f_rank'] / segmented_df['f_rank'].max()) * 100
    segmented_df['m_rank_norm'] = (segmented_df['m_rank'] / segmented_df['m_rank'].max()) * 100

    segmented_df.drop(columns=['r_rank', 'f_rank', 'm_rank'], inplace=True)

    segmented_df['RFM_score'] = 0.15 * segmented_df['r_rank_norm'] + 0.28 * segmented_df['f_rank_norm'] + 0.57 * segmented_df['m_rank_norm']
    segmented_df['RFM_score'] *= 0.05
    segmented_df = segmented_df.round(2)

    segmented_df["customer_segment"] = np.where(
        segmented_df['RFM_score'] > 4.5, "Top customers", (np.where(
            segmented_df['RFM_score'] > 4, "High value customer", (np.where(
                segmented_df['RFM_score'] > 3, "Medium value customer", np.where(
                    segmented_df['RFM_score'] > 1.6, 'Low value customers', 'Lost customers'))))))

    return segmented_df

def visualize_customer_segments(df):
    plt.figure(figsize=(10, 5))

    sns.barplot(
        x="customer_segment",
        y="customer_id",
        data=df.groupby("customer_segment", as_index=False).count().sort_values(by="customer_id", ascending=False)
    )
    plt.title("Number of Customer for Each Segment", loc="center", fontsize=15)
    plt.ylabel(None)
    plt.xlabel(None)
    plt.tick_params(axis='y', labelsize=12)
    plt.xticks(rotation=45)  # Mengatur rotasi label sumbu x agar lebih mudah dibaca

    # Simpan gambar ke dalam variabel
    img = plt.gcf()

    # Tampilkan gambar menggunakan st.pyplot
    st.pyplot(img)

def efficiency_by_region(df):
    efficiency_by_region = df.groupby('geolocation_city')['delivery_duration'].mean().reset_index()
    efficiency_by_region = efficiency_by_region.sort_values(by='delivery_duration', ascending=True)
    return efficiency_by_region

# load data 
merge_rfm = pd.read_csv("https://raw.githubusercontent.com/sanding28/dataset-dico-sub/main/merge_rfm.csv")

rfm_df = create_rfm_df(merge_rfm)

segmented_df = create_segmented_df(merge_rfm)
# orders_dataset_df = pd.read_csv("https://raw.githubusercontent.com/sanding28/dataset-dico-sub/main/orders_dataset_clean.csv")
# orders_item_df = pd.read_csv("https://raw.githubusercontent.com/sanding28/dataset-dico-sub/main/orders_item.csv")
# customers_df = pd.read_csv("https://raw.githubusercontent.com/sanding28/dataset-dico-sub/main/customer_df.csv")
# geolocation_df = pd.read_csv("https://raw.githubusercontent.com/sanding28/dataset-dico-sub/main/geolocation_clean_df.csv")

# merge_geo_df = pd.merge(orders_dataset_df, orders_item_df, on='order_id')
# merge_geo_df = pd.merge(merge_geo_df, customers_df, on='customer_id')

# merge_geo_df['order_purchase_timestamp'] = pd.to_datetime(merge_geo_df['order_purchase_timestamp'])
# merge_geo_df['order_delivered_customer_date'] = pd.to_datetime(merge_geo_df['order_delivered_customer_date'])

# merge_geo_df['delivery_duration'] = (merge_geo_df['order_delivered_customer_date'] - merge_geo_df['order_purchase_timestamp']).dt.days

# merge_geo_df = pd.merge(merge_geo_df, geolocation_df, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix')

# merge_geo = pd.read_csv("merge_geo_df.csv")
#geo_df = efficiency_by_region(merge_geo_df)

st.subheader("1. Berapa jumlah uang yang dihabiskan pelanggan dengan jumlah value signifikan?")

columns = st.columns(3)  

with columns[0]:
    avg_recency = round(rfm_df.recency.mean(), 1)
    st.metric("By Recency", value=avg_recency)

with columns[1]:
    avg_frequency = round(rfm_df.frequency.mean(), 2)
    st.metric("By Frequency", value=avg_frequency)

with columns[2]:
    avg_monetary = format_currency(
        rfm_df.monetary.mean(), "AUD", locale='es_CO')
    st.metric("Average Monetary", value=avg_monetary)

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 20)) 

sns.barplot(x="recency", y="customer_id", data=rfm_df.sort_values(by="recency", ascending=True).head(5), ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel("Customer_id")
ax[0].set_title("By recency", loc="center", fontsize=18)
ax[0].tick_params(axis='y', labelsize=12)
ax[0].tick_params(axis='x', labelsize=12)

sns.barplot(x="frequency", y="customer_id", data=rfm_df.sort_values(by="frequency", ascending=False).head(5), ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("Customer_id")
ax[1].set_title("By Frequency", loc="center", fontsize=18)
ax[1].tick_params(axis='y', labelsize=12)
ax[1].tick_params(axis='x', labelsize=12)

sns.barplot(x="monetary", y="customer_id", data=rfm_df.sort_values(by="monetary", ascending=False).head(5), ax=ax[2])
ax[2].set_ylabel(None)
ax[2].set_xlabel("Customer_id")
ax[2].set_title("By monetary", loc="center", fontsize=18)
ax[2].tick_params(axis='y', labelsize=12)
ax[2].tick_params(axis='x', labelsize=12)

st.pyplot(fig)

st.subheader("2. Bagaimana segmentasi pelanggan dari e-commerce ini?")
visualize_customer_segments(segmented_df)

# st.subheader("3. Kota mana dengan pengiriman tercepat?")
# plt.figure(figsize=(6, 6))
# sns.barplot(y="delivery_duration", x="geolocation_city", data=merge_geo_df.sort_values(by="delivery_duration", ascending=True).head(5))
# plt.ylabel(None)
# plt.xlabel(None)
# plt.title("Top 5 Cities by Delivery Efficiency", fontsize=18)
# plt.xlabel("City", fontsize=15, )
# plt.ylabel("Delivery Duration (days)", fontsize=15)
# plt.xticks(rotation=45)
# st.pyplot(plt)