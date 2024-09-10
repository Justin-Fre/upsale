import time  # to simulate a real time data, time loop
import random
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import pyodbc
from sklearn.preprocessing import LabelEncoder
coder = LabelEncoder()
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
lgbmodel = LGBMClassifier(random_state=42, num_class = 3)
import joblib
##from Feature_engineering import *
import base64
import os

st.set_page_config(
    page_title="Model predict  segment  of  customer",
    page_icon="✅",
    layout="wide",
)

# Tải mô hình đã lưu

def load_model(upsale):
    return joblib.load(upsale)

directory = '/Users/mac/Desktop/Project/upsale'  # Thay thế bằng đường dẫn thực tế
model_path = os.path.join(directory, 'upsale.pkl')
loaded_model = load_model(model_path)

# Tạo giao diện người dùng
st.title("Dự đoán Segment Khách Hàng")

# Nhập thông tin từ người dùng
recency = st.number_input("Recency")
frequency = st.number_input("Frequency")
monetary_value = st.number_input("Monetary Value")

# Dự đoán khi người dùng nhấn nút
if st.button("Dự Đoán"):
    input_data = pd.DataFrame([[recency, frequency, monetary_value]], columns=['Recency', 'Frequency', 'MonetaryValue'])
    prediction = loaded_model.predict(input_data)
    st.write(f"Kết quả Dự đoán: {prediction[0]}")

 # Thêm diễn giải cho kết quả
    if prediction[0] == "At Risk Customers":
        st.write("nhóm báo động, ngày gần đây mua là 13 ngày, tần suất mua không cao và giá trị cũng không cao. Nên chú ý nhóm khách hàng này và kích thích nhóm này bằng các chương trình mua sản phẩm này tặng sản khác kèm theo để tăng khả năng tiếp cận các sản phẩm vào nhóm khách hàng này")
    elif prediction[0] == "Best Customers":
        st.write("nhóm trung thành với công ty và thường xuyên sử dụng sản phẩm, ngày gần nhất đang sử dụng là 5,5 ngày gía trị mang lại cao. Nên có chương trình gọi điện cảm ơn khách hàng và chi ân khách hàng kịp thời.")
    elif prediction[0] == "Lost Customers":
        st.write("nhóm khách hàng đang rời bỏ tần suất mua cực thấp, ngày gần đây mua là 17 ngày chính vì thế giá trị đem lại cũng thấp nhất. Nên có khảo sát xem nguyên nhân từ đâu, gọi điện tư vấn và đưa ra kịch bản thuyết phục.")
    elif prediction[0] == "Loyal Customers":
        st.write("Nhóm khách hàng VIP của công ty tần suất mua lớn, ngày gần đây mua 2 ngày. Giống như nhóm best customer đây là nhóm trung thành nên có những chương trình ưu đãi đặc biệt cho nhóm sản phẩm này hoặc chi ân ngày sinh nhật để khách hàng cảm thấy chân trọng và sử dụng nhiều hơn.")