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
import matplotlib.pyplot as plt

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

# tải dữ  liệu  chọnn
path = '/Users/mac/Desktop/Project/upsale/lost_customer.csv'
lost_customer = pd.read_csv(path)

path_1 = '/Users/mac/Desktop/Project/upsale/best_customer.csv'
best_customer = pd.read_csv(path_1)

path_2 = '/Users/mac/Desktop/Project/upsale/at_risk_customers.csv'
at_risk_customer = pd.read_csv(path_2)

path_3 = '/Users/mac/Desktop/Project/upsale/loyal_customers.csv'
loyal_customers = pd.read_csv(path_3)


# Tạo giao diện người dùng
st.title("🔻 Dự đoán hành vi mua hàng của khách hàng")
st.markdown('''
    :blue[ Sản phẩm  của  luonglv]''')


st.button("🙍‍♂️ Nhập Thông Tin Mua Hàng Vào Đây")
# Nhập thông tin từ người dùng
recency = st.number_input("Recency")
frequency = st.number_input("Frequency")
monetary_value = st.number_input("Monetary Value")

# Dự đoán khi người dùng nhấn nút
if st.button("🎯 Dự Đoán"):
    input_data = pd.DataFrame([[recency, frequency, monetary_value]], columns=['Recency', 'Frequency', 'MonetaryValue'])
    prediction = loaded_model.predict(input_data)
    st.write(f"Kết quả Dự đoán: {prediction[0]}")

 # Thêm diễn giải cho kết quả
    if prediction[0] == "At Risk Customers":
        st.write("Nhóm báo động, ngày gần đây mua là khá xa, tần suất mua không cao và giá trị cũng không cao. Nên chú ý nhóm khách hàng này và kích thích nhóm này bằng các chương trình mua sản phẩm này tặng sản khác kèm theo để tăng khả năng tiếp cận các sản phẩm vào nhóm khách hàng này")
    elif prediction[0] == "Best Customers":
        st.write("Nhóm tiềm  năng với công ty và thường xuyên sử dụng sản phẩm, ngày gần nhất đang sử dụng nhóm  này là tập trung phổ biến 5,5 ngày gần  đâyy giá trị mang lại cao. Nên có chương trình gọi điện cảm ơn khách hàng và chi ân khách hàng kịp thời.")
    elif prediction[0] == "Lost Customers":
        st.write("Nhóm khách hàng đang rời bỏ tần suất mua cực thấp, ngày gần đây mua là tập trung phổ biến 17 ngày gần  đây chính vì thế giá trị đem lại cũng thấp nhất. Nên có khảo sát xem nguyên nhân từ đâu, gọi điện tư vấn và đưa ra kịch bản thuyết phục.")
    elif prediction[0] == "Loyal Customers":
        st.write("Nhóm khách hàng VIP của công ty tần suất mua lớn, ngày gần đây mua tập trung nhiều 2 ngày gân  đây. Giống như nhóm best customer đây là nhóm trung thành nên có những chương trình ưu đãi đặc biệt cho nhóm sản phẩm này hoặc chi ân ngày sinh nhật để khách hàng cảm thấy chân trọng và sử dụng nhiều hơn.")
    
    st.button("💰 Lịch Sử Mua Hàng Trước Đây Nhóm Khách Hàng Này")

        #vẽ  biểu  đồ  lịch  sử  mua  hàng  trước  đây
    if prediction[0] == "At Risk Customers":
        # Vẽ biểu đồ
        plt.figure(figsize=(10, 6))

# Vẽ đường cho từng phân khúc
        for product in at_risk_customer.index:
            plt.plot(at_risk_customer.columns, at_risk_customer.loc[product], marker='o', label=product)

            plt.title('Giá Trị Mua Hàng Của Tập At Risk Customers Theo Tháng Cho Các Sản Phẩm')
            plt.xlabel('Tháng')
            plt.ylabel('Giá Trị')
            plt.legend(title='Sản phẩm')
            plt.xticks(rotation=45)

# Hiển thị biểu đồ trong Streamlit
            st.pyplot(plt)
    if prediction[0] == "Best Customers":
        # Vẽ biểu đồ
        plt.figure(figsize=(10, 4))

# Vẽ đường cho từng phân khúc
        for product in best_customer.index:
            plt.plot(best_customer.columns, best_customer.loc[product], marker='o', label=product)

            plt.title('Giá Trị Mua Hàng Của Tập Best Customers Theo Tháng Cho Các Sản Phẩm')
            plt.xlabel('Tháng')
            plt.ylabel('Giá Trị')
            plt.legend(title='Sản phẩm')
            plt.xticks(rotation=45)

# Hiển thị biểu đồ trong Streamlit
            st.pyplot(plt)
    if prediction[0] == "Lost Customers":
        # Vẽ biểu đồ
        plt.figure(figsize=(10, 4))

# Vẽ đường cho từng sản phẩm
        for product in lost_customer.index:
            plt.plot(lost_customer.columns, lost_customer.loc[product], marker='o', label=product)

            plt.title('Giá Trị Mua Hàng Của Tập Lost Customers Theo Tháng Cho Các Sản Phẩm')
            plt.xlabel('Tháng')
            plt.ylabel('Giá Trị')
            plt.legend(title='Sản phẩm')
            plt.xticks(rotation=45)

# Hiển thị biểu đồ trong Streamlit
            st.pyplot(plt)

    if prediction[0] == "Loyal Customers":
        # Vẽ biểu đồ
        plt.figure(figsize=(10, 4))

# Vẽ đường cho từng phân khúc
        for product in loyal_customers.index:
            plt.plot(loyal_customers.columns, loyal_customers.loc[product], marker='o', label=product)

            plt.title('Giá Trị Mua Hàng Của Tập Loyal Customers Theo Tháng Cho Các Sản Phẩm')
            plt.xlabel('Tháng')
            plt.ylabel('Giá Trị')
            plt.legend(title='Sản phẩm')
            plt.xticks(rotation=45)

# Hiển thị biểu đồ trong Streamlit
            st.pyplot(plt)

    st.button("💹 Gợi ý tư vấn mua sản phẩm mới hoặc mua thêm")
    if prediction[0] == "At Risk Customers":
        st.write("Tại tháng 09 nhóm khách hàng này có  xu  hướng  dùng  sản  phẩm 2 thấp nhưng  bù  lại  sản  phẩm  1 tăng đột  biến, nên  tư  vấn  dòng  sản  phẩm  2 và duy  trì  sản  phẩm  0.")
    elif prediction[0] == "Best Customers":
        st.write("Nhóm khách  hàng  tại  tháng  09 có  xu  hướng  sử  dụng  sản  phẩm  2 giảm mạnhh, sản  phẩm  0 đang phục  hồi, sản phẩm  3 ổn định. Nên tư vân  mua  thêm  sản  phẩm  0.")
    elif prediction[0] == "Lost Customers":
        st.write("Nhóm khách hàng này trong  tháng  09 có  xu  hướng  sử  dụng  2 sản  phẩm 2 và 0, nên tập  trung  tư  vấn  nhóm  02 vì  tháng  10 có  thể  giảm. ")
    elif prediction[0] == "Loyal Customers":
        st.write("Nhóm khách hàng này trong  tháng  09 có  như  cầu  sử  dụng  3 sản phẩm  như nhau, nên  chú  trọng  cách  thức  trọng  sóc.")

st.button("🎈 Gợi ý chi  tiết  sản  phẩm")
