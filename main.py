import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import streamlit as st
from sklearn.preprocessing import OneHotEncoder,LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, roc_auc_score

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from itertools import combinations

warnings.filterwarnings("ignore")
# Thiết lập page
st.set_page_config(     
    page_title="ĐỘI COPILOT",      
    page_icon="🧊"  
)

# Thiết lập tiêu đề
st.subheader('Đánh giá các chỉ số của các mô hình học máy')

df1 = pd.read_csv("dataset/train_1.csv")
df2 = pd.read_csv("dataset/train_2.csv")
df_predict = pd.read_csv("dataset/predict.csv")

df_train = pd.merge(df1, df2, on='Case_ID', how='inner')

df = pd.concat([df_train, df_predict])

# DATA CLEANING-----------------
# Danh sách các biến không cần thiết cần loại bỏ
unnecessary_columns = ['pct_tl_open_L6M', 'pct_tl_closed_L6M', 'pct_active_tl', 'pct_closed_tl','pct_tl_open_L12M',
                       'pct_tl_closed_L12M','pct_of_active_TLs_ever', 'pct_opened_TLs_L6m_of_L12m', 'pct_currentBal_all_TL',
                       'pct_PL_enq_L6m_of_L12m', 'pct_CC_enq_L6m_of_L12m', 'pct_PL_enq_L6m_of_ever', 'pct_CC_enq_L6m_of_ever']

# Loại bỏ các biến không cần thiết từ dataframe
df = df.drop(columns=unnecessary_columns)

# Thay -99999 thành Nan
df.replace(-99999, np.nan, inplace=True)

# Điền Nan là trung bình của cột đó
numeric_columns = df.select_dtypes(include=[np.number]).columns # Những cột có kiểu dữ liệu là int, float
non_numeric_columns = df.columns.difference(numeric_columns)# Những cột có kiểu dữ liệu là object

df_numeric = df[numeric_columns]
df_non_numeric = df[non_numeric_columns]

# Thực hiện tính toán trung bình chỉ trên các cột có thể tính trung bình
df = df_numeric.fillna(df_numeric.mean())

# Gắn lại các cột không thể tính trung bình vào DataFrame
df[non_numeric_columns] = df_non_numeric

# mapping 2 biến MARITALSTATUS và GENDER thành số
MARITALSTATUS_mapping = {'Married': 1, 'Single': 0}
df["MARITALSTATUS"] = df["MARITALSTATUS"].map(MARITALSTATUS_mapping)

GENDER_mapping = {'M': 1, 'F': 0}
df["GENDER"] = df["GENDER"].map(GENDER_mapping)

# one hot encoder 3  biến [ 'EDUCATION', 'last_prod_enq2' , 'first_prod_enq2' ]
df = pd.get_dummies(df, columns=[ 'EDUCATION', 'last_prod_enq2', 'first_prod_enq2'])

df.replace(True, 1, inplace=True)
df.replace(False, 0, inplace=True)

# mapping nhãn thành số
category_mapping = {'P1': 0, 'P2': 1, 'P3': 2, 'P4': 3}
df["Approved_Flag"] = df["Approved_Flag"].map(category_mapping)

# MODEL-----------------

x = df.drop("Approved_Flag",axis = 1)
y  = df["Approved_Flag"]

scaler = MinMaxScaler()
# Chuẩn hóa từng cột
for column in x.columns:
    x[column] = scaler.fit_transform(x[[column]])

x_train, x_test, y_train, y_test = train_test_split(x[:len(df_train)],y[:len(df_train)], test_size = 0.2, random_state = 15, stratify = y[:len(df_train)])

x_predict = x[len(df_train):]

@st.cache_data # Lưu dữ liệu khi được tải lên streamlit

# Hàm đánh giá
def EvaluateModel( y_test, y_pred):
    # Vẽ confusion matrix
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=[0, 1, 2,3])
    fig, ax = plt.subplots()
    display.plot(cmap='Blues', ax=ax)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    # st.pyplot(fig)

    # Tính những thông số chung
    # st.write('Classification Report: \n', classification_report(y_test, y_pred))
    
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(y_test)
    y_true_onehot = label_binarizer.transform(y_test)
    y_pred_onehot = label_binarizer.transform(y_pred)

    auc_score = roc_auc_score(y_true_onehot, y_pred_onehot, average='micro', multi_class="ovr")
    accuracy = accuracy_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred, average='micro')    

    st.write('Accuracy: %.3f' % accuracy)
    st.write('F1-Score: %.3f' % f1score)
    st.write('AUC: %.3f' % auc_score)
    metric = 0.3 * accuracy + 0.3 * f1score + 0.4*auc_score
    st.write('**Metric : %.3f** ' % metric)

# Tạo select box để chọn mô hình
# select_model = st.selectbox('Select Model', ['Logistic Regression', 'Neural Network', 'Random Forest', 'XGBoost', 'LGBM', 'Combined Model'])
options = st.multiselect(
    "Select Model",
    ["Logistic Regression", "Neural Network", "Random Forest", "XGBoost", "LGBM"],
    ["Logistic Regression"])

@st.cache_data
# Kết hợp 3 mô hình có chỉ số Metric cao nhất 
def CombinedModel(options):
    model_reg = LogisticRegression()
    model_RDF = RandomForestClassifier()
    model_lgbm = LGBMClassifier()
    model_xgb = XGBClassifier()
    model_neural = MLPClassifier()
    models = {
        "Logistic Regression": model_reg,
        "Neural Network": model_neural,
        "Random Forest": model_RDF,
        "XGBoost": model_xgb,
        "LGBM": model_lgbm
    }
    selected_models = [(option, models[option]) for option in options]
    model_combined = VotingClassifier(estimators=selected_models, voting='soft')
    model_combined.fit(x_train, y_train)
    y_pred_combined = model_combined.predict(x_test)
    return y_pred_combined
st.subheader(f'Kết quả các chỉ số đánh giá của mô hình: {options}')
st.write(EvaluateModel(y_test, CombinedModel(options)))
