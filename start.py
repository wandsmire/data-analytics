import streamlit as st
import Orange
import pickle
import pandas as pd
import numpy as np




#########################  CONTAINERS  ##############################


# Sidebar
sd_bar = st.sidebar

# Container for the Header
header_cont = st.container()

# Container for the Dataset
dataset_cont = st.container()

# Container for the Features
features_cont = st.container()

# Container for Model Prediction
modelPrediction_cont = st.container()



######################  Load the Stack model  ########################


# StackModel contains Pre-processing pipeline (made on the training dataset)
# + Model (with its parameters & hyperparameters)
with open("model.pkcls","rb") as model:
    loaded_model = pickle.load(model)



########################  CACHE FUNCTION  #############################


#@st.cache
def get_data():
    df = pd.read_csv("datax.csv")
    return df



########################## SIDEBAR ###################################


with sd_bar:
    st.markdown("## Prediksi Potensi BKPN")



###################  Extract input from the user  #####################


def get_user_input():

    df = get_data()
    #st.markdown(list(df.columns))

    data_pp = np.array(df["PENYERAH_PIUTANG"])
    pp_sorted = np.unique(data_pp)    
    pp_val = sd_bar.selectbox(label = "PENYERAH_PIUTANG", options = pp_sorted, index = 0)

    data_lokasi = np.array(df["LOKASI"])
    lokasi_sorted = np.unique(data_lokasi)    
    lokasi_val = sd_bar.selectbox(label = "LOKASI", options = lokasi_sorted, index = 0)

    data_waktu = np.array(df["WAKTU"])
    waktu_sorted = np.unique(data_waktu)    
    waktu_val = sd_bar.selectbox(label = "WAKTU", options = waktu_sorted, index = 0)

    data_utang = np.array(df["UTANG"])
    utang_sorted = np.unique(data_utang)    
    utang_val = sd_bar.selectbox(label = "UTANG", options = utang_sorted, index = 0)

    # define Orange domain
    pp = Orange.data.DiscreteVariable("PENYERAH_PIUTANG",[pp_val])
    lokasi = Orange.data.DiscreteVariable("LOKASI",[lokasi_val])
    waktu = Orange.data.DiscreteVariable("WAKTU",[waktu_val])
    utang = Orange.data.DiscreteVariable("UTANG",[utang_val])


    domain = Orange.data.Domain([pp,lokasi,waktu,utang]) 

    # input values X
    X = np.array([[0,0,0,0]])

    # in this format, the data is now ready to be fed to StackModel
    user_input = Orange.data.Table(domain, X)

    #return user_input


df_userinput = get_user_input()



###########################  HEADER  ##################################

    
with header_cont:
    st.markdown("# Prediksi Potensi Berkas Kasus Piutang Negara pada Kanwil DJKN Papua, Papua Barat, dan Maluku")
    st.markdown("Prediksi ini merupakan implementasi penggunaan machine learning secara sederhana.")



###########################   DATASET  ##################################

    
with dataset_cont:
    st.markdown("## Dataset")
    st.markdown("The ML model was trained on the dataset below (adopted from [Applied Predictive Modeling](https://link.springer.com/book/10.1007/978-1-4614-6849-3)). "
             "The last column (churn) is the target showing whether the customer unsubscribed from the service (yes) or not (no).")
    df = get_data()
    #df = df.drop(df.columns[[0]], axis=1, inplace=True)
    st.dataframe(df)

#########   FEATURES  ##################################

    
with features_cont:
    st.markdown("## Features")
    st.markdown("However, after training the ML model only the following 10 features were identified as "
                "relevant for making predictions:")
    st.markdown("(1) **state** - the US home state of the customer")
    st.markdown("(2) **international_plan** - indicates whether the customer has international plan or not")
    st.markdown("(3) **voice_mail_plan** - indicates whether the customer has voice mail plan or not")
    st.markdown("(4) **number_vmail_messages** - number of voice mail messages made by the customer")


#############################   MODEL PREDICTION   #########################

    
with modelPrediction_cont:
    st.markdown("## Model Prediction")
    st.markdown("The ML model is a Stack consisting of Gradient Boosting and Random Forest "
            "classifiers with a Logistic Regression acting as an aggregate. To make a prediction "
            "with the model, you need to select values for the 10 features by using the "
            "options and sliders in the sidebar on the left.")

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("### Input")
        st.write("1:  ", "AKO")
        st.write("1:  ", df_userinput[0,0])
        st.write("2:  ", df_userinput[0,1])
        st.write("3:  ", df_userinput[0,2])
        st.write("4:  ", df_userinput[0,3])
    
    probs = loaded_model(df_userinput[0], 1)
    prob_no = probs[0]
    prob_yes = probs[1]
        
    with right_col:
    
        st.markdown("### Prediction Probabilities")
        st.markdown("Given the data on the left, the "
                "probability this customer will not churn, i.e. churn, is:")
        st.write("churn (no): ", round(prob_no*100, 1), "%")
        st.write("churn (yes): ", round(prob_yes*100,1), "%")

        st.markdown("### Prediction")
        if prob_no>prob_yes:
            st.markdown("This customer will **_not_** churn.")
        else:
            st.markdown("This customer **_will_** churn.")
        
