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



######################  Load the model  ########################

with open("model.pkcls","rb") as model:
    loaded_model = pickle.load(model)

########################  Load the data  #############################

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

################update this for variable changes.

    #untuk bagian DF harus sesuai dengan nama kolom pada CSV!	
    data_pp = np.array(df["PENYERAH_PIUTANG"])
    pp_sorted = np.unique(data_pp)    
    pp_val = sd_bar.selectbox(label = "Penyerah Piutang", options = pp_sorted, index = 0)

    data_lokasi = np.array(df["LOKASI"])
    lokasi_sorted = np.unique(data_lokasi)    
    lokasi_val = sd_bar.selectbox(label = "Lokasi Debitur", options = lokasi_sorted, index = 0)

    data_waktu = np.array(df["UMUR_BKPN"])
    waktu_sorted = np.unique(data_waktu)    
    waktu_val = sd_bar.selectbox(label = "Umur BKPN", options = waktu_sorted, index = 0)

    data_utang = np.array(df["NILAI_SP3N"])
    utang_sorted = np.unique(data_utang)    
    utang_val = sd_bar.selectbox(label = "Nilai SP3N", options = utang_sorted, index = 0)

    data_jamin = np.array(df["ADA_JAMINAN"])
    jamin_sorted = np.unique(data_jamin)    
    jamin_val = sd_bar.selectbox(label = "Ada Jaminan?", options = jamin_sorted, index = 0)

    # define Orange domain, untuk nama variabel harus sesuai dengan di orange.
    pp = Orange.data.DiscreteVariable("PENYERAH_PIUTANG",[pp_val])
    lokasi = Orange.data.DiscreteVariable("LOKASI",[lokasi_val])
    waktu = Orange.data.DiscreteVariable("UMUR_BKPN",[waktu_val])
    utang = Orange.data.DiscreteVariable("NILAI_SP3N",[utang_val])
    jamin = Orange.data.DiscreteVariable("ADA_JAMINAN",[jamin_val])

    domain = Orange.data.Domain([pp,lokasi,waktu,utang,jamin]) 

    loe = [[pp_val, lokasi_val, waktu_val, utang_val,jamin_val]]

    # in this format, the data is now ready to be fed to Model
    user_input = Orange.data.Table(domain, loe)
    return user_input;

df_userinput = get_user_input()

###########################  HEADER  ##################################

    
with header_cont:
    st.markdown("# Prediksi Potensi BKPN pada Kanwil DJKN Papua, Papua Barat, dan Maluku")
    st.markdown("Prediksi ini merupakan implementasi penggunaan machine learning secara sederhana. "
                "Tujuannya untuk menentukan apakah BKPN dikategorikan sebagai BKPN Potensial atau Tidak Potensial. "
                "BKPN Potensial artinya BKPN yang berdasarkan data historis dengan karakteristik serupa, berpotensi pelunasan. "
		    "BKPN Tidak potensial yaitu yang berdasarkan data historis, cenderung berakhir PSBDT. "
                "Silahkan gunakan kolom input di sebelah kiri untuk melakukan prediksi." 
               )



###########################   DATASET  ##################################

    
with dataset_cont:
    st.markdown("## Dataset")
    st.markdown("Model machine learning ini dilatih menggunakan data sebagai berikut.")
    df = get_data()
    #df.drop(df.columns[[5]], axis=1, inplace=True)
    st.dataframe(df)

#########   FEATURES  ##################################

################update this for display only.
with features_cont:
    st.markdown("## Variabel")
    st.markdown("Lima variabel ini digunakan untuk melakukan prediksi BKPN potensial:")
    st.markdown("(1) **Penyerah Piutang** - Kementerian/Lembaga penyerah piutang,")
    st.markdown("(2) **Lokasi Debitur** - Lokasi debitur apakah di dalam atau di luar kota KPKNL,")
    st.markdown("(3) **Umur BKPN** - Berapa lama BKPN diurus oleh KPKNL,")
    st.markdown("(4) **Nilai SP3N** - Besaran nilai utang masing-masing debitur saat diterbitkan SP3N.")
    st.markdown("(5) **Keberadaan Barang Jaminan** - Apakah ada barang jaminan.")


#############################   MODEL PREDICTION   #########################

    
with modelPrediction_cont:
    st.markdown("## Model yang digunakan")
    st.markdown("Model yang digunakan adalah Naive Bayes. "
                "Model ini dipilih setelah dilakukan validasi silang dengan model Random Forest dan kNN. "
                "Dengan dataset ini, Model Naive Bayes mempunyai keakuratan tertinggi.")

    left_col, right_col = st.columns(2)

################update this for display only.

    with left_col:
        st.markdown("### Input")
        st.write("Penyerah Piutang:  ", df_userinput[0,0])
        st.write("Lokasi Debitur:  ", df_userinput[0,1])
        st.write("Umur BKPN:  ", df_userinput[0,2])
        st.write("Jumlah Utang:  ", df_userinput[0,3])
        st.write("Ada Jaminan:  ", df_userinput[0,4])
    
    probs = loaded_model(df_userinput[0], 1)
    prob_no = probs[0]
    prob_yes = probs[1]
        
    with right_col:
    
        st.markdown("### Probabilitas Prediksi")
        st.markdown("Berdasarkan data di sebelah kiri, maka "
                "kemmungkinan BKPN ini potensial adalah:")
        st.write("potensial: ", round(prob_no*100, 1), "%")
        st.write("tidak potensial: ", round(prob_yes*100,1), "%")

        st.markdown("### Prediksi")
        if prob_no>prob_yes:
            st.markdown("BKPN ini **_potensial_**.")
        else:
            st.markdown("BKPN ini **_tidak potensial_**.")
        
