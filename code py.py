# =====================================================
# SISTEM PREDIKSI HARGA JUAL DAN KUALITAS ROTAN SIGI
# Menggunakan Regresi Linear dan Support Vector Machine
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix
)

# =====================================================
# FUNGSI MEMBACA DATA
# =====================================================

def baca_data():

    try:

        file = r"C:\Users\LENOVO\Downloads\Laporan_Rotan_Sigi (2).xlsx"

        if not os.path.exists(file):
            print("File tidak ditemukan!")
            print("Lokasi yang dicari:")
            print(file)
            return None

        df = pd.read_excel(
            file,
            sheet_name="Dataset Rotan Sigi",
            header=2
        )

        print("\nData berhasil dibaca!")
        print("Jumlah Data :", len(df))
        print("Jumlah Kolom :", len(df.columns))

        return df

    except Exception as e:

        print("\nTerjadi Error:")
        print(e)

        return None


# =====================================================
# MENAMPILKAN DATA
# =====================================================

def tampilkan_data(df):

    print("\n===== DATASET ROTAN =====")
    print(df.head(10))

    print("\nJumlah Data :", len(df))
    print("Jumlah Kolom :", len(df.columns))

    print("\nNama Kolom:")
    print(df.columns.tolist())


# =====================================================
# STATISTIK DATA
# =====================================================

def statistik_data(df):

    print("\n===== STATISTIK DATA =====")

    print(df.describe(include="all"))


# =====================================================
# PREPROCESSING
# =====================================================

def preprocessing(df):

    df = df.copy()

    if "No" in df.columns:
        df.drop("No", axis=1, inplace=True)

    encoder_jenis = LabelEncoder()
    encoder_musim = LabelEncoder()
    encoder_grade = LabelEncoder()

    df["Jenis Rotan"] = encoder_jenis.fit_transform(
        df["Jenis Rotan"]
    )

    df["Musim Panen"] = encoder_musim.fit_transform(
        df["Musim Panen"]
    )

    kolom_grade = [c for c in df.columns if "Grade" in c][0]

    df[kolom_grade] = encoder_grade.fit_transform(
        df[kolom_grade]
    )

    return (
        df,
        encoder_jenis,
        encoder_musim,
        encoder_grade
    )


# =====================================================
# REGRESI LINEAR
# =====================================================

def prediksi_harga(df):

    print("\n===== REGRESI LINEAR =====")

    kolom_harga = [c for c in df.columns if "Harga Jual" in c][0]
    kolom_grade = [c for c in df.columns if "Grade" in c][0]

    X = df.drop(
        [kolom_harga, kolom_grade],
        axis=1
    )

    y = df[kolom_harga]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    hasil = model.predict(X_test)

    mae = mean_absolute_error(y_test, hasil)
    mse = mean_squared_error(y_test, hasil)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, hasil)

    print("MAE  :", round(mae, 2))
    print("RMSE :", round(rmse, 2))
    print("R2   :", round(r2, 4))

    plt.figure(figsize=(8,5))
    plt.plot(y_test.values, label="Aktual")
    plt.plot(hasil, label="Prediksi")
    plt.legend()
    plt.title("Perbandingan Harga Aktual dan Prediksi")
    plt.show()

    return model


# =====================================================
# SVM
# =====================================================

def klasifikasi_grade(df):

    print("\n===== SUPPORT VECTOR MACHINE =====")

    kolom_harga = [c for c in df.columns if "Harga Jual" in c][0]
    kolom_grade = [c for c in df.columns if "Grade" in c][0]

    X = df.drop(
        [kolom_grade, kolom_harga],
        axis=1
    )

    y = df[kolom_grade]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    svm = SVC(
        kernel="rbf",
        C=1
    )

    svm.fit(X_train, y_train)

    hasil = svm.predict(X_test)

    akurasi = accuracy_score(
        y_test,
        hasil
    )

    print("Akurasi :", round(akurasi * 100, 2), "%")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, hasil))

    return svm


# =====================================================
# PREDIKSI DATA BARU
# =====================================================

def data_baru(
        model_reg,
        model_svm,
        encoder_jenis,
        encoder_musim,
        encoder_grade
):

    print("\n===== INPUT DATA BARU =====")

    jenis = input("Jenis Rotan : ")
    panjang = float(input("Panjang Batang (m): "))
    diameter = float(input("Diameter Batang (mm): "))
    kerapatan = float(input("Kerapatan (g/cm3): "))
    kadar_air = float(input("Kadar Air (%): "))
    musim = input("Musim Panen : ")
    pengeringan = int(input("Lama Pengeringan (hari): "))
    cacat = float(input("Cacat Fisik (%): "))

    data = pd.DataFrame({

        "Jenis Rotan": [
            encoder_jenis.transform([jenis])[0]
        ],

        "Panjang Batang\n(m)": [panjang],

        "Diameter Batang\n(mm)": [diameter],

        "Kerapatan\n(g/cm³)": [kerapatan],

        "Kadar Air\n(%)": [kadar_air],

        "Musim Panen": [
            encoder_musim.transform([musim])[0]
        ],

        "Lama Pengeringan\n(hari)": [pengeringan],

        "Cacat Fisik\n(%)": [cacat]

    })

    harga = model_reg.predict(data)[0]

    grade = model_svm.predict(data)[0]

    grade_asli = encoder_grade.inverse_transform(
        [grade]
    )[0]

    print("\n===== HASIL PREDIKSI =====")
    print("Harga Jual Prediksi : Rp", round(harga))
    print("Grade Kualitas      :", grade_asli)


# =====================================================
# MENU
# =====================================================

def menu():

    df = baca_data()

    if df is None:
        return

    tampilkan_data(df)

    statistik_data(df)

    (
        df,
        encoder_jenis,
        encoder_musim,
        encoder_grade
    ) = preprocessing(df)

    model_reg = prediksi_harga(df)

    model_svm = klasifikasi_grade(df)

    while True:

        print("\n==========================")
        print("MENU PROGRAM")
        print("==========================")
        print("1. Prediksi Data Baru")
        print("2. Keluar")

        pilih = input("Pilihan : ")

        if pilih == "1":

            data_baru(
                model_reg,
                model_svm,
                encoder_jenis,
                encoder_musim,
                encoder_grade
            )

        elif pilih == "2":

            print("Program selesai.")
            break

        else:

            print("Pilihan tidak tersedia.")


# =====================================================
# PROGRAM UTAMA
# =====================================================

if __name__ == "__main__":
    menu()