##############################
# Telco Customer Churn Machine Learning
##############################

# Problem: Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir.
# Modeli geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.

# Telco müşteri churn verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan
# hayali bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu içermektedir.

# 21 Değişken 7043 Gözlem

# CustomerId: Müşteri İd’si
# Gender: Cinsiyet
# SeniorCitizen: Müşterinin yaşlı olup olmadığı (1, 0)
# Partner: Müşterinin bir ortağı olup olmadığı (Evet, Hayır) ? Evli olup olmama
# Dependents : Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır) (Çocuk, anne, baba, büyükanne)
# tenure: Müşterinin şirkette kaldığı ay sayısı
# PhoneService: Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
# MultipleLines: Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
# InternetService: Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
# OnlineSecurity: Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# OnlineBackup: Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# DeviceProtection: Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# TechSupport: Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingTV: Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok) Müşterinin, bir üçüncü taraf sağlayıcıdan televizyon programları yayınlamak için İnternet hizmetini kullanıp kullanmadığını gösterir
# StreamingMovies: Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok) Müşterinin bir üçüncü taraf sağlayıcıdan film akışı yapmak için İnternet hizmetini kullanıp kullanmadığını gösterir
# Contract: Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
# PaperlessBilling: Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
# PaymentMethod: Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
# MonthlyCharges: Müşteriden aylık olarak tahsil edilen tutar
# TotalCharges: Müşteriden tahsil edilen toplam tutar
# Churn: Müşterinin kullanıp kullanmadığı (Evet veya Hayır) - Geçen ay veya çeyreklik içerisinde ayrılan müşteriler


# Her satır benzersiz bir müşteriyi temsil etmekte.
# Değişkenler müşteri hizmetleri, hesap ve demografik veriler hakkında bilgiler içerir.
# Müşterilerin kaydolduğu hizmetler - phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
# Müşteri hesap bilgileri – ne kadar süredir müşteri oldukları, sözleşme, ödeme yöntemi, kağıtsız faturalandırma, aylık ücretler ve toplam ücretler
# Müşteriler hakkında demografik bilgiler - cinsiyet, yaş aralığı ve ortakları ve bakmakla yükümlü oldukları kişiler olup olmadığı

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("Machine_Learning/Telco-Customer-Churn.csv")

##################################
# GÖREV 1: KEŞİFCİ VERİ ANALİZİ
##################################

##################################
# GENEL RESİM
##################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Info #####################")
    print(dataframe.info())
    print("##################### Columns #####################")
    print(dataframe.columns)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### Missing Values #####################")
    print(dataframe.isnull().sum())
    print("##################### Duplicate #####################")
    print(dataframe.duplicated().sum())
    print("##################### Count of Unique Values #####################")
    print(dataframe.nunique())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# ADIM 1: Numerik ve kategorik değişkenleri yakalayınız.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# ADIM 2: Gerekli düzenlemeleri yapınız.
# Hedef değişkenimiz olan churn değişkenini binary olarak düzenliyoruz.
df["Churn"].value_counts()
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)


# ADIM 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
# Kategorik değişkenlerin analizi
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col)

# Kategorik değişken analizinin çıkarımları:
# 1. Kadın ve erkek müşterilerin oranları birbirlerine çok yakın. Herhangi bir anlamlı fark yoktur.
# 2. Az da olsa partneri olan veya olmayan arasında küçük bir fark vardır. Daha çok partneri olmayan müşteri vardır.
# 3. Müşterilerin %70 inin bakmakla yükümlü olduğu kişi yoktur. Önemli bir fark mevcuttur.
# 4. Müşterilerin yaklaşık %10 unun telefon servisi yoktur.
# 5. Müşterilerin %50 ye yakınının sadece bir tane hattı bulunmaktadır. Müşterilerin %42 oranında birden fazla hattı vardır.
# 6. Müşterilerin çoğunluğunun (%43) fiber optik internet servisi vardır. Müşterilerin yaklaşık 1/3'ünün internet bağlantısı
# DSL dir. Hiç internet servisi kullanmayanların oranı ise %20 dir.
# 7. Müşterilerin neredeyse yarısının online (çevrim içi) güvenliği yoktur.
# 8. Müşterilerin çoğunluğu cihaz koruması kullanmıyor.
# 9. Müşterileri yaklaşık yarısı teknik destek almamıştır.
# 11. Müşterilerin %50 sinden fazlası aydan aya sözleşme yapmaktadır. Bir ve iki yılloık sözleşmeler birbirine yakındır.
# 12. Müşterilerin çoğunluğu kağıtsız faturaya sahiptir. Bu da fatura ödemelerinin daha çok online tarafta olduğunu göstermektedir.
# 13. Müşterinin en çok tercih ettiği ödeme yöntemi elektronik çektir.
# 14. Müşterilerin büyük çoğunluğu yaşlıdır. Yaşlı olmayan müşterilerin oranı %16 dır.

# Numerik değişkenlerin analizi
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)

# Numeric değişkenlere ilişkin çıkarımlar:
# 1. Müşterilerin şirkette kaldığı ay sayısına baktığımızda, dağılımın en çok olduğu aylar 1-3 ve 70 li aylardır.
# 2. Aylık olarak yapılan sözleşmelerden dolayı ilk 3 ayı kapsayan dönem yoğunluk kazanmaktadır.
# 3. 2 yıllık sözleşmeler ise 70 li aylara gelindiğinde pik yapmaktadır.
df[df["Contract"] == "Month-to-month"]["tenure"].hist(bins=20)
plt.xlabel("tenure")
plt.title("Month-to-month")
plt.show(block=True)

df[df["Contract"] == "Two year"]["tenure"].hist(bins=20)
plt.xlabel("tenure")
plt.title("Two year")
plt.show(block=True)

# 4. Aylık ödemelerin çoğunluğu 20 dolardır. Çünkü aylık sözleşmelerin ve iki yıllık sözleşmelerin büyük çoğunluğu bu
# dönemde gerçekleşmiştir.
df[df["Contract"] == "Month-to-month"]["MonthlyCharges"].hist(bins=20)
plt.xlabel("MonthlyCharges")
plt.title("Month-to-month")
plt.show(block=True)

df[df["Contract"] == "Two year"]["MonthlyCharges"].hist(bins=20)
plt.xlabel("MonthlyCharges")
plt.title("Two year")
plt.show(block=True)


# ADIM 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

# Kategorik değişkenlerin hedef değişken ile ilgili çıkarımları
# 1. Yaşlı müşterilerin churn oranı (%16) düşüktür.
# 2. Taklaşık olarak müşterilerin %25 i churn olmuş durumdadır.
# 3. Aylık sözleşme yapan müşterileri churn olma oranı daha yüksektir.
# 4. Teknik destek alanların churn olma oranı, teknik destek almayanların oranlarına göre önemli derecede daha düşüktür.
# 5. Cihaz korumasına sahip olmayan müşterilerin terk etme oranı daha yüksektir.
# 6. Online yedeği olmayan müşterilerin churn oranı daha yüksektir.
# 7. Online güvenliği bulunan müşterilerin çhurn oranı olmayanlara göre önemli derecede düşüktür.
# 8. İnternet servisi olmayan müşterilerin churn oranı intenet hizmetine sahip olanlara göre düşüktür.
# 9. Bakmakla yükümlü olunan kişilere sahip olan müşterilerin churn oranı olmayanlara göre oldukça düşüktür.


# ADIM 5: Aykırı gözlem var mı inceleyiniz.
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)


# ADIM 6: Eksik gözlem var mı inceleyiniz.
df.isnull().sum()
df.isna().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)

# Numerik ama kategorik olarak gösterlen bazı değişkenleri numerik değişkene çeviriyoruz. Bu işlemler feature engineering'te
# feature çıkarımı yapabilmemiz için zorunlu işlemlerdir.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')

df.columns[:20]
##################################
# GÖREV 2: FEATURE ENGINEERING (ÖZELLİK ÇIKARIMI)
##################################

# ADIM 1: Yeni değişkenler oluşturunuz.
# Tenure değişkeninden yıllık kategorik değişken oluşturma
df.loc[(df["tenure"] >= 0) & (df["tenure"] <= 12), "NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24), "NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36), "NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"] > 36) & (df["tenure"] <= 48), "NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"] > 48) & (df["tenure"] <= 60), "NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"] > 60) & (df["tenure"] <= 72), "NEW_TENURE_YEAR"] = "5-6 Year"

df.columns
# Kontratı 1 veya 2 yıllık müşterileri Engaged olarak belirtme
df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year","Two year"] else 0)

# Herhangi bir teknik destek, online yedek veya cihaz koruması almayan müşteriler
df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)

# Aylık sözleşmesi bulunan ve genç olan müşteriler
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)

# Müşterinin toplam aldığı servis sayısı
df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']] == 'Yes').sum(axis=1)

# Herhangi bir streaming hizmeti alan müşteriler
df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

# Müşterinin otomatik ödeme yaptığının tespit edilmesi
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)", "Credit card (automatic)"] else 0)

# ortalama aylık ödeme
df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)

# Güncel Fiyatın ortalama fiyata göre artışı
df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]

# Servis başına ücret
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1)


# ADIM 2: Encoding işlemlerini gerçekleştiriniz.
# Değişkenlerin tiplerine göre ayrılması işlemi
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# LABEL ENCODING
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)

# ONE-HOT ENCODING İşlemi
# cat_cols listesinin güncelleme işlemi
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn", "NEW_TotalServices"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()

# Yeni değikenlerle beraber encoding işlemi yaptıktan sonra NaN değer var mı diye kontrol ediyoruz.
# Modelleme işlemi gerçekleştirilirken NaN değerlerin olmaması gerektiği için eksik değerleri siliyoruz.
df.isnull().sum()
df.dropna(inplace=True)

# ADIM 3: Numerik değişkenler için standartlaştırma yapınız.
scaler = RobustScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

##################################
# GÖREV 3: MODELLEME
##################################

# ADIM 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyin. En iyi 4 modeli seçiniz.
y = df["Churn"]
X = df.drop(["Churn", "customerID"], axis=1)

models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('SVM', SVC(gamma='auto', random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

# SONUÇLAR

#### 1) Logistic Regression
### 2) SVC
## 3) CatBoost
# 4) LightGBM

# ADIM 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin ve bulduğunuz hiparparametreler ile modeli
# tekrar kurunuz.

################################################
# RANDOM FOREST
################################################

# Model kurulumu
rf_model = RandomForestClassifier(random_state=17)

# Geliştirebileceğimiz veya ekleyebileceğimiz parametreleri kontrol ediyoruz.
rf_model.get_params()

# Parametrelerimiz şunlardır:
# max_depth: Karar ağacındaki maksimum derinliği veya ağaçtaki katman sayısı
# max_features: Her karar ağacı bölünmesinde kullanılabilecek maksimum özellik sayısı
# min_samples_split: İç düğümü bölmek için gereken minimum örnek sayısı
# n_estimators: Bir rastgele orman algoritmasındaki ağaç sayısını
rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "sqrt"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# En iyi parametre değerlerini alıyoruz.
rf_best_grid.best_params_

# En iyi skoru alıyoruz.
rf_best_grid.best_score_

# Oluşturulan hiperparametre optimizasyonu sonucunda ortaya çıkan en optimal değerler ile final modelini oluşturuyoruz.
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

# Cross Validation yaptıktan sonra değerlerimizi inceliyoruz.
cv_results = cross_validate(rf_final,
                            X, y,
                            cv=10,
                            scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.8001949841588001
cv_results['test_f1'].mean()
# 0.5574485042819524
cv_results['test_roc_auc'].mean()
# 0.8447386039005019

################################################
# GBM (Gradient Boosting Machines)
################################################

# Model kurulumu
gbm_model = GradientBoostingClassifier(random_state=17)

# Parametre kontrolü
gbm_model.get_params()

# Hiperparametre optimizasyonu öncesindeki hatalarımızı tespit ediyoruz
cv_results = cross_validate(gbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.8003390889486726
cv_results['test_f1'].mean()
# 0.579870633641461
cv_results['test_roc_auc'].mean()
# 0.8439231390444697

# Parametre optimizasyonunu kuruyoruz
gbm_params = {"learning_rate": [0.01, 0.1],     # Learning rate'in düşük olması daha iyi tahmin yapmamızı sağlayacaktır.
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}       # Subsample göz önünde bulundurulacak gözlem oranını belirliyor.

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# En iyi parametre değerlerini alıyoruz.
gbm_best_grid.best_params_

# Aldığımız parametre modelleriyle birlikte final modeli oluşturuyoruz.
gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)


cv_results = cross_validate(gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.0.8031832303631203
cv_results['test_f1'].mean()
# 0.5863115458260272
cv_results['test_roc_auc'].mean()
# 0.8461356329359001

################################################
# XGBoost
################################################

# Model kurulumu
xgboost_model = XGBClassifier(random_state=17)

# Parametrelere bakıyoruz.
xgboost_model.get_params()

# Parametrelerimiz şunlardır:
# max_depth: Karar ağacındaki maksimum derinliği veya ağaçtaki katman sayısı
# n_estimators: Bir rastgele orman algoritmasındaki ağaç sayısını
# learning_rate: Bir artırma modeli eğitilirken her iterasyonda adım büyüklüğünü kontrol eder.
# colsample_bytree: Gradyan arttırma modelinde her bir ağaç için rastgele örneklenmesi gereken sütunların oranını temsil eder.
xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.7, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# En iyi parametre değerlerini alıyoruz.
xgboost_best_grid.best_params_

# Oluşturulan hiperparametre optimizasyonu sonucunda ortaya çıkan en optimal değerler ile final modelini oluşturuyoruz.
xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final,
                            X, y,
                            cv=10,
                            scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.8023311214923059
cv_results['test_f1'].mean()
# 0.5886574834446977
cv_results['test_roc_auc'].mean()
# 0.844366439397373

################################################
# LightGBM
################################################

# Model kurulumu
lgbm_model = LGBMClassifier(random_state=17)

# Geliştirebileceğimiz veya ekleyebileceğimiz parametreleri kontrol ediyoruz.
lgbm_model.get_params()

# Parametrelerimiz şunlardır:
# learning_rate: Bir artırma modeli eğitilirken her iterasyonda adım büyüklüğünü kontrol eder.
# n_estimators: Bir rastgele orman algoritmasındaki ağaç sayısını
# colsample_bytree: Gradyan arttırma modelinde her bir ağaç için rastgele örneklenmesi gereken sütunların oranını temsil eder.
lgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# En iyi parametre değerlerini alıyoruz.
lgbm_best_grid.best_params_

# Oluşturulan hiperparametre optimizasyonu sonucunda ortaya çıkan en optimal değerler ile final modelini oluşturuyoruz.
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final,
                            X, y,
                            cv=10,
                            scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.8033252376180007
cv_results['test_f1'].mean()
# 0.5886664743915179
cv_results['test_roc_auc'].mean()
# 0.8452279239233882

################################################
# CatBoost
################################################

# Model kurulumu
catboost_model = CatBoostClassifier(random_state=17, verbose=False)

# Geliştirebileceğimiz veya ekleyebileceğimiz parametreleri kontrol ediyoruz.
catboost_model.get_params()

# Parametrelerimiz şunlardır:
# iterations: eğitim sırasında gerçekleştirilen iterasyon sayısı
# learning_rate: Bir artırma modeli eğitilirken her iterasyonda adım büyüklüğünü kontrol eder.
# "depth" parametresi karar ağacı algoritmasının izin verdiği maksimum derinliği veya seviyeleri ifade eder.
catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# Oluşturulan hiperparametre optimizasyonu sonucunda ortaya çıkan en optimal değerler ile final modelini oluşturuyoruz.
catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.8037544048234837
cv_results['test_f1'].mean()
# 0.5845444699021235
cv_results['test_roc_auc'].mean()
# 0.8454673211029202

##################################
# KNN ALGORITHM
##################################

#  Modeli kuruyoruz.
knn_model = KNeighborsClassifier().fit(X, y)

# Cross Validation ile başarıyı değerlendirme
cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7714719432708435
cv_results['test_f1'].mean()
# 0.551744937939627
cv_results['test_roc_auc'].mean()
# 0.7842717044203219

# Hiperparametre Optimizasyonu
knn_model = KNeighborsClassifier()

knn_params = {"n_neighbors": range(2, 50)}

knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,                 # Hiperparametre optimizasyonu için cross validation kullanıyoruz.
                           n_jobs=-1,
                           verbose=1).fit(X, y)

# Bize en optimum komşuluk sayısını verecek
knn_gs_best.best_params_

# Oluşturulan hiperparametre optimizasyonu sonucunda ortaya çıkan en optimal değerler ile final modelini oluşturuyoruz.
knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(knn_final,
                            X, y,
                            cv=10,
                            scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.7942263888846763
cv_results['test_f1'].mean()
# 0.5766754412690828
cv_results['test_roc_auc'].mean()
# 0.8373895800024801

################################################
# SONUÇLAR
################################################

###### 1) LigthGBM
##### 2) GBM
#### 3) CatBoost
### 4) XGBBoost
## 5) Random Forest
# 6) KNN Algorithm

