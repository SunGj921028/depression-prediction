### 實作資料集
- 資料集屬於 Kaggle Playground series - Season 4 的 Epiosode 11
- 是一份**合成資料集**（synthetically-generated datasets）
  - 用以平衡現實性或提供一些更有趣的數據集

---

### 資料集敘述與說明：
- 透過一個深度學習模型基於 Depression Survey/Dataset for Analysis 而生成的數據集，但保留了與原始資料相似的分佈特性
- 資料集敘述中有提到有刻意的留下一些「**數據異常**」或「**特徵干擾**」
- 資料集包含兩份檔案
  - train.csv
    - 訓練數據集，包含多個特徵（數值型、類別型）以及目標欄位（Depression）
    - 總共有（0 - 140699）筆
  - test.csv
    - 測試數據集，參賽者需利用模型**預測其目標欄位**

---

### 資料集目的：
- 基於心理健康調查數據，探索影響個體是否可能經歷憂鬱症的**因素**，並建立**分類模型**來**預測憂鬱症的可能性**

---

#### 資料集來源：
- [Exploring Mental Health Data](https://www.kaggle.com/competitions/playground-series-s4e11/data)

---

### 採用的 Machine Learning Method：
#### 資料前處理：
- 將資料的 **`Depression`** 欄位與原始資料分成兩份資料
- 將以下兩個欄位進行 **mapping**，以防止 One-hot vector 轉換成**全 0** 的數值
  - **Sleep Duration**
    ```py
    sleep_mapping = {
        "More than 8 hours":9,
        'Less than 5 hours':4,
        '5-6 hours':5.5,
        '7-8 hours':7.5,
        '1-2 hours':1.5,
        '6-8 hours':7,
        '4-6 hours':5,
        '6-7 hours':6.5,
        '10-11 hours':10.5,
        '8-9 hours':8.5,
        '9-11 hours':10,
        '2-3 hours':2.5,
        '3-4 hours':3.5,
        'Moderate':6,
        '4-5 hours':4.5,
        '9-6 hours':7.5,
        '1-3 hours':2,
        '1-6 hours':4,
        '8 hours':8,
        '10-6 hours':8,
        'Unhealthy': 3,
        'Work_Study_Hours':6,
        '3-6 hours':3.5,
        '9-5':7,
        '9-5 hours': 7,
        '49 hours': 7,
        '35-36 hours': 5,
        '40-45 hours': 6,
        'Pune': 6,
        'Indore': 6,
        '45': 6,
        '55-66 hours': 8.5,
        'than 5 hours': 5,
        '45-48 hours': 6.5,
        'No': 6,
        'Sleep_Duration': 6,
    }
    ```
  - **Dietary Habits**
    ```py
    dietary_mapping = {
        'Unhealthy': 0,
        'Less Healthy': 0,
        'No Healthy': 0,
        'Less than Healthy': 0,
        'Moderate': 1,
        'Healthy': 2,
        'More Healthy': 2,
        'Yes': 2, 
        'No': 0,
    }
    ```
- Drop 掉以下的欄位
  - **`id`**
  - **`Name`**
  - **`City`**
  - **`Profession`**
  - **`Degree`**
- 因為有觀察到 **missing value**，所以進行缺失值填補
  - **`SimpleImputer()`**
    - **類別型**的資料透過**眾數（most_frequent）填補**，**數值型**的資料透過**平均數（median）填補**
- 特徵處理：
  - 對類別特徵進行 One-Hot Encoder，對數值特徵進行 **`StandardScaler()`** 處理

---

### 模型的訓練：
- 使用 **support vector machine（SVM**）進行分類任務
  - 呼叫 **`SVC（）`**，並設定參數
- 使用 **`train_test_split()`** 將資料以（8:2）分成訓練、測試集
- 採取 5-fold （**`KFold(5)`**） 的方式進行交叉驗證，以 accuracy 為準
- 訓練過程中透過 **`compute_class_weight()`** 計算 Depression 兩種類別的 weight
- 透過 **`GridSearchCV()`**，尋找最佳參數組合

---

### 執行環境與設定：
- 使用語言及版本：**python 3.12.6**
- 使用的 python module 如下（需要預先裝好）
  - `matplotlib.pyplot`
  - `pandas`
  - `numpy`
  - `sklearn`
    - `model_selection`：模型設定相關
    - `impute`：填補缺失值
    - `preprocessing`：轉換型別與標準化
    - `sklearn.utils.class_weight`：進行訓練資料類別平衡
    - `metrics`：評估相關
    - `class_weight`：計算類別的 weight
    - `SVM`：SVC（）的模型
    - `impute`：數據填補
    - `compose, pipeline`

---

### 執行方式：
1. 將所需的 kaggle 資料集放入名為 **data** 的資料夾中
2. 執行 python 指令
    ```python
    python main.py
    ```
3. 觀察評估結果（accuracy, recall, precision, f1-score, confusion matrix）以及實際的三種 submission file

---

### Result：
#### 三種參數的比較效果

#### 評估指標比較圖

#### Kaggle Leaderboard











