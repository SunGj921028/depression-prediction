import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from preprocessing import preprocess_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight


def main():
    print("- data/train.csv:", os.path.exists("data/train.csv"))
    print("- data/test.csv:", os.path.exists("data/test.csv"))
    print("- preprocessing.py:", os.path.exists("preprocessing.py"))
    # Load the data
    print("Loading data...")
    try:
        train_data = pd.read_csv("data/train.csv")
        print(f"Successfully loaded train data with shape: {train_data.shape}")
    except FileNotFoundError:
        print("Error: Could not find data/train.csv")
        print("Current working directory:", os.getcwd())
        return
    except Exception as e:
        print(f"Error loading train data: {str(e)}")
        return

    try:
        test_data = pd.read_csv("data/test.csv")
        print(f"Successfully loaded test data with shape: {test_data.shape}")
    except FileNotFoundError:
        print("Error: Could not find data/test.csv")
        print("Current working directory:", os.getcwd())
        return
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        return
    
    # Preprocess the data
    print("Preprocessing data...")
    idList = test_data['id']
    y = train_data['Depression']
    train_data, test_data = preprocess_data(train_data, test_data)
    
    # Check if the data is sparse and convert to DataFrame if necessary
    if isinstance(train_data, np.ndarray):
        train_data = pd.DataFrame(train_data)
    if isinstance(test_data, np.ndarray):
        test_data = pd.DataFrame(test_data)

    X = train_data
    
    # 分割資料集為訓練集與測試集
    try:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError as e:
        print(f"Error during train-test split: {str(e)}")
        return
    
    # 定義要測試的 kernel 和對應的參數
    kernels = ['linear', 'rbf', 'sigmoid']
    # kernels = ['linear']
    results = []

    for kernel in kernels:
        print(f"\n{'='*50}")
        print(f"Starting SVM training with kernel = {kernel}")
        print(f"{'='*50}")
        
        # 計算 class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights_dict = dict(enumerate(class_weights))
        print("Class Weights:", class_weights_dict)
        
        # 定義模型和參數網格
        parameters = {'kernel': [kernel], 'C': [0.001, 0.1, 1], 'class_weight': [class_weights_dict]}
        if kernel in ['rbf', 'sigmoid']:
            parameters['gamma'] = ['scale']  # 固定 gamma 為 'scale'
        
        clSvm = SVC()
        grid_search = GridSearchCV(clSvm, parameters, cv=KFold(5), scoring='accuracy', verbose=1)
        grid_search.fit(X_train, y_train)
        
        # 顯示最佳模型資訊
        print("\nBest Model Information:")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")
        
        # 顯示所有參數組合的結果
        print("\nAll Parameter Combinations Results:")
        results_df = pd.DataFrame(grid_search.cv_results_)
        for mean_score, params in zip(results_df['mean_test_score'], results_df['params']):
            print(f"Parameters: {params}")
            print(f"Mean Score: {mean_score:.4f}\n")
        
        # 預測
        y_pred_for_acc = grid_search.predict(X_val)
        y_pred_for_submit = grid_search.predict(test_data)
                
        # 評估指標
        acc = accuracy_score(y_val, y_pred_for_acc)
        precision = precision_score(y_val, y_pred_for_acc)
        recall = recall_score(y_val, y_pred_for_acc)
        f1 = f1_score(y_val, y_pred_for_acc)
        
        # 儲存結果
        results.append({
            'Kernel': kernel,
            'Best Params': grid_search.best_params_,
            'accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        # 顯示分類報告
        print(f"\nClassification Report for kernel = {kernel}:\n")
        print(classification_report(y_val, y_pred_for_acc))
        
        # 繪製混淆矩陣
        disp = ConfusionMatrixDisplay.from_estimator(grid_search, X_val, y_val, cmap='Blues', values_format='d')
        disp.ax_.set_title(f"Confusion Matrix for kernel = {kernel}")
        plt.savefig(f"images/confusion_matrix_{kernel}.png")
        plt.show()
        
        # Create submission file
        submission = pd.DataFrame({'id': idList, 'Depression': y_pred_for_submit})
        submission.to_csv(f"submission_{kernel}.csv", index=False)
        print(f"Submission file created as 'submission_{kernel}.csv'")
        
    # 將結果轉為 DataFrame 方便查看
    results_df = pd.DataFrame(results)
    print("\nComparison of SVM Kernels:\n")
    print(results_df)

    # 可視化不同 kernel 的結果
    metrics = ['accuracy', 'Precision', 'Recall', 'F1-Score']
    results_df.set_index('Kernel', inplace=True)

    plt.figure(figsize=(12, 8))
    results_df[metrics].plot(kind='bar', colormap='viridis')
    plt.title('SVM Kernel Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig("images/svm_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
