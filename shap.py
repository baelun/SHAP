
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

df_data = pd.read_excel("C://python_flask/model_6/train_data_6.xlsx")


X = df_data.drop(labels=['label'],axis=1).values # 移除Label並取得剩下欄位資料
y = df_data['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print('train shape:', X_train.shape)
print('test shape:', X_test.shape)

#使用KNN演算法
from sklearn.neighbors import KNeighborsClassifier
knnModel = KNeighborsClassifier(n_neighbors=1)
knnModel.fit(X_train,y_train)
predicted_test_y = knnModel.predict(X_test)


#use shap
x_feature_names = df_data.drop(['label'], axis=1).columns
y_label_names = ['0', '1','2','3','4','5','6','7','8','9']
# Create an explainer object
explainer = shap.KernelExplainer(knnModel.predict, X_train[:50]) 
#有.predict與.predict_proba
#類別機率：當您解釋機率時，您會深入了解為什麼模型為每個類別分配特定的機率，而不僅僅是理解為什麼它選擇最高機率的類別。這有助於識別模型不確定或將多個類別視為潛在預測的情況。
shap_values = explainer.shap_values(X_test)
# Visualize SHAP values for the first class
 shap.summary_plot(shap_values[:, :, 1], X_train, feature_names=x_feature_names)
# Visualize SHAP values for all classes
shap.summary_plot(shap_values, X_test,plot_type='dot', feature_names=x_feature_names)
#plot_type---'bar','dot','violin'
