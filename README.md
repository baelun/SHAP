# SHAP
using SHAP to explain model, base on the KIM-LHC posture predict model



### use predict api
explainer = shap.KernelExplainer(knnModel.predict, X_train[:50]) 
![圖片](https://github.com/baelun/SHAP/assets/121599449/ed1b6e50-aff2-4d55-afc9-c85382d0a371=50%)


### use predict_proba api
explainer = shap.KernelExplainer(knnModel.predict_proba, X_train[:50]) 
![圖片](https://github.com/baelun/SHAP/assets/121599449/d991379a-2321-4fc2-ab8c-afa7e6be7fcb=50%)


