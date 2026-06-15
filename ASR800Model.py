import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import seaborn as sns
import joblib
import shap

df = pd.read_excel('Rp-实验值-20250908-new.xlsx')

X = df[['avg_a_radius','avg_b_radius','avg_a_en','avg_b_en','avg_a_sa','avg_b_sa','avg_a_ie','avg_b_ie','t']].values
y = np.log10(df['Rp @ 800 °C'].values)

scA = StandardScaler()
Xs = scA.fit_transform(X)

joblib.dump(scA, 'scA.pkl')

X_train,X_test,y_train,y_test = train_test_split(Xs,y,test_size=0.2,random_state=1501)

model1 = SVR(kernel='rbf',C=50,gamma='auto')
model2 = RandomForestRegressor(n_estimators=100,random_state=42)
model3 = XGBRegressor(n_estimators=100,random_state=42)
model4 = GradientBoostingRegressor(n_estimators=100,random_state=42)

model = VotingRegressor([('SVR',model1),('RF',model2),('XGB',model3),('GBDT',model4)])
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

rc = {'font.sans-serif': 'SimHei',
      'axes.unicode_minus': False}
sns.set(rc=rc)

sns.set_theme(style='whitegrid', font='Times New Roman', font_scale=1.2)
palette = sns.color_palette('pastel')   # 柔和配色

# 2. 创建画布
fig, ax = plt.subplots(figsize=(7, 6))

# 3. 散点图
ax.scatter(
    y_train, y_train_pred,
    color=palette[0], alpha=0.75, s=60, label='Train'
)
ax.scatter(
    y_test, y_test_pred,
    color=palette[2], alpha=0.75, s=60, label='Test'
)

# 4. 45° 参考线
lim_min = min(ax.get_xlim()[0], ax.get_ylim()[0])
lim_max = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([lim_min, lim_max], [lim_min, lim_max],
        ls='--', lw=1.5, color='gray', zorder=0)

# 5. 轴标签 & 标题
ax.set_xlabel('Emission-true', fontsize=14)
ax.set_ylabel('Emission-predict', fontsize=14)
ax.set_title(
    f'Train R² = {r2_score(y_train, y_train_pred):.2f} | '
    f'Test R² = {r2_score(y_test, y_test_pred):.1f}',
    fontsize=16, pad=20
)

# 6. 图例
ax.legend(frameon=True, fancybox=True, shadow=True)

# 7. 紧凑布局 & 显示
plt.tight_layout()
plt.show()

print('trainset_r2score')
print(r2_score(y_train, y_train_pred))
print('testset_r2score')
print(r2_score(y_test, y_test_pred))
print('trainset_mae')
print(mean_absolute_error(y_train, y_train_pred))
print('testset_mae')
print(mean_absolute_error(y_test, y_test_pred))
print('trainset_rmse')
print(np.sqrt(mean_squared_error(y_train, y_train_pred)))
print('testset_rmse')
print(np.sqrt(mean_squared_error(y_test, y_test_pred)))

explainer = shap.KernelExplainer(model.predict, Xs)
shap_values = explainer.shap_values(Xs)
features = ['avg_a_radius','avg_b_radius','avg_a_en','avg_b_en','avg_a_sa','avg_b_sa','avg_a_ie','avg_b_ie','t']
shap.summary_plot(shap_values,Xs,plot_type="bar",feature_names=features)
shap.summary_plot(shap_values, Xs,feature_names=features,)




