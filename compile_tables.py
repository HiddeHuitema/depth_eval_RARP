import pandas as pd

import pandas as pd

data_train = {
  "Model": ["EndoDac", "Depth anything v2", "Depth anything", "DinoV2 DPT"],
  "Abs rel": ["0.062", "0.131", "0.138", "0.116"],
  "Sq rel": ["0.535", "2.514", "2.430", "1.918"],
  "RMSE": ["5.515", "10.777", "10.998", "8.936"],
  "RMSE log": ["0.083", "0.159", "0.167", "0.141"],
  "$\delta_1$": ["0.962", "0.836", "0.817", "0.882"],
  "$\delta_2$": ["0.998", "0.973", "0.976", "0.975"],
  "$\delta_3$": ["1.000", "0.993", "0.994", "0.992"]
}

len_train = 15351
df_train = pd.DataFrame(data_train)
data_val = {
  "Model": ["EndoDac", "Depth anything v2", "Depth anything", "DinoV2 DPT"],
  "Abs rel": ["0.079", "0.162", "0.182", "0.126"],
  "Sq rel": ["0.740", "3.239", "3.958", "1.668"],
  "RMSE": ["6.660", "13.098", "14.407", "9.160"],
  "RMSE log": ["0.105", "0.196", "0.220", "0.155"],
  "$\delta_1$": ["0.953", "0.763", "0.715", "0.846"],
  "$\delta_2$": ["0.995", "0.959", "0.945", "0.985"],
  "$\delta_3$": ["0.999", "0.991", "0.985", "0.995"]
}

len_val = 1705
df_val = pd.DataFrame(data_val)


data_test = {
  "Model": ["EndoDac", "Depth anything v2", "Depth anything", "DinoV2 DPT"],
  "Abs rel": ["0.051", "0.091", "0.120", "0.091"],
  "Sq rel": ["0.389", "1.107", "1.774", "1.066"],
  "RMSE": ["4.688", "7.602", "9.599", "7.607"],
  "RMSE log": ["0.073", "0.118", "0.154", "0.124"],
  "$\delta_1$": ["0.980", "0.919", "0.850", "0.920"],
  "$\delta_2$": ["0.996", "0.990", "0.980", "0.986"],
  "$\delta_3$": ["0.999", "0.999", "0.996", "0.996"]
}

len_test = 551
df_test = pd.DataFrame(data_test)


data_total = pd.DataFrame()
# print(df_test.loc[0])
print(type(data_train['$\delta_1$']))
for metric in ['Abs rel','Sq rel','RMSE','RMSE log','$\delta_1$','$\delta_2$','$\delta_3$']:
    data_total[metric] = [(float(data_train[metric][i])*len_train+float(data_test[metric][i])*len_test+float(data_val[metric][i])*len_val)/(len_val+len_test+len_train) for i in range(len(data_train[metric]))]



print(data_total)