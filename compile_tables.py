import pandas as pd
import numpy as np

data_train = {
  "": ["EndoDac", "Depth anything v2", "Depth anything", "DinoV2 DPT","SurgeDepth"],
  "Abs rel": ["0.061", "0.072", "0.078", "0.104 ", "0.078"],
  "Sq rel": ["0.479", "0.705", "0.792", "1.333", "0.768"],
  "RMSE": ["5.180", "6.056", "6.444", "8.347", "6.631"],
  "RMSE log": ["0.080", "0.091", "0.098", "0.129", "0.103"],
  "$\delta_1$": ["0.967", "0.959", "0.950", "0.907", "0.946"],
  "$\delta_2$": ["0.999", "0.997", "0.997", "0.987", "0.997"],
  "$\delta_3$": ["1.000", "0.999", "0.999", "0.997", "1.000"],
  "PSNR": ["35.689","35.102","34.776","33.542","34.635" ],
  "SSIM": ["0.309","0.310","0.309","0.299","0.315"]
}

len_train = 15351
df_train = pd.DataFrame(data_train)
df_train = df_train.replace(r'^\s*$', "0", regex=True)
print(df_train)

data_val = {
  "": ["EndoDac", "Depth anything v2", "Depth anything", "DinoV2 DPT", "SurgeDepth"],
  "Abs rel": ["0.077", "0.089", "0.098", "0.121", "0.095"],
  "Sq rel": ["0.690", "1.007", "1.271", "1.548", "1.127"],
  "RMSE": ["6.311", "7.383", "8.005", "9.624", "7.945"],
  "RMSE log": ["0.102", "0.112", "0.120", "0.148", "0.122"],
  "$\delta_1$": ["0.952", "0.939", "0.899", "0.872", "0.908"],
  "$\delta_2$": ["0.996", "0.995", "0.995", "0.990", "0.996"],
  "$\delta_3$": ["0.999", "0.999", "1.000", "0.999", "0.999"],
  "PSNR": ["35.012 ","34.551","34.400","33.107","34.365" ],
  "SSIM": ["0.289","0.287","0.286","0.281","0.291"]
}

len_val = 1705
df_val = pd.DataFrame(data_val)
df_val = df_val.replace(r'^\s*$', "0", regex=True)

data_test = {
  "": ["EndoDac", "Depth anything v2", "Depth anything", "DinoV2 DPT", "SurgeDepth"],
  "Abs rel": ["0.053", "0.063", "0.076", "0.085", "0.065"],
  "Sq rel": ["0.413", "0.508", "0.808", "0.841", "0.550"],
  "RMSE": ["4.818", "5.094", "6.197", "7.048", "5.617"],
  "RMSE log": ["0.076", "0.084", "0.099", "0.113", "0.092"],
  "$\delta_1$": ["0.977", "0.969", "0.943", "0.939", "0.956"],
  "$\delta_2$": ["0.996", "0.997", "0.995", "0.994", "0.996"],
  "$\delta_3$": ["0.999", "1.000", "0.999", "0.998", "1.000"],
  "PSNR": ["35.285 ","34.871","34.361","33.456","34.189" ],
  "SSIM": ["0.360","0.370","0.373","0.358","0.371"]
}

len_test = 551
df_test = pd.DataFrame(data_test)
df_test = df_test.replace(r'^\s*$', '0', regex=True)

data_total = pd.DataFrame()
# print(df_test.loc[0])
# print(type(data_train['$\delta_1$']))
for metric in ['Abs rel','Sq rel','RMSE','RMSE log','$\delta_1$','$\delta_2$','$\delta_3$',"PSNR","SSIM"]:
    data_total[metric] = [(float(data_train[metric][i])*len_train+float(data_test[metric][i])*len_test+float(data_val[metric][i])*len_val)/(len_val+len_test+len_train) for i in range(len(data_train[metric]))]



print(data_total)