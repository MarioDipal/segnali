import pandas as pd
from featextr import will_ampl, max_fact_len, peak_frequency, waveform_lenght, slope_sign_change, mean_abs_value, root_mean_square, ar_coeff
from dropbaseline import featured_df, add_list, featured_df_continuo
from plotfun import plot_dataframe, plot_feature


df = pd.read_excel('segnali excel/2022p65cb_PCCortBulb Trace Data.xlsx') #scelta segnale

df = df.set_index(df.columns[0])
plot_dataframe(df) #mostra dataframe

featdf = pd.DataFrame(featured_df_continuo(df)) #crea il df delle feature
featdf = featdf.rename(index={0: 'Target'})
featdf.insert(0, "baseline", 0)

add_list(featdf,will_ampl(df, 112), 'Willison')
add_list(featdf, max_fact_len(df), 'Max Fract Len')
add_list(featdf, waveform_lenght(df), 'Waveform Len')
add_list(featdf, slope_sign_change(df), 'Slope Change')
add_list(featdf, peak_frequency(df, 200), 'Freq Picco')
add_list(featdf, mean_abs_value(df), 'Mean Abs Value')
add_list(featdf, root_mean_square(df), 'Root Mean Square') #valore per riferimento
mean_coeffs, var_coeffs = ar_coeff(df)
add_list(featdf, mean_coeffs, 'Media AR') #valore per riferimento
add_list(featdf, var_coeffs, 'Varianza AR') #valore per riferimento

plot_feature(featdf)

