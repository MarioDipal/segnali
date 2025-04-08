import os
import numpy as np
import pandas as pd
import re
from featextr import will_ampl, max_fact_len, peak_frequency, waveform_lenght, slope_sign_change, mean_abs_value, \
    root_mean_square, ar_coeff, mean_med_freq, av_ampl_cha, wavelet_correlations, MinR, hilbert_tras, \
    discrete_wavelet_tr, short_time_ft, intrinsic_tcd, svd  # ,ramunjan_ft
from dropbaseline import featured_df, add_list
from allarmi import estrai_lista, estrai_ore_per_paziente, estrai_testo_da_docx, ottieni_frasi, \
    cerca_eventi_in_cartella, trova_posizioni_precedenti, starting_time

cartella_segnali = 'segnali2ex'  ###
allelem = os.listdir(cartella_segnali)
segnali = [file for file in allelem if os.path.isfile(os.path.join(cartella_segnali, file))]
df_list = []  # lista per salvare i DataFrame
processed_file = set()
####
testo = estrai_testo_da_docx('filewarning.docx')
warning_kw = ottieni_frasi(testo)
cartellaword = "wordsegnali"
risultati = cerca_eventi_in_cartella(cartellaword, warning_kw)
warning = estrai_lista(risultati)
###

for nome_file in segnali:
    if nome_file in processed_file:  # evita di leggere più volte lo stesso segnale
        continue

    percorso_completo = os.path.join(cartella_segnali, nome_file)
    print(f"Leggendo file: {nome_file}")  # vede a che punto siamo

    df = pd.read_excel(percorso_completo)  # legge il file
    df = df.set_index(df.columns[0])
    start_time = df.iloc[0]  ###
    df = df[1:]  ###
    df = df.apply(pd.to_numeric, errors='coerce')  # trasforma gli element del df in float###

    match = re.search(r'p(\d+)cb_', nome_file)
    if match:
        numero_segnale = match.group(1)
    else:
        numero_segnale = ''
    numero_segnale = np.array(numero_segnale, dtype=int)
    # plot_dataframe(df) #mostra dataframe
    featdf = pd.DataFrame(featured_df(df))  # crea il df delle feature
    featdf = featdf.rename(index={0: 'Target'})
    featdf.insert(0, "baseline", 0)
    add_list(featdf, will_ampl(df), 'Willison')
    add_list(featdf, max_fact_len(df), 'Max Fract Len')
    add_list(featdf, waveform_lenght(df), 'Waveform Len')
    add_list(featdf, slope_sign_change(df), 'Slope Change')
    add_list(featdf, peak_frequency(df, 200), 'Freq Picco')
    add_list(featdf, mean_abs_value(df), 'Mean Abs Value')
    add_list(featdf, root_mean_square(df), 'Root Mean Square')  # valore per riferimento
    mean_coeffs, var_coeffs = ar_coeff(df)
    add_list(featdf, mean_coeffs, 'Media AR')  # valore per riferimento
    add_list(featdf, var_coeffs, 'Varianza AR')  # valore per riferimento
    freq_media, freq_medn = mean_med_freq(df, 200)
    add_list(featdf, freq_media, 'Frequenza Media')
    add_list(featdf, freq_medn, 'Freqeunza Mediana')
    add_list(featdf, av_ampl_cha(df), 'Avg Ampl Change')
    add_list(featdf, wavelet_correlations(df, 200), 'Wavelet Correlations')
    add_list(featdf, MinR(df, 10), 'MinR')
    rms_amp, rms_fre, plv = hilbert_tras(df)
    add_list(featdf, rms_amp, 'RMS Amp Env Hil')
    add_list(featdf, plv, 'plv Fase istantanea Hil')
    add_list(featdf, rms_fre, 'RMS Freq istantanea Hil')
    cA_mean, cD_mean = discrete_wavelet_tr(df)
    add_list(featdf, cA_mean, 'DWT HF')
    add_list(featdf, cD_mean, 'DWT LF')
    mav_mag, mav_pha, max_mag = short_time_ft(df)
    add_list(featdf, mav_mag, 'STFT MAV Magn')
    add_list(featdf, mav_pha, 'STFT MAV Fase')
    add_list(featdf, max_mag, 'STFT MAX Magn')
    comp, resid = intrinsic_tcd(df)
    add_list(featdf, comp, 'Mean Intrinsic TSD Comp')
    add_list(featdf, resid, 'Mean Intrinsic TSD Res')
    cos_sim, eucl_dis = svd(df)
    add_list(featdf, cos_sim, 'SVD cos similaties')
    add_list(featdf, eucl_dis, 'SVD eucl dist')
    featdf.insert(0, 'id_pa', numero_segnale)
    df_list.append(featdf)  # aggiunge il DataFrame alla lista

    time = starting_time(start_time)
    warning_pp = estrai_ore_per_paziente(warning, numero_segnale)
    posizioni = trova_posizioni_precedenti(warning_pp, time)

    ###

#ora c'è da vedere se i warning segnati hanno relaiozne con quelli dati dal calo della baseline o dei parametri
# unisce tutti i DataFrame in un unico DataFrame finale
df_feats = pd.concat(df_list, ignore_index=True)
#df_feats.to_excel('output.xlsx', index=False)
#print(df_feats)
