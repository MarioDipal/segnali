import os
import pandas as pd
from findwarnings import trova_posizionix, numero_segnalex
from featureinsert import calculate_all_features
from plotfun import plot_dataframe, plot_feature_tobs, plot_feat

cartella_segnali = 'segnali2ex'
allelem = os.listdir(cartella_segnali)
segnali = [file for file in allelem if os.path.isfile(os.path.join(cartella_segnali, file))]
df_list = []  # lista per salvare i DataFrame
processed_file = set()

for nome_file in segnali:
    if nome_file in processed_file: continue
    percorso_completo = os.path.join(cartella_segnali, nome_file)
    print(f"Leggendo file: {nome_file}")
    df = pd.read_excel(percorso_completo)
    df = df.set_index(df.columns[0])
    num_trial = trova_posizionix(nome_file, df)
    df = df[1:]  ###
    df = df.apply(pd.to_numeric, errors='coerce')
    featdf = calculate_all_features(df)
    #plot_feat(featdf)
    #print(featdf)

    featdf.insert(0, 'id_pa', numero_segnalex(nome_file))
    df_list.append(featdf)

df_feats = pd.concat(df_list, ignore_index=True) #unisce tutti i DataFrame in un unico DataFrame finale
df_feats.to_excel('feattabella', index=False)
#print(df_feats)




'''
from decisiontree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

unique_ids = df_feats['id_pa'].unique()
train_id, test_id = train_test_split(unique_ids, test_size=0.2, random_state=42)
train_df = df_feats[df_feats['id_pa'].isin(train_id)]
test_df = df_feats[df_feats['id_pa'].isin(test_id)]

X_train = train_df.iloc[:, 1:-1].values
Y_train = train_df.iloc[:, -1].values.reshape(-1, 1)
X_test = test_df.iloc[:, 1:-1].values
Y_test = test_df.iloc[:, -1].values.reshape(-1, 1)

#fit the model
classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
classifier.fit(X_train,Y_train)
classifier.print_tree()
#test the model
Y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, Y_pred)'''
