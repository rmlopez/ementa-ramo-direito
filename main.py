import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from TextClassification import *

pd.options.mode.chained_assignment = None

###################################      Limpeza dos dados       #####################################################

data = pd.read_csv('ementa_ramo_direito.csv', encoding='utf-8', sep=';',
                   dtype={'EMENTA': str, 'SG_RAMO_DIREITO': str})
# Remoção de valores ausentes e de ementas com o texto indicando que não há ementa
data = data.dropna()
filtro_menos_de_50_caracteres = data['EMENTA'].str.len() < 50
filtro_contem_palavra_ementa = data['EMENTA'].str.contains('(?:^|\W)EMENTA', regex=True)
filtro_comeca_com_palavra_nao = data['EMENTA'].str.contains('^N[AÃ]O', regex=True)
filtro_comeca_com_palavra_sem = data['EMENTA'].str.contains('^SEM', regex=True)
filtro_contem_expressao_acordao_nao = data['EMENTA'].str.contains('AC[ÓO]RD[ÃA]O N[ÃA]O', regex=True)

data = data[np.invert(filtro_menos_de_50_caracteres & \
                      (filtro_contem_palavra_ementa | filtro_comeca_com_palavra_nao | \
                       filtro_comeca_com_palavra_sem | filtro_contem_expressao_acordao_nao))]
ramos_direito_desconsiderados = ['CM','IN','MA', 'TB']
data = data[np.invert(data['SG_RAMO_DIREITO'].isin(ramos_direito_desconsiderados))]

print('Dados Filtrados')

###################################      Separação dos dados em treino e validação     #####################################################
# print(data['SG_RAMO_DIREITO'].unique())
label_names = sorted(data['SG_RAMO_DIREITO'].unique())
data['INDICES'] = data['SG_RAMO_DIREITO'].apply(lambda x: label_names.index(x))

train, validation = train_test_split(data, test_size=0.2, random_state=42)  # , stratify=data['INDICES'])
trainY = train['INDICES']
trainX = train['EMENTA']
validationY = validation['INDICES']
validationX = validation['EMENTA']

if TextCls.exists_saved_model():
    model = TextCls()
    model.load()
    print('modelo carregado de arquivo')
else:
    model = TextCls(label_names, trainX, trainY)
    model.fit()
    model.save()

y_pred = []
'''for doc in validationX:
    top_preds = model.predict_single(doc)[:2]
    print(doc)
    y_pred.append(label_names.index(top_preds[0][0]))
    for label, score in top_preds:
        print('\t{}\t{}'.format(label, score))'''
y_pred = model.predict(validationX)
model.report(validationX, validationY, y_pred)

print('Fim')




