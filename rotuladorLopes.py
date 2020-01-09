import man_dados
import pandas as pd
from discretizacao import discretizador as disc
from classificacao import classifica_bd as clas
from rotulacao import rotular as rotulador 
from sklearn import metrics

class Rotulador(object):
	def __init__ (self, bd, attr_cluster_name, discre_method, bins, per_trein, V, folds):
		self.bd = bd
		self.frames_originais = man_dados.group_separator(bd, attr_cluster_name)

		discretizacao = disc(bd, bins, discre_method, attr_cluster_name)
		self.base_discretizada = discretizacao.ddb
		self.infor = discretizacao.infor
		self.frames_discretizados = man_dados.group_separator(self.base_discretizada, attr_cluster_name)
		
		self.classificacao = clas(self.frames_discretizados, attr_cluster_name, per_trein, folds)
		self.rotulo = rotulador(self.frames_originais, self.frames_discretizados, attr_cluster_name, self.classificacao, V, self.infor)

dataset = man_dados.read_csv('./databases/iris.csv')
rotulo = Rotulador(dataset, 'classe', 'EFD', [3,3,3,3], 60, 10, 10).rotulo
print(rotulo)
IS = metrics.silhouette_score(dataset.drop(['classe'], axis=1), dataset['classe'])
BD = metrics.davies_bouldin_score(dataset.drop(['classe'], axis=1), dataset['classe'])
print(IS, BD)
for clt in dataset['classe'].unique():
	regras = [i[1] for i in rotulo if i[0] == clt][0]
	for regra in regras:
		dataset.drop(dataset[(~(dataset[regra[0]]>= regra[1]) & (dataset[regra[0]]<= regra[2])) & (dataset['classe'] == clt)].index, axis=0, inplace=True)
IS = metrics.silhouette_score(dataset.drop(['classe'], axis=1), dataset['classe'])
BD = metrics.davies_bouldin_score(dataset.drop(['classe'], axis=1), dataset['classe'])
print(IS, BD)