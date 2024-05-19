from sklearn import tree

# X e Y são o conjunto de dados que o modelo vai receber para treinar

# Dados de pessoas com suas respectivas alturas, pesos e tamanhos de sapato
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

# Labels de cada um destes
Y = ['male','female','female','female', 'male','male','male','female','male','female','male']


# Chamamos o decision tree classifier
clf = tree.DecisionTreeClassifier()

# Passamos os dados de treino
clf = clf.fit(X, Y)


# Chamamos o método predict
prediction = clf.predict([[190, 70, 43]])

print(prediction)