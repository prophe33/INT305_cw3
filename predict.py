import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import itertools
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data = pd.read_csv("./vgsales/vgsales-12-4-2019-short.csv")

data['Global_Sales'][data['Global_Sales'].isnull()] = data['Total_Shipped'][data['Global_Sales'].isnull()]
data["sales"] = data['Global_Sales']
# data["Year"] = data["Year_of_Release"]
data = data[['Name', 'Platform', 'Genre', 'Publisher', 'Developer', 'Year', 'sales']]
data = data.dropna()


def hit(sales):
    if sales >= 1:
        return 2
    elif 0.2 <= sales < 1:
        return 1
    else:
        return 0


Platform_Dict = {
    "PC": 1,
    "DS": 2,
    "PS2": 3,
    "Wii": 4,
    "PS3": 5,
    "PSP": 6,
    "X360": 7,
    "PS": 8,
    "PS4": 9,
    "GBA": 10,
    "XB": 11,
    "PSV": 12,
    "3DS": 13,
    "GC": 14,
    "XOne": 15,
    "N64": 16,
    "NS": 17,
    "SNES": 18,
    "SAT": 19,
    "WiiU": 20,
    "2600": 21,
    "NES": 22,
    "GB": 23,
}

data['Hit'] = data['sales']
data['Hit'] = data['Hit'].apply(lambda x: hit(x))
data1 = data.drop('sales', axis=1)

plt.figure(figsize=(10, 5))
sns.countplot(x='Platform', data=data, order=data['Platform'].value_counts().index)
plt.xticks(rotation=45)
plt.show()

data1 = data1[~data1["Platform"].str.contains("DC|GEN|PSN|NG|XBL|GBC|WS|VC|SCD|3DO|Mob|WW|PCE|PCFX|GG|Amig|OSX")]

plt.figure(figsize=(10, 5))
sns.countplot(x='Platform', data=data1, order=data1['Platform'].value_counts().index)
plt.xticks(rotation=45)
plt.show()

data1['Platform'] = data1['Platform'].map(Platform_Dict)

plt.figure(figsize=(10, 5))
sns.countplot(x='Genre', data=data1, order=data1['Genre'].value_counts().index)
plt.xticks(rotation=45)
plt.show()

data1 = data1[~data1["Genre"].str.contains("MMO|Party|Sandbox|Education|Board Game")]

plt.figure(figsize=(10, 5))
sns.countplot(x='Genre', data=data1, order=data1['Genre'].value_counts().index)
plt.xticks(rotation=45)
plt.show()

Genre_Dict = {
    "Action": 1,
    "Sports": 2,
    "Misc": 3,
    "Adventure": 4,
    "Role-Playing": 5,
    "Shooter": 6,
    "Racing": 7,
    "Simulation": 8,
    "Platform": 9,
    "Strategy": 10,
    "Fighting": 11,
    "Puzzle": 12,
    "Action-Adventure": 13,
    "Music": 14,
    "Visual Novel": 15,
}


data1['Genre'] = data1['Genre'].map(Genre_Dict)


data1 = pd.get_dummies(data1, columns=['Publisher', 'Developer'],
                       prefix=['Publisher', 'Developer'])
data2 = data1.drop(["Name"], axis=1)
data2 = data2.drop(["Year"], axis=1)
print(data2.shape)
y = data2["Hit"].values
X = data2.drop(["Hit"], axis=1).values

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=5)


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.show()


rd = RandomForestClassifier(n_estimators=110, max_depth=150, random_state=90)
li = OneVsRestClassifier(LinearSVC(C=0.1, random_state=90))
rd.fit(Xtrain, ytrain)
li.fit(Xtrain, ytrain)
print("RandomForest")
print("Traing Score:%f" % rd.score(Xtrain, ytrain), "Testing Score:%f" % rd.score(Xtest, ytest))
print("LinearSVC")
print("Traing Score:%f" % li.score(Xtrain, ytrain), "Testing Score:%f" % li.score(Xtest, ytest))
prediction_rd = rd.predict(Xtest)
prediction_li = li.predict(Xtest)

cm = confusion_matrix(ytest, prediction_rd)
plot_confusion_matrix(cm, classes=['< 200k', 'between', '> 1million'])

cm = confusion_matrix(ytest, prediction_li)
plot_confusion_matrix(cm, classes=['< 200k', 'between', '> 1million'])
