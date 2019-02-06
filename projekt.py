import numpy as np
#np.set_printoptions(threshold=np.inf)
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


class Project(object):
    def __init__(self, datasetURL):									#constructor with dataset url	
        dataset=pd.read_csv(datasetURL)
        self.data = dataset.loc[0:,["MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","MDVP:Jitter(%)","MDVP:Jitter(Abs)","MDVP:RAP","MDVP:PPQ",      #load data
         "Jitter:DDP","MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5","MDVP:APQ","Shimmer:DDA",
         "NHR","HNR","RPDE","DFA","spread1","spread2","D2","PPE"]].values.astype(np.float)
        self.target = dataset.loc[0:, ['status']].values.astype(np.float)                                                                       #load status
        print("Dane zostały pomyslnie wczytane.")
    
    def splitData(self,ts):                                                                                                           #to to rozmiar zbioru testującego
        self.data_train, self.data_test, self.target_train, self.target_test = \
        train_test_split(self.data, self.target, test_size=ts)
        print(f"Dane zostały pomyslnie podzielone. Rozmiar danych trenujących: {np.size(self.target_train)}. Rozmiar danych testujących: {np.size(self.target_test)}.")
    
    def classifier(self):
        option = 0
        clf = 0
        while True:
            print('')
            print('-----------------MENU---------------------')
            print("0.Generuj nowy podział danych")
            print("1.Algorytm DecisionTreeClassifier")
            print("2.Algorytm GaussianNB")
            print("Wybierz q, aby zakończyć")
            option = input('>>Podaj kod operacji, którą chcesz wykonać: ')
            print('')
            if option=='1':
                print("=======Wybrany algorytm: DecisionTreeClassifier=======")
                clf=DecisionTreeClassifier()
            elif option=='2':
                print("=======Wybrany algorytm: GaussianNB=======")
                clf=GaussianNB()
            elif option=='0':
                print('Generacja nowego podziału.')
                size = input('>>Podaj rozmiar danych testujących (np. 0.2): ')
                self.splitData(float(size))
            elif option=='q':
                break;
            else:
                print("Niepoprawny kod. Spróbuj ponownie.")        
           
            if option!='0':
                clf.fit(self.data_train,self.target_train)
                conf_matrix = confusion_matrix(self.target_test, clf.predict(self.data_test))
                print("Macierz konfusji: ")
                print(conf_matrix)
                acc = accuracy_score(self.target_test, clf.predict(self.data_test))
                print("Precyzja modelu wynosi: {0:0.2f}".format(acc))                                                                                                                        
            

project=Project("https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data")
size = input('>>Podaj rozmiar danych testujących (np. 0.2): ')
project.splitData(float(size))
project.classifier()
