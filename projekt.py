import numpy as np
#np.set_printoptions(threshold=np.inf)
import pandas as pd
from sklearn.metrics import confusion_matrix,precision_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


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
        train_test_split(self.data, self.target, test_size=ts, random_state=5)
        print(f"Dane zostały pomyslnie podzielone. Rozmiar danych trenujących: {np.size(self.target_train)}. Rozmiar danych testujących: {np.size(self.target_test)}.")
    
    def classifier(self):
        option = 0
        clf = 0
        while True:
            print('')
            print('-----------------MENU---------------------')
            print("0.Generuj nowy podział danych")
            print("1.Klasyfikacja DecisionTreeClassifier")
            print("2.Klasyfikacja GaussianNB")
            print("3.Klasyfikacja SVC")
            print("3.Klasyfikacja KNeighborsClassifier (n=3)")
            print("Wybierz q, aby zakończyć")
            option = input('>>Podaj kod operacji, którą chcesz wykonać: ')
            print('')
            if option=='1':
                print("=======Wybrana Klasyfikacja: DecisionTreeClassifier=======")
                clf=DecisionTreeClassifier()
            elif option=='2':
                print("=======Wybrana Klasyfikacja: GaussianNB=======")
                clf=GaussianNB()
            elif option=='3':
                print("=======Wybrana Klasyfikacja: SVC=======")
                clf=SVC()
            elif option=='4':
                print("=======Wybrana Klasyfikacja: KNeighborsClassifier (n=3)=======")
                clf=KNeighborsClassifier(n_neighbors=3)
            elif option=='0':
                print('Generacja nowego podziału.')
                size = input('>>Podaj ilosc elementów w zbiorze testującym: ')
                self.splitData(int(size))
            elif option=='q':
                break;
            else:
                print("Niepoprawny kod. Spróbuj ponownie.")        
           
            if option!='0':
                print(f"Rozmiar danych trenujących: {np.size(self.target_train)}. Rozmiar danych testujących: {np.size(self.target_test)}.")
                clf.fit(self.data_train,self.target_train)
                
                predicted_train_val = clf.predict(self.data_train)
                conf_matrix = confusion_matrix(self.target_train, predicted_train_val)
                print("Macierz konfuzji dla danych trenujących: ")
                print(conf_matrix)
                p_score = precision_score(self.target_train, predicted_train_val, average='micro')
                print("Precision_score dla danych trenujących: {0:0.2f}".format(p_score))
                
                predicted_test_val = clf.predict(self.data_test)
                conf_matrix = confusion_matrix(self.target_test, predicted_test_val)     
                print("Macierz konfuzji dla danych testujących: ")
                print(conf_matrix)
                p_score = precision_score(self.target_test, predicted_test_val, average='micro')
                print("Precision_score dla danych testujących: {0:0.2f}".format(p_score))                                                                                                                        
            

project=Project("https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data")
size = input('>>Podaj ilosc elementów w zbiorze testującym: ')
project.splitData(int(size))
project.classifier()
