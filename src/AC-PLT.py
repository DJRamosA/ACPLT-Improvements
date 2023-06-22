import numpy as np
import pandas as pd
from collections import Counter 
import sklearn.cluster
from scipy.spatial import distance


class AC_PLT:

    def __init__(self, n_clusters = 500, vec_len=300, random_state=0):
        """
        n_clusters: number of cluster in the k-Means model
        """
        
        self.vec_len=vec_len
        self.n_clusters = n_clusters # number of clusters
        self.KMeans_dict = {} # dictionary of all the humans codifications for each Cluster
        self.KMeans_categories = {} # dictionary for the most frecuent value in the centroid
        self.km = sklearn.cluster.KMeans(           # creates de k-means object
            n_clusters=self.n_clusters, 
            random_state=random_state,
            n_init='auto'
        ) 
        
        
    def most_frequent(self, List): 
        """
        Recives a list of words, and return the word most frequente of
        the list
        """
        # counter of occurence of a code in a list
        occurence_count = Counter(List) 
        
        # Return the first code with more occurence
        return occurence_count.most_common(1)[0][0] 


    def fit(self, train):
        """
        Recives the train dataset and the number of clusters to train 
        the k-means model
        """
        # Train the k-means algorithm
        self.km.fit(train[:,:self.vec_len])

        # Dataframe of train dataset
        df = pd.DataFrame(
            np.concatenate([
                np.reshape(train[:,self.vec_len+2], (-1, 1)),          # Human codification
                np.reshape(self.km.labels_, (-1, 1)),       # Number of the KMean centroid
                np.reshape(train[:,self.vec_len], (-1, 1))           # Concept of the codification
                ], axis=1), 
            columns=['Human', 'KMeans', 'Concept'])

        # create a dictionary of all the humans codifications for each Cluster
        self.KMeans_dict = df.groupby(by='KMeans')['Human'].apply(list).to_dict()

        # Fill a dictionary with the most frecuent value in the centroid
        for key, val in self.KMeans_dict.items():
            self.KMeans_categories[key] = self.most_frequent(val)
        
        # Generates the prediction for the train dataset
        df['KM_Prediction'] = df['KMeans'].map(self.KMeans_categories)


    def get_distances(self, test):
        """
        recives the test data to calculate the distances of each frase, return 
        a matrix with the distances sorted
        """
        
        # Distance matrix of each test point to each cluster center
        distance_matrix = distance.cdist(test[:,:self.vec_len].astype(float), self.km.cluster_centers_, 'euclidean')
        
        # Sorting distances
        self.topk=np.argsort(distance_matrix,axis=1)
        
    
    def set_labels(self):
        """
        Create a new matrix from the clusters sorted and change the value
        from numeric to the string according the codification
        """
        # Change of the numeric value to the codification 
        self.topKS=pd.DataFrame(self.topk)

        # create a temporal array of the kmeans categories
        tempData = np.array([value for (_, value) in sorted(self.KMeans_categories.items())])
        
        # print(tempData)

        # for each cluster center
        for j in range(self.topKS.shape[1]):
            # set the codification of the numeric value in the topk list
            self.topKS.iloc[:,j]=tempData[self.topk[:,j]]


    def get_accuracies(self, test):
        """
        Recives the test matrix and return the accuracies of the 
        diferents predictions
        """
        #Creating the accuracy table to check each data point
        testLabel=np.zeros(self.topKS.shape)
        indexes_method0=pd.DataFrame(np.zeros((self.topKS.shape[0],2)), columns=['index', 'value']) 

        #For each data point
        for i in range(testLabel.shape[0]):
            #Checking if some of the cluster is able to classify it right
            boolClass=self.topKS.iloc[i,:]==test[i,self.vec_len+2]
            if sum(boolClass)>0:
                getIndex=boolClass.idxmax()
                indexes_method0.iloc[i,0] = getIndex
                indexes_method0.iloc[i,1] = self.topKS.iloc[i,getIndex]
                #Setting the rest of the data point as 1
                testLabel[i,getIndex:]=1
            else:
                indexes_method0.iloc[i,0] = np.nan
                indexes_method0.iloc[i,1] = np.nan
        accuracies=testLabel.sum(axis=0)/testLabel.shape[0]

        return accuracies

    
    def transform(self, test):
        """
        Recives two numpy bi-dimentionals arrays and returns the accuracy of the model
        """
        self.get_distances(test)
        self.set_labels()
        return self.get_accuracies(test)
    
    def suggestions(self, test, n_codes):
        self.get_distances(test)
        self.set_labels()
        return pd.DataFrame(
            np.concatenate([
                np.reshape(test[:, self.vec_len], (-1, 1)), 
                np.reshape(test[:, self.vec_len+1], (-1, 1)), 
                self.topKS.iloc[:, :n_codes]],
                axis=1
                ), 
            columns=['Concept', 'Description']+['top-{} suggestion'.format(i+1) for i in range(n_codes)]
            )
        
    def get_inertia(self):
        return self.km.inertia_