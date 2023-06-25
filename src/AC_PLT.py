import numpy as np
import pandas as pd
from collections import Counter 
import sklearn.cluster
from scipy.spatial import distance


class AC_PLT:

    def __init__(
            self, 
            n_clusters:int = '500', 
            init:str = 'k-means++', 
            n_init='auto', 
            tol:float= 1e-4,  
            random_state:int = 0, 
            algorithm:str = 'lloyd', 
            copy_x:bool =True, 
            max_iter:int =300,
            verbose:int =0
            ):
        """
        n_clusters: number of cluster in the k-Means model
        """
        
        self.n_clusters = n_clusters # number of clusters
        self.KMeans_dict = {} # dictionary of all the humans codifications for each Cluster
        self.KMeans_categories = {} # dictionary for the most frecuent value in the centroid
        self.km = sklearn.cluster.KMeans(           # creates de k-means object
            n_clusters=self.n_clusters, 
            random_state=random_state,
            init=init,
            n_init=n_init,
            algorithm=algorithm, 
            copy_x=copy_x,
            max_iter=max_iter,
            tol=tol, 
            verbose=verbose
        ) 
        
        
    def most_frequent(self, List:list) -> list: 
        """
        Recives a list of words, and return the word most frequente of
        the list
        """
        # counter of occurence of a code in a list
        occurence_count = Counter(List) 
        
        # Return the first code with more occurence
        return occurence_count.most_common(1)[0][0] 


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Recives the train dataset and the number of clusters to train 
        the k-means model
        """
        # Train the k-means algorithm
        self.km.fit(X)

        # Dataframe of train dataset
        df = pd.DataFrame(
            np.concatenate([
                np.reshape(y, (-1, 1)),                     # Human codification
                np.reshape(self.km.labels_, (-1, 1)),       # Number of the KMean centroid
                ], axis=1), 
            columns=['Human', 'KMeans'])

        # create a dictionary of all the humans codifications for each Cluster
        self.KMeans_dict = df.groupby(by='KMeans')['Human'].apply(list).to_dict()

        # Fill a dictionary with the most frecuent value in the centroid
        for key, val in self.KMeans_dict.items():
            self.KMeans_categories[key] = self.most_frequent(val)
        
        # Generates the prediction for the train dataset
        df['KM_Prediction'] = df['KMeans'].map(self.KMeans_categories)


    def get_distances(self, X: np.ndarray) -> None:
        """
        recives the test data to calculate the distances of each frase, return 
        a matrix with the distances sorted
        """
        
        # Distance matrix of each test point to each cluster center
        distance_matrix = distance.cdist(X.astype(float), self.km.cluster_centers_, 'euclidean')
        
        # Sorting distances
        self.topk=np.argsort(distance_matrix,axis=1)
        
    
    def set_labels(self) -> None:
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


    def get_accuracies(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Recives the test matrix and return the accuracies of the 
        diferents predictions
        """
        self.get_distances(X)
        self.set_labels()
        #Creating the accuracy table to check each data point
        testLabel=np.zeros(self.topKS.shape)
        indexes_method0=pd.DataFrame(np.zeros((self.topKS.shape[0],2)), columns=['index', 'value']) 

        #For each data point
        for i in range(testLabel.shape[0]):
            #Checking if some of the cluster is able to classify it right
            boolClass=self.topKS.iloc[i,:]==y[i]
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

    
    def set_params(self, **params): 
        return self.km.set_params(**params)

    def get_params(self, deep:bool=True): 
        return self.km.get_params(deep=deep)
    # def get(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    #     """
    #     Recives two numpy bi-dimentionals arrays and returns the accuracy of the model
    #     """
    #     self.get_distances(X)
    #     self.set_labels()
    #     return self.get_accuracies(y)
    
    
    def suggestions(self, X: np.ndarray, n_codes: int=1) -> pd.DataFrame:
        self.get_distances(X)
        self.set_labels()
        return np.array(self.topKS.iloc[:, :n_codes])
    
    def predict(self, X: np.ndarray):
        self.get_distances(X)
        self.set_labels()
        return np.reshape(self.topKS.iloc[:, :1], (1, -1))[0]
                
                
    def get_inertia(self):
        return self.km.inertia_