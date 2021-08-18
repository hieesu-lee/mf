import numpy as np
import pandas as pd

class MatrixFactorization(object):
    """Matrix factorization
    from https://machinelearningcoban.com/2017/05/31/matrixfactorization/
    """

    def __init__(self, Y_data, K, lamb, X_init = None, W_init = None, \
                    learning_rate = 0.5, max_iter = 1000, print_every = 100, user_based = 1):
        self.Y_raw_data = Y_data
        self.K = K
        # regularization parameter
        self.lamb = lamb
        # learning rate for gradient descent
        self.learning_rate = learning_rate
        # maximum number of iterations
        self.max_iter = max_iter
        # print results after print_every iterations
        self.print_every = print_every
        # user-based or item-based
        self.user_based = user_based
        # number of users, items, and ratings. Remember to add 1 since id starts from 0
        self.n_users = int(np.max(Y_data[:, 0])) + 1
        self.n_items = int(np.max(Y_data[:, 1])) + 1
        self.n_ratings = Y_data.shape[0]

        if X_init is None:
            self.X = np.random.randn(self.n_items, K)
        else: # from saved data
            self.X = X_init

        if W_init is None:
            self.W = np.random.randn(K, self.n_users)
        else: # from saved data
            self.W = W_init

        # normalized data, update later in normalized_Y function
        self.Y_data_n = self.Y_raw_data.copy()

    def normalize_Y(self):
        if self.user_based:
            user_col = 0
            item_col = 1
            n_objects = self.n_users
        # if we want to normolize based on item, just switch first two columns of data
        else: # item based
            user_col = 1
            item_col = 0
            n_objects = self.n_items

        users = self.Y_raw_data[:, user_col]
        self.mu = np.zeros((n_objects,))
        for n in range(n_objects):
            # row indices of rating done by user n
            # since indices need to be integers, we need to convert
            ids = np.where(users == n)[0].astype(np.int32)
            # and the corresponding ratings
            ratings = self.Y_data_n[ids, 2]
            # take mean
            m = np.mean(ratings)
            if np.isnan(m):
                m = 0 # to avoid empty array and nan value
            self.mu[n]=m
            # normalize
            self.Y_data_n[ids, 2]=ratings-self.mu[n]

    def loss(self):
        L=0
        for i in range(self.n_ratings):
            # user, item, rating
            n, m, rate = int(self.Y_data_n[i,0], itn(self.Y_data_n[i,1]), self.Y_data_n[ids,2])
            L+=0.5*(rate-self.X[m,:].dot(self.W[:,n]))**2

        # take average
        L/=self.n_ratings
        # regularization
        L+=0.5*self.lamb*(np.linalg.norm(self.X,'fro'))+np.linalg.norm(self.W, 'fro')

        return L



        

def main():
    r_cols=['user_id','movie_id','rating','unix_timestamp']
    ratings_base=pd.read_csv('ml-100k/ub.base',sep='\t',name=r_cols,encoding='latin-1')
    ratings_test=pd.read_csv('ml-100k/ub.test',sep='\t',name=r_cols,encoding='latin-1')

    rate_train=ratings_base.as_matrix()
    rate_test=ratings_test.as_matrix()

    # indices start from 0
    rate_train[:,:2]-=1
    rate_test[:,:2]-=1

if __name__ == "__main__":
    main()
