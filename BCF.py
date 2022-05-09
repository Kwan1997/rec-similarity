from surprise import SVD
from collections import defaultdict
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import prediction_algorithms
from surprise import similarities
import numpy as np
from surprise import PredictionImpossible
from six import iteritems
from surprise import AlgoBase
import heapq
from statistics import median
from statistics import stdev
from surprise.model_selection import train_test_split
from scipy.stats import wasserstein_distance
from scipy.stats import dirichlet
import collections
from tqdm import tqdm
import math
from surprise import accuracy
from collections import OrderedDict
# from surprise.prediction_algorithms.predictions import Prediction




class mySymmetricAlgo(AlgoBase):
    def __init__(self, sim_options={}, verbose=True, **kwargs):

        AlgoBase.__init__(self, sim_options=sim_options, **kwargs)
        self.verbose = verbose

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        ub = self.sim_options['user_based']
        self.n_x = self.trainset.n_users if ub else self.trainset.n_items   # |U|or|I|
        self.n_y = self.trainset.n_items if ub else self.trainset.n_users   # |U|or|I|
        self.xr = self.trainset.ur if ub else self.trainset.ir   # user ratings or item ratings
        self.yr = self.trainset.ir if ub else self.trainset.ur   # user ratings or item ratings

        return self

    def switch(self, u_stuff, i_stuff):
        """Return x_stuff and y_stuff depending on the user_based field."""

        if self.sim_options['user_based']:
            return u_stuff, i_stuff
        else:
            return i_stuff, u_stuff


class BCF(mySymmetricAlgo):
    def __init__(self, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):

        mySymmetricAlgo.__init__(self, sim_options=sim_options,
                               verbose=verbose, **kwargs)

        self.k = k
        self.min_k = min_k
        self.testusers = []

    def fit(self, trainset):

        mySymmetricAlgo.fit(self, trainset)
        print('Computing item similarity matrix...')  # TODO: here
        #  Compute item similarity first
        mySimilarity = np.zeros((self.trainset.n_items, self.trainset.n_items), dtype=np.double)
        tempEmpiricalDistributioni = {}
        tempEmpiricalDistributionj = {}
        for _, _, rate in self.trainset.all_ratings():
            tempEmpiricalDistributioni[rate] = 0
            tempEmpiricalDistributionj[rate] = 0
        tempEmpiricalDistributioni = OrderedDict(sorted(tempEmpiricalDistributioni.items()))
        tempEmpiricalDistributionj = OrderedDict(sorted(tempEmpiricalDistributionj.items()))
        pbar = tqdm(total=self.trainset.n_items * self.trainset.n_items)
        for itemi, ratesi in self.trainset.ir.items():
            for itemj, ratesj in self.trainset.ir.items():
                FinalDistri = []
                FinalDistrj = []
                totali = 0
                totalj = 0
                for _, ratei in ratesi:
                    tempEmpiricalDistributioni[ratei] += 1
                    totali += 1
                for _, ratej in ratesj:
                    tempEmpiricalDistributionj[ratej] += 1
                    totalj += 1
                for rate in sorted(tempEmpiricalDistributioni):
                    FinalDistri.append(tempEmpiricalDistributioni[rate]/totali)
                    FinalDistrj.append(tempEmpiricalDistributionj[rate]/totalj)
                mySimilarity[itemi, itemj] = np.sum(np.sqrt(np.array(FinalDistri)*np.array(FinalDistrj)))
                for key in sorted(tempEmpiricalDistributioni):
                    tempEmpiricalDistributioni[key] = 0
                    tempEmpiricalDistributionj[key] = 0
                pbar.update(1)
        pbar.close()
        print('Done computing item similarity matrix.')
        print('Computing user similarity matrix...')
        mySimilarity2 = np.zeros((self.trainset.n_users, self.trainset.n_users), dtype=np.double)
        mySimilarity3 = np.zeros((self.trainset.n_users, self.trainset.n_users), dtype=np.double)
        self.means = np.zeros(self.n_x)
        for x, ratings in iteritems(self.xr):
            self.means[x] = np.mean([r for (_, r) in ratings])

        self.median = np.zeros(self.n_x)
        for x, ratings in iteritems(self.xr):
            self.median[x] = median([r for (_, r) in ratings])

        self.std = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            self.std[x] = np.std([r for (_, r) in ratings])

        self.medianvalue = (trainset.rating_scale[1] + trainset.rating_scale[0]) / 2.0
        self.maxminesmin = (trainset.rating_scale[1] - trainset.rating_scale[0]) * 1.0

        self.medstd = np.zeros(self.n_x)
        for x, ratings in iteritems(self.xr):
            # self.medstd[x] = math.sqrt(sum(pow(x - self.median[x], 2) for (_, x) in ratings) / len(ratings))
            # self.medstd[x] = math.sqrt(np.sum(np.power([x - self.median[x] for (_, x) in ratings], 2)))
            self.medstd[x] = math.sqrt(np.sum(np.power([x - self.medianvalue for (_, x) in ratings], 2)))

        pbar = tqdm(total=self.trainset.n_users * self.trainset.n_users)
        for useri, ratesi in self.trainset.ur.items():
            if useri not in self.testusers:
                pbar.update(trainset.n_users)
                continue
            for userj, ratesj in self.trainset.ur.items():
                commonsum = 0
                for itemi, ratei in ratesi:
                    for itemj, ratej in ratesj:
                        # print(itemi, itemj)
                        # case 1: TODO: change mode here!!!!
                        loc = (ratei - self.means[useri]) * (ratej - self.means[userj]) / ((self.std[useri]) * (self.std[userj])) if ((self.std[useri]) * (self.std[userj])) != 0 else 0
                        # case 2:
                        # loc = (ratei - self.median[useri]) * (ratej - self.median[userj]) / ((self.medstd[useri]) * (self.medstd[userj]))
                        # loc = (ratei - self.medianvalue) * (ratej - self.medianvalue) / ((self.medstd[useri]) * (self.medstd[userj]))
                        mySimilarity2[useri, userj] += mySimilarity[itemi, itemj] * loc
                        if itemi == itemj:
                            commonsum += 1
                itemsum = len(ratesi) + len(ratesj) - commonsum
                if itemsum == 0:
                    jaccardsimi = 0
                else:
                    jaccardsimi = commonsum / itemsum

                mySimilarity3[useri, userj] = jaccardsimi
                pbar.update(1)
        pbar.close()
        self.sim = (mySimilarity2 - mySimilarity2.min()) / (mySimilarity2.max() - mySimilarity2.min()) + mySimilarity3
        # print(1)
        # print(mySimilarity)
        print(self.sim)
        print('Done computing user similarity matrix.')
        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):  # both know
            raise PredictionImpossible('User and/or item is unknown.')

        x, y = self.switch(u, i)

        neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[1])

        est = self.means[x]

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (nb, sim, r) in k_neighbors:
            # if sim > 0:
            sum_sim += math.fabs(sim)
            sum_ratings += sim * (r - self.means[nb])
            actual_k += 1

        if actual_k < self.min_k:
            sum_ratings = 0

        try:
            est += sum_ratings / sum_sim
        except ZeroDivisionError:
            pass  # return mean

        details = {'actual_k': actual_k}
        # print(details)
        return est, details

    def nspCalc(self):
        sp = 0
        total = 0

        for user in self.testusers:
            for item in self.trainset.ir.keys():
                total += 1
                x, y = user, item
                neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y]]
                if not neighbors:  # empty list
                    continue
                k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[1])
                sum_sim = sum_ratings = actual_k = 0
                for (nb, sim, r) in k_neighbors:
                    if sim > 0:
                        sum_sim += sim
                        sum_ratings += sim * (r - self.means[nb])
                        actual_k += 1

                if actual_k < self.min_k:
                    continue
                if sum_sim == 0:
                    continue
                sp += 1
        return sp/total

    def test2(self, testset, verbose=False):
        # The ratings are translated back to their original scale.
        predictions = [self.predict(uid,
                                    iid,
                                    r_ui_trans,
                                    clip=False,
                                    verbose=verbose)
                       for (uid, iid, r_ui_trans) in testset]
        return predictions


if __name__ == '__main__':
    # from cdsds import CalMetric
    # # mae, rmse, p, rec, f, npp, nsp = CalMetric().cvcalculate(BCF, 5)
    # mae, rmse, p, rec, f, npp, nsp = CalMetric().nocvcalculate(BCF)
    # print('#'*100)
    # print('This is BCF model')
    # print('#'*100)
    # print('mae = ' + str(mae))
    # print('rmse = ' + str(rmse))
    # print('pre = ' + str(p))
    # print('rec = ' + str(rec))
    # print('f1 = ' + str(f))
    # print('npp = ' + str(npp))
    # print('nsp = ' + str(nsp))

    resultsDict = {}
    nppnsp = {}
    neighbours = [20, 40, 60, 80, 100, 120]
    neighbours2 = [40, 80, 120, 160, 200, 240]
    from cdsds import CalMetric
    resultsDict['mae'], resultsDict['rmse'], resultsDict['pre'], resultsDict['rec'], resultsDict['f1'], nppnsp['npp'], nppnsp['nsp'] = CalMetric().Curvecvcalculate(BCF, fold=5, neighbours=neighbours2)
    print('#'*100)
    print('This is BCF model')
    print('#'*100)
    for key, val in resultsDict.items():
        print(key, val)
    print('Saving dictionary to memory......')
    np.save('./bcf.npy', resultsDict)
    np.save('./bcf2.npy', nppnsp)
    print('Saving dictionary to memory successfully!')
    print('#'*100)
    print('This is BCF model')
    print('#'*100)
