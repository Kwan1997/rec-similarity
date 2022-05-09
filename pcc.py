from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import prediction_algorithms
from collections import defaultdict
from surprise import similarities
import numpy as np
from surprise import PredictionImpossible
from six import iteritems
from surprise import AlgoBase
import heapq
from scipy import spatial
from statistics import median
from scipy.stats import entropy
from statistics import stdev
from surprise.model_selection import train_test_split
from scipy.stats import wasserstein_distance
from scipy.stats import dirichlet
import collections
from tqdm import tqdm
import math
from cdsds import CalMetric
from surprise import accuracy
# from surprise.prediction_algorithms.predictions import Prediction
from collections import OrderedDict
from surprise import KNNWithMeans



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


class PCC(mySymmetricAlgo):
    def __init__(self, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):

        mySymmetricAlgo.__init__(self, sim_options=sim_options,
                               verbose=verbose, **kwargs)

        self.k = k
        self.min_k = min_k
        self.testusers = []
        self.trust_neighbours = defaultdict(dict)

    def fit(self, trainset):

        mySymmetricAlgo.fit(self, trainset)
        # self.sim = self.compute_similarities()  # TODO:change codes here
        # self.sim = np.identity(self.sim.shape[0])

        print('Computing user similarity matrix...')
        mySimilarity = np.zeros((self.trainset.n_users, self.trainset.n_users), dtype=np.double)
        self.means = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            self.means[x] = np.mean([r for (_, r) in ratings])
        # print(self.means)

        self.itemmeans = np.zeros(self.trainset.n_items)
        for x, ratings in iteritems(self.trainset.ir):
            self.itemmeans[x] = np.mean([r for (_, r) in ratings])

        # print(self.itemmeans)

        self.median = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            self.median[x] = median([r for (_, r) in ratings])

        self.std = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            self.std[x] = np.std([r for (_, r) in ratings])

        self.medianvalue = (trainset.rating_scale[1] + trainset.rating_scale[0]) / 2.0
        self.maxminesmin = (trainset.rating_scale[1] - trainset.rating_scale[0]) * 1.0

        self.medstd = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            # self.medstd[x] = math.sqrt(sum(pow(x - self.median[x], 2) for (_, x) in ratings) / len(ratings))
            # self.medstd[x] = math.sqrt(np.sum(np.power([x - self.median[x] for (_, x) in ratings], 2)))
            self.medstd[x] = math.sqrt(np.sum(np.power([x - self.medianvalue for (_, x) in ratings], 2)))

        pbar = tqdm(total=self.trainset.n_users * self.trainset.n_users)
        for useri, ratesi in self.trainset.ur.items():
            if useri not in self.testusers:
                pbar.update(trainset.n_users)
                continue
            for userj, ratesj in self.trainset.ur.items():
                # if mySimilarity[userj, useri] != 0:
                #     mySimilarity[useri, userj] = mySimilarity[userj, useri]
                #     continue
                a = b = c = 0
                for itemi, ratei in ratesi:
                    for itemj, ratej in ratesj:
                        if itemi == itemj:
                            a += (ratei - self.means[useri]) * (ratej - self.means[userj])
                            b += (ratei - self.means[useri]) * (ratei - self.means[useri])
                            c += (ratej - self.means[userj]) * (ratej - self.means[userj])
                mySimilarity[useri, userj] = a/(math.sqrt(b*c)) if b*c != 0 else 0
                # print(PIP)
                pbar.update(1)
        pbar.close()
        self.sim = mySimilarity
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
            if sim > 0:
                sum_sim += sim
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
        return sp / total

    def test(self, testset, verbose=False):
        # The ratings are translated back to their original scale.
        predictions = []
        pbar = tqdm(total=len(testset))
        for (uid, iid, r_ui_trans) in testset:
            predictions.append(self.predict(uid, iid, r_ui_trans, verbose=verbose))
            pbar.update(1)
        pbar.close()
        return predictions


    def test2(self, testset, verbose=False):
        # The ratings are translated back to their original scale.
        predictions = [self.predict(uid,
                                    iid,
                                    r_ui_trans,
                                    clip=False,
                                    verbose=verbose)
                       for (uid, iid, r_ui_trans) in testset]
        return predictions



class CPCC(mySymmetricAlgo):
    def __init__(self, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):

        mySymmetricAlgo.__init__(self, sim_options=sim_options,
                               verbose=verbose, **kwargs)

        self.k = k
        self.min_k = min_k
        self.testusers = []

    def fit(self, trainset):

        mySymmetricAlgo.fit(self, trainset)
        # self.sim = self.compute_similarities()  # TODO:change codes here
        # self.sim = np.identity(self.sim.shape[0])

        print('Computing user similarity matrix...')
        mySimilarity = np.zeros((self.trainset.n_users, self.trainset.n_users), dtype=np.double)
        self.means = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            self.means[x] = np.mean([r for (_, r) in ratings])
        # print(self.means)

        self.itemmeans = np.zeros(self.trainset.n_items)
        for x, ratings in iteritems(self.trainset.ir):
            self.itemmeans[x] = np.mean([r for (_, r) in ratings])

        # print(self.itemmeans)

        self.median = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            self.median[x] = median([r for (_, r) in ratings])

        self.std = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            self.std[x] = np.std([r for (_, r) in ratings])

        self.medianvalue = (trainset.rating_scale[1] + trainset.rating_scale[0]) / 2.0
        self.maxminesmin = (trainset.rating_scale[1] - trainset.rating_scale[0]) * 1.0

        self.medstd = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            # self.medstd[x] = math.sqrt(sum(pow(x - self.median[x], 2) for (_, x) in ratings) / len(ratings))
            # self.medstd[x] = math.sqrt(np.sum(np.power([x - self.median[x] for (_, x) in ratings], 2)))
            self.medstd[x] = math.sqrt(np.sum(np.power([x - self.medianvalue for (_, x) in ratings], 2)))

        pbar = tqdm(total=self.trainset.n_users * self.trainset.n_users)
        for useri, ratesi in self.trainset.ur.items():
            if useri not in self.testusers:
                pbar.update(trainset.n_users)
                continue
            for userj, ratesj in self.trainset.ur.items():
                # if mySimilarity[userj, useri] != 0:
                #     mySimilarity[useri, userj] = mySimilarity[userj, useri]
                #     continue
                a = b = c = 0
                for itemi, ratei in ratesi:
                    for itemj, ratej in ratesj:
                        if itemi == itemj:
                            a += (ratei - self.medianvalue) * (ratej - self.medianvalue)
                            b += (ratei - self.medianvalue) * (ratei - self.medianvalue)
                            c += (ratej - self.medianvalue) * (ratej - self.medianvalue)
                            # a += (ratei - self.median[useri]) * (ratej - self.median[userj])
                            # b += (ratei - self.median[useri]) * (ratei - self.median[useri])
                            # c += (ratej - self.median[userj]) * (ratej - self.median[userj])
                mySimilarity[useri, userj] = a/(math.sqrt(b*c)) if b*c != 0 else 0
                # print(PIP)
                pbar.update(1)
        pbar.close()
        self.sim = mySimilarity
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
            if sim > 0:
                sum_sim += sim
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
        return sp / total

    def test2(self, testset, verbose=False):
        # The ratings are translated back to their original scale.
        predictions = [self.predict(uid,
                                    iid,
                                    r_ui_trans,
                                    clip=False,
                                    verbose=verbose)
                       for (uid, iid, r_ui_trans) in testset]
        return predictions


class JMSD(mySymmetricAlgo):
    def __init__(self, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):

        mySymmetricAlgo.__init__(self, sim_options=sim_options,
                               verbose=verbose, **kwargs)

        self.k = k
        self.min_k = min_k
        self.testusers = []

    def fit(self, trainset):

        mySymmetricAlgo.fit(self, trainset)
        # self.sim = self.compute_similarities()  # TODO:change codes here
        # self.sim = np.identity(self.sim.shape[0])

        print('Computing user similarity matrix...')
        mySimilarity = np.zeros((self.trainset.n_users, self.trainset.n_users), dtype=np.double)
        self.means = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            self.means[x] = np.mean([r for (_, r) in ratings])
        # print(self.means)

        self.itemmeans = np.zeros(self.trainset.n_items)
        for x, ratings in iteritems(self.trainset.ir):
            self.itemmeans[x] = np.mean([r for (_, r) in ratings])

        # print(self.itemmeans)

        self.median = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            self.median[x] = median([r for (_, r) in ratings])

        self.std = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            self.std[x] = np.std([r for (_, r) in ratings])

        self.medianvalue = (trainset.rating_scale[1] + trainset.rating_scale[0]) / 2.0
        self.maxminesmin = (trainset.rating_scale[1] - trainset.rating_scale[0]) * 1.0

        self.medstd = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            # self.medstd[x] = math.sqrt(sum(pow(x - self.median[x], 2) for (_, x) in ratings) / len(ratings))
            # self.medstd[x] = math.sqrt(np.sum(np.power([x - self.median[x] for (_, x) in ratings], 2)))
            self.medstd[x] = math.sqrt(np.sum(np.power([x - self.medianvalue for (_, x) in ratings], 2)))

        pbar = tqdm(total=self.trainset.n_users * self.trainset.n_users)
        for useri, ratesi in self.trainset.ur.items():
            if useri not in self.testusers:
                pbar.update(trainset.n_users)
                continue
            for userj, ratesj in self.trainset.ur.items():
                # if mySimilarity[userj, useri] != 0:
                #     mySimilarity[useri, userj] = mySimilarity[userj, useri]
                #     continue
                a = b = 0
                for itemi, ratei in ratesi:
                    for itemj, ratej in ratesj:
                        if itemi == itemj:
                            a += 1
                            b += (ratei - ratej) ** 2
                jaccard = a / (len(ratesi) + len(ratesj) - a) if (len(ratesi) + len(ratesj) - a) != 0 else 0
                msd = 1 - b/(self.trainset.rating_scale[1]**2*a) if (self.trainset.rating_scale[1]**2*a) != 0 else 0
                mySimilarity[useri, userj] = jaccard * msd
                # print(PIP)
                pbar.update(1)
        pbar.close()
        self.sim = mySimilarity
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
            if sim > 0:
                sum_sim += sim
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
        return sp / total

    def test2(self, testset, verbose=False):
        # The ratings are translated back to their original scale.
        predictions = [self.predict(uid,
                                    iid,
                                    r_ui_trans,
                                    clip=False,
                                    verbose=verbose)
                       for (uid, iid, r_ui_trans) in testset]
        return predictions


class ACPCC(mySymmetricAlgo):
    def __init__(self, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):

        mySymmetricAlgo.__init__(self, sim_options=sim_options,
                               verbose=verbose, **kwargs)

        self.k = k
        self.min_k = min_k
        self.testusers = []

    def fit(self, trainset):

        mySymmetricAlgo.fit(self, trainset)
        # self.sim = self.compute_similarities()  # TODO:change codes here
        # self.sim = np.identity(self.sim.shape[0])

        print('Computing user similarity matrix...')
        mySimilarity = np.zeros((self.trainset.n_users, self.trainset.n_users), dtype=np.double)
        self.means = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            self.means[x] = np.mean([r for (_, r) in ratings])
        # print(self.means)

        self.itemmeans = np.zeros(self.trainset.n_items)
        for x, ratings in iteritems(self.trainset.ir):
            self.itemmeans[x] = np.mean([r for (_, r) in ratings])

        # print(self.itemmeans)

        self.median = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            self.median[x] = median([r for (_, r) in ratings])

        self.std = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            self.std[x] = np.std([r for (_, r) in ratings])

        self.medianvalue = (trainset.rating_scale[1] + trainset.rating_scale[0]) / 2.0
        self.maxminesmin = (trainset.rating_scale[1] - trainset.rating_scale[0]) * 1.0

        self.medstd = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            # self.medstd[x] = math.sqrt(sum(pow(x - self.median[x], 2) for (_, x) in ratings) / len(ratings))
            # self.medstd[x] = math.sqrt(np.sum(np.power([x - self.median[x] for (_, x) in ratings], 2)))
            self.medstd[x] = math.sqrt(np.sum(np.power([x - self.medianvalue for (_, x) in ratings], 2)))

        pbar = tqdm(total=self.trainset.n_users * self.trainset.n_users)
        for useri, ratesi in self.trainset.ur.items():
            if useri not in self.testusers:
                pbar.update(trainset.n_users)
                continue
            for userj, ratesj in self.trainset.ur.items():
                # if mySimilarity[userj, useri] != 0:
                #     mySimilarity[useri, userj] = mySimilarity[userj, useri]
                #     continue
                a = b = c = commonitems = 0
                # iRatings = np.array([0, 0, 0, 0, 0], dtype=np.float64)
                # jRatings = np.array([0, 0, 0, 0, 0], dtype=np.float64)
                iRatings2 = np.array([0, 0, 0, 0, 0], dtype=np.float64)
                jRatings2 = np.array([0, 0, 0, 0, 0], dtype=np.float64)
                for itemi, ratei in ratesi:
                    iRatings2[int(ratei - 1)] += 1
                    for itemj, ratej in ratesj:
                        jRatings2[int(ratej - 1)] += 1
                        if itemi == itemj:
                            commonitems += 1
                            a += (ratei - self.means[useri]) * (ratej - self.means[userj])
                            b += (ratei - self.means[useri]) * (ratei - self.means[useri])
                            c += (ratej - self.means[userj]) * (ratej - self.means[userj])
                            # iRatings[int(ratei-1)] += 1
                            # jRatings[int(ratej - 1)] += 1
                jRatings2 /= len(ratesi)
                pccvalue = a/(math.sqrt(b*c)) if b*c != 0 else 0
                # bigA = np.dot(iRatings, jRatings) / (np.linalg.norm(iRatings) * np.linalg.norm(jRatings)) if (np.linalg.norm(iRatings) * np.linalg.norm(jRatings)) != 0 else 0
                # bigA = 1 - spatial.distance.cosine(iRatings, jRatings)
                bigA = np.dot(iRatings2, jRatings2) / (np.linalg.norm(iRatings2) * np.linalg.norm(jRatings2)) if (np.linalg.norm(iRatings2) * np.linalg.norm(jRatings2)) != 0 else 0
                bigC = 1 - np.exp(-commonitems/len(ratesi))
                mySimilarity[useri, userj] = pccvalue * bigA * bigC
                # print(PIP)
                pbar.update(1)
        pbar.close()
        self.sim = mySimilarity
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
            if sim > 0:
                sum_sim += sim
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
        return sp / total

    def test2(self, testset, verbose=False):
        # The ratings are translated back to their original scale.
        predictions = [self.predict(uid,
                                    iid,
                                    r_ui_trans,
                                    clip=False,
                                    verbose=verbose)
                       for (uid, iid, r_ui_trans) in testset]
        return predictions


class Cosine(mySymmetricAlgo):
    def __init__(self, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):

        mySymmetricAlgo.__init__(self, sim_options=sim_options,
                                 verbose=verbose, **kwargs)

        self.k = k
        self.min_k = min_k
        self.testusers = []

    def fit(self, trainset):

        mySymmetricAlgo.fit(self, trainset)
        # self.sim = self.compute_similarities()  # TODO:change codes here
        # self.sim = np.identity(self.sim.shape[0])

        print('Computing user similarity matrix...')
        mySimilarity = np.zeros((self.trainset.n_users, self.trainset.n_users), dtype=np.double)
        self.means = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            self.means[x] = np.mean([r for (_, r) in ratings])
        # print(self.means)

        self.itemmeans = np.zeros(self.trainset.n_items)
        for x, ratings in iteritems(self.trainset.ir):
            self.itemmeans[x] = np.mean([r for (_, r) in ratings])

        # print(self.itemmeans)

        self.median = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            self.median[x] = median([r for (_, r) in ratings])

        self.std = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            self.std[x] = np.std([r for (_, r) in ratings])

        self.medianvalue = (trainset.rating_scale[1] + trainset.rating_scale[0]) / 2.0
        self.maxminesmin = (trainset.rating_scale[1] - trainset.rating_scale[0]) * 1.0

        self.medstd = np.zeros(self.trainset.n_users)
        for x, ratings in iteritems(self.trainset.ur):
            # self.medstd[x] = math.sqrt(sum(pow(x - self.median[x], 2) for (_, x) in ratings) / len(ratings))
            # self.medstd[x] = math.sqrt(np.sum(np.power([x - self.median[x] for (_, x) in ratings], 2)))
            self.medstd[x] = math.sqrt(np.sum(np.power([x - self.medianvalue for (_, x) in ratings], 2)))

        pbar = tqdm(total=self.trainset.n_users * self.trainset.n_users)
        for useri, ratesi in self.trainset.ur.items():
            if useri not in self.testusers:
                pbar.update(trainset.n_users)
                continue
            for userj, ratesj in self.trainset.ur.items():
                # if mySimilarity[userj, useri] != 0:
                #     mySimilarity[useri, userj] = mySimilarity[userj, useri]
                #     continue
                a = b = c = 0
                for itemi, ratei in ratesi:
                    for itemj, ratej in ratesj:
                        if itemi == itemj:
                            a += ratei ** 2
                            b += ratej ** 2
                            c += ratei * ratej
                mySimilarity[useri, userj] = c / (math.sqrt(a) * math.sqrt(b)) if (math.sqrt(a) * math.sqrt(b)) != 0 else 0
                # print(PIP)
                pbar.update(1)
        pbar.close()
        self.sim = mySimilarity
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
            if sim > 0:
                sum_sim += sim
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
        return sp / total

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
    # resultsDict = {}
    # nppnsp = {}
    # neighbours = [20, 40, 60, 80, 100, 120]
    # neighbours2 = [40, 80, 120, 160, 200, 240]
    # from cdsds import CalMetric
    # resultsDict['mae'], resultsDict['rmse'], resultsDict['pre'], resultsDict['rec'], resultsDict['f1'], nppnsp['npp'], nppnsp['nsp'] = CalMetric().Curvecvcalculate(PCC, fold=5, neighbours=neighbours2)
    # print('#'*100)
    # print('This is PCC model')
    # print('#'*100)
    # for key, val in resultsDict.items():
    #     print(key, val)
    # print('Saving dictionary to memory......')
    # np.save('./pcc.npy', resultsDict)
    # np.save('./pcc2.npy', nppnsp)
    # print('Saving dictionary to memory successfully!')
    # print('#'*100)
    # print('This is PCC model')
    # print('#'*100)
    #
    #
    # resultsDict = {}
    # nppnsp = {}
    # neighbours = [20, 40, 60, 80, 100, 120]
    # neighbours2 = [40, 80, 120, 160, 200, 240]
    # from cdsds import CalMetric
    # resultsDict['mae'], resultsDict['rmse'], resultsDict['pre'], resultsDict['rec'], resultsDict['f1'], nppnsp['npp'], nppnsp['nsp'] = CalMetric().Curvecvcalculate(CPCC, fold=5, neighbours=neighbours2)
    # print('#'*100)
    # print('This is CPCC model')
    # print('#'*100)
    # for key, val in resultsDict.items():
    #     print(key, val)
    # print('Saving dictionary to memory......')
    # np.save('./cpcc.npy', resultsDict)
    # np.save('./cpcc2.npy', nppnsp)
    # print('Saving dictionary to memory successfully!')
    # print('#'*100)
    # print('This is CPCC model')
    # print('#'*100)
    #
    #
    # resultsDict = {}
    # nppnsp = {}
    # neighbours = [20, 40, 60, 80, 100, 120]
    # neighbours2 = [40, 80, 120, 160, 200, 240]
    # from cdsds import CalMetric
    # resultsDict['mae'], resultsDict['rmse'], resultsDict['pre'], resultsDict['rec'], resultsDict['f1'], nppnsp['npp'], nppnsp['nsp'] = CalMetric().Curvecvcalculate(JMSD, fold=5, neighbours=neighbours2)
    # print('#'*100)
    # print('This is JMSD model')
    # print('#'*100)
    # for key, val in resultsDict.items():
    #     print(key, val)
    # print('Saving dictionary to memory......')
    # np.save('./jmsd.npy', resultsDict)
    # np.save('./jmsd2.npy', nppnsp)
    # print('Saving dictionary to memory successfully!')
    # print('#'*100)
    # print('This is JMSD model')
    # print('#'*100)
    #
    #
    # resultsDict = {}
    # nppnsp = {}
    # neighbours = [20, 40, 60, 80, 100, 120]
    # neighbours2 = [40, 80, 120, 160, 200, 240]
    # from cdsds import CalMetric
    # resultsDict['mae'], resultsDict['rmse'], resultsDict['pre'], resultsDict['rec'], resultsDict['f1'], nppnsp['npp'], nppnsp['nsp'] = CalMetric().Curvecvcalculate(Cosine, fold=5, neighbours=neighbours2)
    # print('#'*100)
    # print('This is cosine model')
    # print('#'*100)
    # for key, val in resultsDict.items():
    #     print(key, val)
    # print('Saving dictionary to memory......')
    # np.save('./cosine.npy', resultsDict)
    # np.save('./cosine2.npy', nppnsp)
    # print('Saving dictionary to memory successfully!')
    # print('#'*100)
    # print('This is cosine model')
    # print('#'*100)
    #
    # resultsDict = {}
    # nppnsp = {}
    # neighbours = [20, 40, 60, 80, 100, 120]
    # neighbours2 = [40, 80, 120, 160, 200, 240]
    # from cdsds import CalMetric
    # resultsDict['mae'], resultsDict['rmse'], resultsDict['pre'], resultsDict['rec'], resultsDict['f1'], nppnsp['npp'], nppnsp['nsp'] = CalMetric().Curvecvcalculate(ACPCC, fold=5, neighbours=neighbours2)
    # print('#'*100)
    # print('This is ACPCC model')
    # print('#'*100)
    # for key, val in resultsDict.items():
    #     print(key, val)
    # print('Saving dictionary to memory......')
    # np.save('./acpcc.npy', resultsDict)
    # np.save('./acpcc2.npy', nppnsp)
    # print('Saving dictionary to memory successfully!')
    # print('#'*100)
    # print('This is ACPCC model')
    # print('#'*100)



    # from cdsds import CalMetric
    # mae, rmse, p, rec, f, npp, nsp = CalMetric().cvcalculate(PCC, 5)
    # print('#' * 100)
    # print('This is pcc model')
    # print('#' * 100)
    # print('mae = ' + str(mae))
    # print('rmse = ' + str(rmse))
    # print('pre = ' + str(p))
    # print('rec = ' + str(rec))
    # print('f1 = ' + str(f))
    # print('npp = ' + str(npp))
    # print('nsp = ' + str(nsp))
    #
    # mae, rmse, p, rec, f, npp, nsp = CalMetric().cvcalculate(CPCC, 5)
    # print('#' * 100)
    # print('This is cpcc model')
    # print('#' * 100)
    # print('mae = ' + str(mae))
    # print('rmse = ' + str(rmse))
    # print('pre = ' + str(p))
    # print('rec = ' + str(rec))
    # print('f1 = ' + str(f))
    # print('npp = ' + str(npp))
    # print('nsp = ' + str(nsp))
    #
    # mae, rmse, p, rec, f, npp, nsp = CalMetric().cvcalculate(JMSD, 5)
    # print('#' * 100)
    # print('This is jmsd model')
    # print('#' * 100)
    # print('mae = ' + str(mae))
    # print('rmse = ' + str(rmse))
    # print('pre = ' + str(p))
    # print('rec = ' + str(rec))
    # print('f1 = ' + str(f))
    # print('npp = ' + str(npp))
    # print('nsp = ' + str(nsp))

    # mae, rmse, p, rec, f, npp, nsp = CalMetric().cvcalculate(Cosine, 5)
    # print('#' * 100)
    # print('This is cosine model')
    # print('#' * 100)
    # print('mae = ' + str(mae))
    # print('rmse = ' + str(rmse))
    # print('pre = ' + str(p))
    # print('rec = ' + str(rec))
    # print('f1 = ' + str(f))
    # print('npp = ' + str(npp))
    # print('nsp = ' + str(nsp))


    # mae, rmse, p, rec, f, npp, nsp = CalMetric().cvcalculate(ACPCC, 5)
    # mae, rmse, p, rec, f, npp, nsp = CalMetric().nocvcalculate(ACPCC)
    print('#' * 100)
    # print('This is cosine model')
    # print('#' * 100)
    # print('mae = ' + str(mae))
    # print('rmse = ' + str(rmse))
    # print('pre = ' + str(p))
    # print('rec = ' + str(rec))
    # print('f1 = ' + str(f))
    # print('npp = ' + str(npp))
    # print('nsp = ' + str(nsp))


