import numpy as np
from Game import Game
import math

# class LookupTable:
#     def __init__(self, j_buckets: list, k_buckets: list, err_lvls: list):
#         self.table = np.zeros((len(j_buckets), len(k_buckets), len(err_lvls)))
#         self.j_buckets = j_buckets
#         self.k_buckets = k_buckets
#         self.err_lvls = err_lvls
#
#         # for lvl in err_lvls:
#         #     for k in k_buckets:
#         #         for _ in range(200):
#         #             A, B = random_matrix(2)
#         #             game = Game(A, B)
#         #             real_att_1 = np.random.normal()
#         #             real_att_2 = np.random.normal()
#         #             modded_game = game.modify(real_att_1, real_att_2)
#
#
#
#         games = []
#         for _ in range(10000):
#             A, B = random_matrix(2)
#             g = Game(A, B)
#             games.append(g)
#
#         self.table = np.zeros((len(j_buckets), len(k_buckets), len(err_lvls)))
#
#         for game in games:
#             real_att = np.random.normal()
#             real_bel = np.random.normal()
#             atts = []
#             bels = []
#             for _ in range(200):
#                 atts.append(np.random.normal())
#                 bels.append(np.random.normal())
#             est_att = sum(atts)/len(atts)
#             est_bel = sum(bels)/len(bels)
#             nb_labels = sum(game.shape())
#             label = np.random.randint(low = 0, high = nb_labels)
#             k = coop(est_att, est_bel)
#             err = euclidian_dist(real_att, est_att, real_bel, est_bel)
#             modded_game = game.modify(real_att, real_bel)
#             _, ne_opp = modded_game.lemke_howson(label)
#             move =  np.random.choice(np.arange(len(ne_opp)), 1, p=ne_opp)
#             j = ne_opp[move]
#
#             j_id, k_id, err_id = self.assign_j_bucket(j), self.assign_k_bucket(k), self.assign_err_bucket(err)
#             self.table[j_id, k_id, err_id] += 1
#
#         s = self.table.sum(axis=2)
#         s[s == 0] = 1.0
#         self.table = self.table / s  #np.sum(self.table, axis=0)
#         print(self.table)
#         print("Lookup table ready!")

class LookupTable:
    def __init__(self, j_bins, k_bins, err_bins):
        self.__j_bins = j_bins
        self.__k_bins = k_bins
        self.__err_bins = err_bins
        self.table = np.ones((len(self.__j_bins),len(self.__k_bins),len(self.__err_bins)))
        for i in range(100000):
            self.add_game()
            if i%1000 == 0:
                print("progress: ", i/1000)


    def add_game(self):
        real_att = np.random.normal()
        real_bel = np.random.normal()

        est_att = np.average(np.random.normal(size=200))
        est_bel = np.average(np.random.normal(size=200))

        A, B = random_matrix(2)
        game = Game(A, B)
        nb_labels = sum(game.shape())
        label = np.random.randint(low = 0, high = nb_labels)
        mgame = game.modify(real_att, real_bel)
        _, ne_opp = mgame.lemke_howson(label)

        j =  np.random.choice(ne_opp, 1, p=ne_opp)[0]
        k = coop(est_att, est_bel)
        err = euclidian_dist(real_att, est_att, real_bel, est_bel)

        j_id, k_id, err_id = self.assign_j_bucket(j), self.assign_k_bucket(k), self.assign_err_bucket(err)

        #print("j=",j)
        #print("j-bucket: ", j_id)

        #print("k=",k)
        #print("k-bucket: ", k_id)

        #print("err=",err)
        #print("error bucket: ", err_id)

        self.__add(j_id, k_id, err_id)





    def t(self, j, k, l):
        return self.table[j, k, l]

    def __add(self, j, k, l):
        self.table[j, k, l] += 1

    def lookup(self, j, k, err_lvl):
        j_id = self.assign_j_bucket(j)
        k_id = self.assign_k_bucket(k)
        err_id = self.assign_err_bucket(err_lvl)
        return self.table[j_id, k_id, err_id]

    def assign_j_bucket(self, j: float):
        return np.digitize(j, self.__j_bins) - 1
        # j_buckets = np.array(self.j_buckets) - j
        # j_buckets[j_buckets <= 0.0] = 1.0
        # return np.argmin(j_buckets)

    def assign_k_bucket(self, k: float):
        return np.digitize(k, self.__k_bins) - 1
        # k_buckets = np.array(self.k_buckets) - k
        # k_buckets[k_buckets <= 0.0] = 1.0
        # return np.argmin(k_buckets)

    def assign_err_bucket(self, err: float):
        return np.digitize(err, self.__err_bins) - 1
        # err_buckets = np.array(self.err_lvls) - err
        # err_buckets[err_buckets <= 0.0] = 1.0
        # return np.argmin(err_buckets)



def get_buckets(start: int, stop: int, step_size):
    return np.arange(start=start, stop=stop, step=step_size)


def random_matrix(nb_moves):
    matrix_1 = np.array([[np.random.random() for _ in range(nb_moves)] for _ in range(nb_moves)])
    matrix_2 = np.array([[np.random.random() for _ in range(nb_moves)] for _ in range(nb_moves)])
    return matrix_1, matrix_2

def coop(att, bel):
   return (att + bel) / (math.sqrt(att ** 2 + 1) * math.sqrt(bel ** 2 + 1))

def euclidian_dist(true_att, est_att, true_bel, est_bel):
    return math.sqrt((true_att - est_att)**2 + (true_bel - est_bel)**2)

# tbl = LookupTable()
# for _ in range(10000):
#     tbl.add_game()
# print(tbl.table)

# j_buckets = get_buckets(0, 1, 0.1)
# print(j_buckets)
# k_buckets = get_buckets(-1, 1, 0.2)
# print(k_buckets)
# err_lvls = [.0, .001, .002, .004, .008, .016, .032, .064, .128, .256, .512, 1.0]
# t = LookupTable(j_buckets, k_buckets, err_lvls)

