import numpy as np
from Game import Game
import math

class LookupTable:
    def __init__(self, j_buckets: list, k_buckets: list, err_lvls: list):
        self.j_buckets = j_buckets
        self.k_buckets = k_buckets
        self.err_lvls = err_lvls
        games = []
        for _ in range(10000):
            A, B = random_matrix(2)
            g = Game(A, B)
            games.append(g)

        self.table = np.zeros((len(j_buckets), len(k_buckets), len(err_lvls)))

        for game in games:
            real_att = np.random.normal()
            real_bel = np.random.normal()
            atts = []
            bels = []
            for _ in range(200):
                atts.append(np.random.normal())
                bels.append(np.random.normal())
            est_att = sum(atts)/len(atts)
            est_bel = sum(bels)/len(bels)
            nb_labels = sum(game.shape())
            label = np.random.randint(low = 0, high = nb_labels)
            k = coop(est_att, est_bel)
            err = euclidian_dist(real_att, est_att, real_bel, est_bel)
            modded_game = game.modify(real_att, real_bel)
            _, ne_opp = modded_game.lemke_howson(label)
            move =  np.random.choice(np.arange(len(ne_opp)), 1, p=ne_opp)
            j = ne_opp[move]

            j_id, k_id, err_id = self.assign_j_bucket(j), self.assign_k_bucket(k), self.assign_err_bucket(err)
            self.table[j_id, k_id, err_id] += 1
        self.table = self.table / np.sum(self.table)
        print("Lookup table ready!")





    def lookup(self, j, k, err_lvl):
        j_id = self.assign_j_bucket(j)
        k_id = self.assign_k_bucket(k)
        err_id = self.err_lvls.index(err_lvl)
        return self.table[j_id, k_id, err_id]

    def assign_j_bucket(self, j: float):
        j_buckets = np.array(self.j_buckets) - j
        j_buckets[j_buckets <= 0.0] = 1.0
        return np.argmin(j_buckets)

    def assign_k_bucket(self, k: float):
        k_buckets = np.array(self.k_buckets) - k
        k_buckets[k_buckets <= 0.0] = 1.0
        return np.argmin(k_buckets)

    def assign_err_bucket(self, err: float):
        err_buckets = np.array(self.err_lvls) - err
        err_buckets[err_buckets <= 0.0] = 1.0
        return np.argmin(err_buckets)



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

# j_buckets = get_buckets(0, 1, 0.1)
# print(j_buckets)
# k_buckets = get_buckets(-1, 1, 0.2)
# print(k_buckets)
# err_lvls = [.0, .001, .002, .004, .008, .016, .032, .064, .128, .256, .512, 1.0]
# t = LookupTable(j_buckets, k_buckets, err_lvls)

