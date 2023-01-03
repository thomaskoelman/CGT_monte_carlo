import numpy as np
from Particle import Particle
from Game import Game
from Agent import Agent
import matplotlib.pyplot as plt
from LookupTable import LookupTable

def main():
    n = 200
    r = 0.1
    f_ab = 0.2
    f_nash = 0.1
    err_lvls = [.0, .001, .002, .004, .008, .016, .032, .064, .128, .256, .512, 1.0]
    nb_labels = sum((2, 2))
    j_buckets = get_buckets(0, 1, 0.1)
    k_buckets = get_buckets(-1, 1, 0.2)
    lookup = LookupTable(j_buckets, k_buckets, err_lvls)
    agent_1 = Agent(n, nb_labels, err_lvls, lookup)
    agent_2 = Agent(n, nb_labels, err_lvls, lookup)

    games = []
    for _ in range(1000):
        A, B = random_matrix(2)
        g = Game(A, B)
        games.append(g)

    rewards_1 = []
    rewards_2 = []
    cooperation_1 = []

    for game in games:
        agent_1.observe_game(game)
        agent_2.observe_game(game.switch_viewpoint())

        move_1 = agent_1.pick_move(r)
        move_2 = agent_2.pick_move(r)

        r_1 = agent_1.award(move_1, move_2)
        r_2 = agent_2.award(move_2, move_1)
        rewards_1.append(r_1)
        rewards_2.append(r_2)
        cooperation_1.append(agent_1.cooperation_lvl())
        print(agent_1.cooperation_lvl())

        agent_1.update_model(move_2, err_lvls, f_ab, f_nash)
        agent_2.update_model(move_1, err_lvls, f_ab, f_nash)

    fig, ax = plt.subplots(figsize=(8, 8))
    x_axis = list(range(0, 1000, 10))
    ax.plot(x_axis, cooperation_1[0:1000:10])
    plt.show()




def init_particles(nb_labels, n: int):
    labels = range(nb_labels)

    particles = []
    for _ in range(n):
        label = np.random.choice(labels)
        att = np.random.normal()
        bel = np.random.normal()
        p = Particle(att, bel, label)
        particles.append(p)
    return particles

def init_err_probability(err_lvls: list):
    def probability(error: float):
        return 1 / len(err_lvls)
    return probability

def random_matrix(nb_moves):
     matrix_1 = np.array([[np.random.random() for _ in range(nb_moves)] for _ in range(nb_moves)])
     matrix_2 = np.array([[np.random.random() for _ in range(nb_moves)] for _ in range(nb_moves)])
     return matrix_1, matrix_2

def get_buckets(start: int, stop: int, step_size):
    return np.arange(start=start, stop=stop, step=step_size)

main()