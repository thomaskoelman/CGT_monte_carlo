from Game import Game
from MonteCarlo import MonteCarlo
import numpy as np
from Particle import Particle
from LookupTable import LookupTable

class Agent:
    def __init__(self, n: int, nb_labels, err_lvls, lookup: LookupTable):
        prob = init_err_probability(err_lvls)
        particles= init_particles(nb_labels, n)
        self.model = MonteCarlo(prob, particles, lookup)
        self.game = None
        self.cumreward = 0

    def observe_game(self, game: Game):
        self.game = game

    def get_current_game(self):
        return self.game

    def pick_move(self, r: float):
        att_opp = self.model.estimate_att()
        bel_opp = self.model.estimate_bel()
        label = int(self.model.estimate_opponents_method())

        att_agent = att_opp + r
        modded_game: Game = self.game.modify(att_agent, att_opp)
        ne_agent, ne_opp = modded_game.lemke_howson(label)
        move: int = pick_from_distribution(ne_agent)
        return move

    def update_model(self, opp_move: int, err_lvls: list, f_ab: float, f_nash: float):
        game = self.get_current_game()
        self.model: MonteCarlo = self.model.update(game, opp_move, err_lvls, f_ab, f_nash)

    def award(self, action_agent, action_opp):
        game: Game = self.get_current_game()
        reward = game.get_payoff(action_agent, action_opp)
        self.cumreward += reward
        return reward

    def cooperation_lvl(self):
        return self.model.coop()

    # def pick_move(self):
    #
    # def update_model(self, opp_move: int):
    #     self.model = self.model.take_step()

# model = Model(parameters)
# model_2 = model.take_step()
# model_3 = model_2.take_step()


#main()
    # generate 1000 games:
    # for each game:
    #     agent_1 = Agent(...)
    #     agent_2 = Agent(...)
    #     agent_1.observe_game(game)
    #     game2 = game.invert()
    #     agent_2.observe_game(game2)
    #     move1 = agent_1.move()
    #     move_2 = agent_2.move()
    #     agent_1.update_model(move_2)
    #     agent_2.update_model(move_1)

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

def pick_from_distribution(distribution: list) -> int:
    rand = np.random.random()
    prob = 0
    for action, p in enumerate(distribution):
        prob += p
        if rand < prob:
            return action
    return -1




def random_matrix(nb_moves):
    matrix_1 = np.array([[np.random.random() for _ in range(nb_moves)] for _ in range(nb_moves)])
    matrix_2 = np.array([[np.random.random() for _ in range(nb_moves)] for _ in range(nb_moves)])
    return matrix_1, matrix_2

# err_lvls = [.0, .001, .002, .004, .008, .016, .032, .064, .128, .256, .512, 1.0]
# a = Agent(200, 4, err_lvls)
# A, B = random_matrix(2)
# A = np.array([[1,2], [3,4]])
# B = np.array([[5,6], [7,8]])
# g = Game(A, B)
# a.observe_game(g)