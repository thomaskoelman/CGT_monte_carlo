import numpy as np
import quantecon.game_theory as gt

class Game:
    def __init__(self, agentMatrix: np.ndarray, oppMatrix: np.ndarray):
        agent_size = agentMatrix.shape
        opp_size = oppMatrix.shape
        if not (agent_size[0] == opp_size[1] and agent_size[1] == opp_size[0]):
            raise AssertionError("The input matrices have the wrong dimensions!")
        self.__agent = agentMatrix
        self.__opp = oppMatrix

    def __repr__(self):
        return "Game(" + str(self.__agent.tolist()) + ", " + str(self.__opp.tolist()) + ")"

    def __str__(self):
        return "payoffs player 1: \n" + str(self.__agent.tolist()) + "\n\npayoffs player 2: \n" +str(self.__opp.tolist()) + "\n"

    def agent_payoffs(self):
        return self.__agent

    def opp_payoffs(self):
        return self.__opp

    def get_payoff(self, i, j):
        try:
            return self.__agent[i, j]
        except IndexError:
            raise IndexError("i should be lower than " + str(self.shape()[0]) + " and j lower than " + str(self.shape()[1]))

    def get_payoff_opp(self, i, j):
        try:
            return self.__opp[i, j]
        except IndexError:
            raise IndexError("i should be lower than " + str(self.shape()[0]) + " and j lower than " + str(self.shape()[1]))

    def shape(self):
        return self.__agent.shape

    def switch_viewpoint(self):
        return Game(self.__opp, self.__agent)

    def modify(self, attAgent: float, attOpp: float):
        newAgent = self.__agent + attAgent * self.__opp.T
        newOpp = self.__opp + attOpp * self.__agent.T
        return Game(newAgent, newOpp)

    def lemke_howson(self, label=0):
        player_1 = gt.Player(self.__agent)
        player_2 = gt.Player(self.__opp)
        game = gt.NormalFormGame((player_1, player_2))
        return gt.lemke_howson(game, init_pivot=label)


    def __best_response_polytopes(self):
        n, m = self.shape()
        A = self.__agent
        B = self.__opp

        eye_A = np.identity(n)
        eye_B = np.identity(m)

        ones_A = np.full((n, 1), 1.0)
        ones_B = np.full((m, 1), 1.0)

        T_column = np.concatenate([eye_A, A, ones_A], 1)
        T_row = np.concatenate([B, eye_B, ones_B], 1)

        return T_column, T_row

    def __non_basic_variables(self, tableau):
        return None

a = np.array([[1,2], [3,4]])
b = np.array([[5,6], [7,8]])
g = Game(a, b)
h = g.switch_viewpoint()

