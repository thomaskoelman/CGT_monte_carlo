from Game import Game
from Particle import Particle
import numpy as np
import math
from LookupTable import LookupTable

class MonteCarlo:
    def __init__(self, prob_func, particles: list, lookup: LookupTable):
        self.particles = particles
        self.prob_func = prob_func
        self.lookup_table = lookup


    def update(self, game: Game, opp_move: int, err_lvls, f_ab, f_nash):
        label = int(self.estimate_opponents_method())
        err_distr, err = self.__update_errors(game, label, opp_move, err_lvls)
        new_particles = self.__resample_particles(game, opp_move)
        self.__perturb_particles(new_particles, err, f_ab, f_nash, game)
        return MonteCarlo(err_distr, new_particles, self.lookup_table)


    def __update_errors(self, game: Game, label: int, opp_move: int, err_lvls: list):
        att_opp = self.estimate_att()
        bel_opp = self.estimate_bel()
        att_agent = bel_opp
        modded_game = game.modify(att_agent, att_opp)
        nash_agent, nash_opp = modded_game.lemke_howson(label)
        j = nash_opp[opp_move]
        k = self.coop()
        err_distr = dict()
        for lvl in err_lvls:
            err_distr[str(lvl)] = self.prob_func(lvl)  * self.lookup_table.lookup(j, k, lvl)
        normalize_dictionary(err_distr)
        def probability(error: float):
            return err_distr[str(error)]
        err = est_error(probability, err_lvls)
        return probability, err

    def __resample_particles(self, game: Game, opp_move) -> list:
        weights = [game.modify(p.attitude(), p.belief()).lemke_howson(p.get_nash())[1][opp_move] for p in self.particles]
        normalize_list(weights)
        new_particles = []
        for _ in range(len(self.particles)):
            particle = self.particles[pick_from_distribution(weights)]
            new_particle = Particle(particle.attitude(), particle.belief(), particle.get_nash())
            new_particles.append(new_particle)
        return new_particles

    def __perturb_particles(self, new_particles, err, f_ab: float, f_nash: float, g: Game) -> None:
        for particle in new_particles:
            particle.set_att(np.random.normal(particle.attitude(), err * f_ab))
            particle.set_bel(np.random.normal(particle.belief(), err * f_ab))
        if np.random.random() < (err * f_nash):
            shape = g.shape()
            labels = range(sum(shape))
            for particle in new_particles:
                label = np.random.choice(labels)
                particle.set_nash(label)


    def estimate_att(self):
        return sum([p.attitude() for p in self.particles])/len(self.particles)

    def estimate_bel(self):
        return sum([p.belief() for p in self.particles]) / len(self.particles)

    def estimate_opponents_method(self):
        counts = dict()
        for p in self.particles:
            nash = p.get_nash()
            if str(nash) in counts:
                counts[nash] += 1
            else:
                counts[nash] = 1
        keys = list(counts.keys())
        values = list(counts.values())
        max_id = np.argmax(values)
        return keys[max_id]

    def coop(self):
        att = self.estimate_att()
        bel = self.estimate_bel()
        return (att + bel) / (math.sqrt(att ** 2 + 1) * math.sqrt(bel ** 2 + 1))

def est_error(err_distr, err_lvls: list):
   error = 0
   for lvl in err_lvls:
       error += lvl * err_distr(lvl)
   return error

def normalize_dictionary(d: dict):
    print(d)
    if sum(d.values()) == 0:
        factor = 1
    else:
        factor = 1.0 / sum(d.values())
    print("factor: ", factor)
    for k in d:
        d[k] = d[k] * factor

def normalize_list(l):
    factor = 1 / sum(l)
    for i, e in enumerate(l):
        l[i] = e * factor

def pick_from_distribution(distribution: list) -> int:
    rand = np.random.random()
    prob = 0
    for action, p in enumerate(distribution):
        prob += p
        if rand < prob:
            return action
    return -1







