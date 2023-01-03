
class Particle:
    def __init__(self, att, bel, nash: int):
        self.att = att
        self.bel = bel
        self.nash_method = nash

    def attitude(self):
        return self.att

    def set_att(self, att):
        self.att = att

    def belief(self):
        return self.bel

    def set_bel(self, bel):
        self.bel = bel

    def get_nash(self):
        return self.nash_method

    def set_nash(self, nash: int):
        self.nash = nash

    def __str__(self):
        return "Particle(att=" + str(self.att) + ", bel=" + str(self.bel) + ", label=" + str(self.nash_method) + ")"