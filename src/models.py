

class MagnetModel(object):
    
    def __init__(self, pos_x, pos_y, magnetic_strength):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.magnetic_strength = magnetic_strength


class PendulumModel(object):
    
    def __init__(self, pos_x0, pos_y0, vel_x0, vel_y0, friction_constant):
        self.pos_x0 = pos_x0
        self.pos_y0 = pos_y0
        self.vel_x0 = vel_x0
        self.vel_y0 = vel_y0
        self.magnets = []
        
        self.friction_constant = friction_constant
        