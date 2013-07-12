import unittest
from src.models import MagnetModel, PendulumModel


class TestMagnetModel(unittest.TestCase):
    
    def test_magnet_model_initalization(self):
        pos_x = 0.1
        pos_y = 0.6
        magnetic_strength = 0.8

        magnet = MagnetModel(pos_x, pos_y, magnetic_strength)
        
        self.assertEqual(magnet.pos_x, pos_x)
        self.assertEqual(magnet.pos_y, pos_y)
        self.assertEqual(magnet.magnetic_strength, magnetic_strength)


class TestPendulumModel(unittest.TestCase):
    
    def test_pendulum_model_initalization(self):
        magnet1 = MagnetModel(0.1, 0.6, 0.8)
        magnet2 = MagnetModel(1.1, -0.3, 1.8)
        magnet3 = MagnetModel(-0.1, 0.45, -0.4)
        
        magnets = [magnet1, magnet2,  magnet3]
        
        pos_x0 = 0.8
        pos_y0 = 0.3
        vel_x0 = 1.2
        vel_y0 = -0.1
        
        friction_constant = 0.3
        
        pendulum = PendulumModel(pos_x0, pos_y0, vel_x0, vel_y0, friction_constant)
        pendulum.magnets = magnets
        
        self.assertEqual(pendulum.pos_x0, pos_x0)
        self.assertEqual(pendulum.pos_y0, pos_y0)
        self.assertEqual(pendulum.vel_x0, vel_x0)
        self.assertEqual(pendulum.vel_y0, vel_y0)
        self.assertEqual(pendulum.friction_constant, friction_constant)
        self.assertEqual(len(pendulum.magnets), len(magnets))





if __name__ == '__main__':
    unittest.main()