import unittest
from src.models import MagnetModel, PendulumModel
from src.functions import CommonFunctions


class TestCommonFunctions(unittest.TestCase):
    
    def test_array_to_float_carray(self):
        array = [-0.1, 23.222, -45.2, 1.0]
        
        self.assertEqual(CommonFunctions.array_to_float_carray(array), "{-0.1f, 23.222f, -45.2f, 1.0f}")


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
    
    def setUp(self):
        magnet1 = MagnetModel(0.1, 0.6, 0.8)
        magnet2 = MagnetModel(1.1, -0.3, 1.8)
        magnet3 = MagnetModel(-0.1, 0.45, -0.4)
        
        self.magnets = [magnet1, magnet2,  magnet3]
        
        self.pos_x0 = 0.8
        self.pos_y0 = 0.3
        self.vel_x0 = 1.2
        self.vel_y0 = -0.1
        self.plane_distance = 0.25
        
        self.friction = 0.3
        self.gravity_pullback = 0.5
        
        self.pendulum = PendulumModel(self.pos_x0, self.pos_y0, self.vel_x0, self.vel_y0, self.friction, self.gravity_pullback, self.plane_distance)
        self.pendulum.magnets = self.magnets
    
    def test_pendulum_model_initalization(self):
        self.assertEqual(self.pendulum.pos_x0, self.pos_x0)
        self.assertEqual(self.pendulum.pos_y0, self.pos_y0)
        self.assertEqual(self.pendulum.vel_x0, self.vel_x0)
        self.assertEqual(self.pendulum.vel_y0, self.vel_y0)
        self.assertEqual(self.pendulum.friction, self.friction)
        self.assertEqual(self.pendulum.gravity_pullback, self.gravity_pullback)
        self.assertEqual(self.pendulum.plane_distance, self.plane_distance)
        self.assertEqual(len(self.pendulum.magnets), len(self.magnets))

    def test_prepare_gpu_source(self):
        self.pendulum.prepare_gpu_source()
        
        gpu_source = """
            __device__ inline void diff_eq(float t, float &nx, float &ny, float &nvx, float &nvy, float x, float y, float vx, float vy) { 
                
                float kf = 0.3f;
                float kg = 0.5f;
                float d2 = 0.25f * 0.25f;
                 
                int n = 3; 
      
                float x[n] = {0.1f, 1.1f, -0.1f};
                float y[n] = {0.6f, -0.3f, 0.45f};
                float km[n] = {0.8f, 1.8f, -0.4f};                
                
                float amx = 0.0f;
                float amy = 0.0f;
                             
                for (int i = 0 ; i < n ; i++) {
                    float deltaX = x[i] - x;
                    float deltaY = y[i] - y;
                    
                    float dist = Math.sqrt(deltaX * deltaX + deltaY * deltaY + d2);
                    float distPow3 = dist * dist *dist;
                
                    amx += km[i] * deltaX / distPow3;
                    amy += km[i] * deltaY / distPow3;
                }
                      
                nvx = -kf * vx - kg * x + amx;
                nvy = -kf * vy - kg * y + amy;
                
                nx = vx;
                ny = vy;
            }"""
        
        gpu_source = "".join(gpu_source.split())
        pendulum_gpu_source = "".join(self.pendulum.gpu_source.split())
 
        self.assertEqual(pendulum_gpu_source, gpu_source)



if __name__ == '__main__':
    unittest.main()
    
    
    
    
    
        