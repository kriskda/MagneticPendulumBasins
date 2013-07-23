import unittest
import colorsys
import random

from src.models import MagnetModel, PendulumModel     
from src.functions import CommonFunctions        
from src.integrators import EulerIntegrator    
from src.graphics import BasicImageGenerator    
from src.basins import BasinsGenerator           
                        
 

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
        
        self.plane_distance = 0.25        
        self.friction = 0.3
        self.gravity_pullback = 0.5
        
        self.pendulum = PendulumModel(self.friction, self.gravity_pullback, self.plane_distance)
        self.pendulum.magnets = self.magnets
    
    def test_pendulum_model_initalization(self):
        self.assertEqual(self.pendulum.friction, self.friction)
        self.assertEqual(self.pendulum.gravity_pullback, self.gravity_pullback)
        self.assertEqual(self.pendulum.plane_distance, self.plane_distance)
        self.assertEqual(len(self.pendulum.magnets), len(self.magnets))

    def test_prepare_gpu_source(self):
        self.pendulum.prepare_gpu_source()
        
        gpu_source = """
            __device__ inline void diff_eq(float &nx, float &ny, float &nvx, float &nvy, float x, float y, float vx, float vy) { 
                
                const float kf = 0.3f;
                const float kg = 0.5f;
                const float d2 = 0.25f * 0.25f;
                 
                const int n = 3; 
      
                const float xm[n] = {0.1f, 1.1f, -0.1f};
                const float ym[n] = {0.6f, -0.3f, 0.45f};
                const float km[n] = {0.8f, 1.8f, -0.4f};                
                
                float amx = 0.0f;
                float amy = 0.0f;
                             
                for (int i = 0 ; i < n ; i++) {
                    float deltaX = xm[i] - x;
                    float deltaY = ym[i] - y;
                    
                    float dist = sqrtf(deltaX * deltaX + deltaY * deltaY + d2);
                    float distPow3 = dist * dist * dist;
                
                    amx += km[i] * deltaX / distPow3;
                    amy += km[i] * deltaY / distPow3;
                }
                      
                nvx = -kf * vx - kg * x + amx;
                nvy = -kf * vy - kg * y + amy;
                
                nx = vx;
                ny = vy;
            }
    
            __device__ int determineMagnet(float x, float y, float delta) {
                bool m0dx = ((0.1f - delta) <= x) && (x <= (0.1f + delta));
                bool m0dy = ((0.6f - delta) <= y) && (y <= (0.6f + delta));
   
                if (m0dx && m0dy) {
                    return 0;
                } 
            
                bool m1dx = ((1.1f - delta) <= x) && (x <= (1.1f + delta));
                bool m1dy = ((-0.3f - delta) <= y) && (y <= (-0.3f + delta));
   
                if (m1dx && m1dy) {
                    return 1;
                } 
            
                bool m2dx = ((-0.1f - delta) <= x) && (x <= (-0.1f + delta));
                bool m2dy = ((0.45f - delta) <= y) && (y <= (0.45f + delta));
   
                if (m2dx && m2dy) {
                    return 2;
                }  
                                           
                return -1; 
            }
            """
            
        gpu_source = "".join(gpu_source.split())
        pendulum_gpu_source = "".join(self.pendulum.gpu_source.split())
 
        self.assertEqual(pendulum_gpu_source, gpu_source)


class TestEulerIntegrator(unittest.TestCase):
    
    def setUp(self):
        self.time_step = 0.01
        self.integrator = EulerIntegrator(self.time_step)
    
    def test_euler_integrator_instance(self):
        self.assertEqual(self.integrator.__class__.__name__, "EulerIntegrator")
        self.assertEqual(self.integrator.time_step, self.time_step)
    
    def test_euler_integrator_gpu_source(self):
        gpu_source = """
            __device__ inline void calculateStep(float &x, float &y, float &vx, float &vy) {                
                float nx, ny, nvx, nvy;
                
                diff_eq(nx, ny, nvx, nvy, x, y, vx, vy);
        
                vx = vx + nvx * dt;
                vy = vy + nvy * dt;
                
                x = x + nx * dt;
                y = y + ny * dt;                
            }
        """
        
        gpu_source = "".join(gpu_source.split())
        integrator_gpu_source = "".join(self.integrator.gpu_source.split())
 
        self.assertEqual(integrator_gpu_source, gpu_source)


class TestBasicImageGenerator(unittest.TestCase):
    
    def setUp(self):
        self.r = 255
        self.g = 0
        self.b = 0
        self.image_generator = BasicImageGenerator(self.r, self.g, self.b)
    
    def test_generator_initalization(self):
        self.assertEqual(self.image_generator.base_hsv, colorsys.rgb_to_hsv(self.r / 255.0, self.g /255.0, self.b / 255.0))

    def test_image_generation_1(self):
        number_of_colors = 3
        test_data = [[0, 2, 1], [1, 0, 1], [1, 2, 0]]
        
        self.image_generator.generate_image("test_image_1", test_data, number_of_colors)
        
        self.assertEqual(len(self.image_generator.color_list), number_of_colors)
       
    def test_image_generation_2(self): 
        number_of_colors = 10
        test_data = [[random.randint(0, number_of_colors - 1) for j in range(1, 100)] for i in range(1, 100)]
                
        self.image_generator.generate_image("test_image_2", test_data, number_of_colors)    
        
        self.assertEqual(len(self.image_generator.color_list), number_of_colors)    

    def test_image_generation_3(self):
        number_of_colors = 5
        test_data = [[0, 2, 2, 4, 1], [1, 0, 2, 4, 1], [1, 1, 0, 4, 1], [4, 4, 4, 0, 1], [3, 3, 3, 3, 0]]
        
        self.image_generator.generate_image("test_image_3", test_data, number_of_colors)
        
        self.assertEqual(len(self.image_generator.color_list), number_of_colors)


class TestBasinsGenerator(unittest.TestCase):
    
    def setUp(self):
        magnet1 = MagnetModel(1.0, 0.0, 1.5)
        magnet2 = MagnetModel(-1.0, 1.0, 1.5)
        magnet3 = MagnetModel(-1.0, -1.0, 1.5)
        magnet4 = MagnetModel(0.0, 0.0, 0.5)
        
        self.magnets = [magnet1, magnet2,  magnet3, magnet4]
        
        self.plane_distance = 0.25
        self.friction = 0.3
        self.gravity_pullback = 0.5
        
        self.pendulum = PendulumModel(self.friction, self.gravity_pullback, self.plane_distance)
        self.pendulum.magnets = self.magnets
        
        self.time_step = 0.01
        self.integrator = EulerIntegrator(self.time_step)
        
        self.r = 255
        self.g = 0
        self.b = 0
        self.image_generator = BasicImageGenerator(self.r, self.g, self.b)
        
        self.size = 5
        self.resolution = 640
        self.basins_generator = BasinsGenerator(self.size, self.resolution)
        self.basins_generator.pendulum_model = self.pendulum
        self.basins_generator.integrator = self.integrator
        self.basins_generator.image_generator = self.image_generator
    
    def test_basins_initalization(self):
        self.assertEqual(self.basins_generator.size, self.size)
        self.assertEqual(self.basins_generator.resolution, self.resolution)
        self.assertEqual(self.basins_generator.cuda_device_number, 0)
        
        self.basins_generator.pendulum_model = self.pendulum
        self.basins_generator.integrator = self.integrator
        self.basins_generator.image_generator = self.image_generator

        self.assertEqual(self.basins_generator.pendulum_model, self.pendulum)
        self.assertEqual(self.basins_generator.integrator, self.integrator)
        self.assertEqual(self.basins_generator.image_generator, self.image_generator)
        
        self.assertEqual(self.basins_generator.pendulum_model.magnets, self.magnets)
 
    def test_basins_gpu_calculation(self):
        vel_vect = [0, 0]
        sim_time = 5
        delta = 0.2
        file_name = "test_image"
        
        self.basins_generator.calculate_basins(vel_vect, sim_time, delta)        
        self.basins_generator.draw_basins(file_name)

        self.assertEqual(len(self.basins_generator.result_data), 640)


if __name__ == '__main__':
    unittest.main()
    
    
    
    
    
        