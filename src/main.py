from models import *
from integrators import EulerIntegrator
from graphics import BasicImageGenerator
from basins import BasinsGenerator
 
 
def main():
    # Parameters: pos_x, pos_y, magnet strength constant
    magnet1 = MagnetModel(1.0, 1.0, 1.5)
    magnet2 = MagnetModel(-1.0, 1.0, 1.5)
    magnet3 = MagnetModel(-1.0, -1.0, 1.5)
    magnet4 = MagnetModel(1.0, -1.0, 1.5)
        
    magnets = [magnet1, magnet2,  magnet3, magnet4]
    
    # Parameters: friction, gravity_pullback, plane_distance
    pendulum = PendulumModel(0.3, 0.5, 0.25)
    pendulum.magnets = magnets 

    # Parameters: time_step
    integrator = EulerIntegrator(0.01)  
    
    # Paramaters: r, g, b - startign color definition
    image_generator = BasicImageGenerator(255, 0, 0)
    
    # Parameters: size, resolution
    basins_generator = BasinsGenerator(5, 640)
    basins_generator.pendulum_model = pendulum
    basins_generator.integrator = integrator
    basins_generator.image_generator = image_generator
   
    # Parameters: initial velocity vect, simulation time, delta
    basins_generator.calculate_basins([0, 0], 80, 0.2)   
    
    # Parameters: file_name
    basins_generator.draw_basins("basins") 
    

if __name__ == "__main__":
    main()
    
    
    
    
    
    