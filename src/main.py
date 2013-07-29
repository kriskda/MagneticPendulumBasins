from models import *
from integrators import EulerIntegrator
from graphics import BasicImageGenerator
from basins import BasinsGenerator

import math

 
def main():
    magnets_example = 5
    
    if magnets_example == 1:
        # Parameters: pos_x, pos_y, magnet strength constant
        magnet1 = MagnetModel(1.0, 1.0, 1.5)
        magnet2 = MagnetModel(-1.0, 1.0, 1.5)
        magnet3 = MagnetModel(-1.0, -1.0, 1.5)
        magnet4 = MagnetModel(1.0, -1.0, 1.5)
        
        magnets = [magnet1, magnet2, magnet3, magnet4]
    elif magnets_example == 2:
        magnets = []
        
        for i in range(1, 7):
            angle = i * math.pi / 3
            magnets.append(MagnetModel(math.cos(angle), math.sin(angle), 0.5)) 
            
        magnets.append(MagnetModel(0.0, 0.0, 0.1))
    elif magnets_example == 3:
        magnet1 = MagnetModel(1.0, 1.0, 1.5)
        magnet2 = MagnetModel(-1.0, -1.0, 1.5)
  
        magnets = [magnet1, magnet2]
    elif magnets_example == 4:
        magnets = []
        
        for i in range(1, 13):
            angle = i * math.pi / 6
            magnets.append(MagnetModel(math.cos(angle), math.sin(angle), 1.5)) 
            
        magnets.append(MagnetModel(0.0, 0.0, 0.1))  
    elif magnets_example == 5:
        magnet1 = MagnetModel(1.0, 0.0, 1.5)
        magnet2 = MagnetModel(-1.0, -1.0, 1.5)
        magnet3 = MagnetModel(-1.0, 1.0, 1.5)
 
        magnets = [magnet1, magnet2, magnet3]      
    

    # Parameters: friction, gravity_pullback, plane_distance
    pendulum = PendulumModel(0.3, 0.5, 0.25)
    pendulum.magnets = magnets 

    # Parameters: time_step
    integrator = EulerIntegrator(0.01)  
    
    # Paramaters: r, g, b - startign color definition
    image_generator = BasicImageGenerator(255, 0, 0)
    image_generator.antialiasing = True     # image will be 2x smaller
    
    # Parameters: size
    basins_generator = BasinsGenerator(20, 2000)
    basins_generator.pendulum_model = pendulum
    basins_generator.integrator = integrator
    basins_generator.image_generator = image_generator
   
    # Parameters: initial velocity vect, simulation time, delta, kernel sim time
    basins_generator.calculate_basins([0, 0], 50, 0.4, 2)   
    
    # Parameters: file_name
    basins_generator.draw_basins("basins") 
    

if __name__ == "__main__":
    main()
    
    
    
    
    
    
