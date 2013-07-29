MagneticPendulumBasins
======================

MagneticPendulumBasins is a CUDA generator of "basins of attraction" of magnetic pendulum. Pendulum model is simplified and its description can by found here: http://beltoforion.de/magnetic_pendulum/magnetic_pendulum_en.html

Installation
============
Download and run main.py: 'python main.py'


Simple example
==============
    # Parameters: pos_x, pos_y, magnet strength constant
    magnet1 = MagnetModel(1.0, 0.0, 1.5)
    magnet2 = MagnetModel(-1.0, -1.0, 1.5)
    magnet3 = MagnetModel(-1.0, 1.0, 1.5)
    
    magnets = [magnet1, magnet2, magnet3]     	

    # Parameters: friction, gravity pullback, plane distance
    pendulum = PendulumModel(0.3, 0.5, 0.25)
    pendulum.magnets = magnets 

    # Parameters: time_step
    integrator = EulerIntegrator(0.01)  
    
    # Paramaters: r, g, b - startign color definition
    image_generator = BasicImageGenerator(255, 0, 0)
    image_generator.antialiasing = True     # image will be 2x smaller
    
    # Parameters: size, image size in pixels
    basins_generator = BasinsGenerator(200, 2000)
    basins_generator.pendulum_model = pendulum
    basins_generator.integrator = integrator
    basins_generator.image_generator = image_generator
   
    # Parameters: initial velocity vect, simulation time, delta, kernel sim time
    basins_generator.calculate_basins([0, 0], 50, 0.4, 2)   
    
    # Parameters: file_name
    basins_generator.draw_basins("basins") 	
    
