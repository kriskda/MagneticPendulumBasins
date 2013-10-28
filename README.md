MagneticPendulumBasins
======================

MagneticPendulumBasins is a CUDA generator of "basins of attraction" of magnetic pendulum for arbitrary number of magnets. Pendulum model is simplified and its description can by found here: http://beltoforion.de/magnetic_pendulum/magnetic_pendulum_en.html

![Alt text](/example.png "Example result")

Installation
============
Simply clone this repository and run either `python src/main.py` (png files generator) or `python src/vizualizer.py` (OpenGL visualizer). 
Note that the latter is experimental at the moment. It may also require you to:

	pip install PyDispatcher PyVRML97 OpenGLContext

Simple example
==============
Similar code as the following example can be found in 'main.py':

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
    #image_generator = BasicImageGenerator(255, 0, 0)	# just colors the regions 
    image_generator = AdvancedImageGenerator(255, 0, 0)	# colors and adds gradient based on trajectory length
    image_generator.draw_grid = False		
    image_generator.antialiasing = True     # if true image is 2x smaller
 
    # Parameters: simulation size, image size in pixels
    basins_generator = BasinsGenerator(5, 2000)
    basins_generator.pendulum_model = pendulum
    basins_generator.integrator = integrator
    basins_generator.image_generator = image_generator
   
    # Parameters: initial velocity vect, simulation time, delta, kernel sim time
    basins_generator.calculate_basins([0, 0], 50, 0.4, 2)   
    
    # Parameters: file_name
    basins_generator.draw_basins("basins") 	
    

