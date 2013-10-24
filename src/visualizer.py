import numpy
import colorsys 
import math

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from OpenGL.GL import shaders
from OpenGL.arrays import vbo
from OpenGLContext.arrays import *

from models import MagnetModel, PendulumModel
from integrators import EulerIntegrator
from basins import BasinsGenerator
 

SCREEN_SIZE = 600

''' Some functions in my first prototype which presumably will be deleted as work progresses '''  
class DataConverter(object):
    
    NO_DATA_COLOR = [0, 0, 0, 255]  
    
    def _correct_rgb_color(self, rgb_color):
        return map(lambda x: int(255.0 * x), rgb_color)
    
    def _generate_color_list(self, number_of_colors):
        self.color_list = []
        RGB_COLOR_SIZE = 255.0
    
        golden_ratio = 0.618033988749895
        hue, saturation, value = colorsys.rgb_to_hsv(255 / RGB_COLOR_SIZE, 0 / RGB_COLOR_SIZE, 0 / RGB_COLOR_SIZE)
    
        for i in range(number_of_colors):            
            rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)
            r, g, b = self._correct_rgb_color(rgb_color)
    
            self.color_list.append([r, g, b, 255])
                
            hue = hue + golden_ratio
            hue = hue % 1

    def _colorize_pixels(self, result_data, track_length):      
        number_of_colors = len(self.color_list)
  
        for i in range(number_of_colors): 
            indices = numpy.where(result_data == i)  
            r, g, b, a = self.color_list[i]    
            
            tracks = track_length[indices]
   
            scale_factor = 1 - tracks / tracks.max()

            scaled = self.pixels[indices]
            scaled[:, 0] = scale_factor * r
            scaled[:, 1] = scale_factor * g
            scaled[:, 2] = scale_factor * b
            #scaled[:, 3] = a                    # no need for alpha, already set

            self.pixels[indices] = scaled

    def generate_pixel_data(self, basins_generator, number_of_magnets):
        self._generate_color_list(number_of_magnets)

        width, height = basins_generator.result_data.shape   
        
        self.pixels = self.NO_DATA_COLOR * numpy.ones((width, height, 4), dtype=numpy.uint8)
        self._colorize_pixels(basins_generator.result_data, basins_generator.track_length) 

        return width, height, self.pixels


class OpenGLvisualizer(object):
    
    def __init__(self, basins_generator, number_of_magnets):
        self.basins_generator = basins_generator
        self.number_of_magnets = number_of_magnets
        self.is_control_points = True
        self.is_lmb_magnet_pressed = False
        self.dragged_magnet = None
        self.span = basins_generator.size / 2.0
        
        self._init_gl()
        
    def _init_gl(self):
        glutInit()
        glutInitWindowSize(SCREEN_SIZE, SCREEN_SIZE)
        glutInitWindowPosition(SCREEN_SIZE / 2, SCREEN_SIZE / 2)
        glutCreateWindow("Magnetic Pendulum Basins Visualizer")
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA)
        glutDisplayFunc(self.draw)
        glutKeyboardFunc(self._keyboard_action)
        glutMouseFunc(self._mouse_click_action)
        glutMotionFunc(self._mouse_move_action)

        glClearColor(0.0, 0.0, 0.0, 0.0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
                
        gluOrtho2D(-self.span, self.span, -self.span, self.span)        
        glutMainLoop()
      
    def _win_to_world_coords(self, win_x, win_y):
        x = self.span * (2.0 * win_x / SCREEN_SIZE - 1)
        y = self.span * (-2.0 * win_y / SCREEN_SIZE + 1)
        
        return (x, y)
      
    def _keyboard_action(self, key, x, y):
        if key == '\033':   # exit on Esc key         
            exit()
        elif key == 'r':
            self._render_image()
        elif key == 'c':       
            self._control_points_action()     
            
    def _mouse_click_action(self, button, state, win_x, win_y): 
        if button == 0 and state == 0:  # LMB pressed
            x, y = self._win_to_world_coords(win_x, win_y)
            
            magnets = self.basins_generator.pendulum_model.magnets
            
            radius = 0.05 # TBD change to variable
            
            for magnet in magnets:
                pos_x, pos_y = magnet.pos_x, magnet.pos_y
                
                check_radius = math.sqrt((x - pos_x) * (x - pos_x) + (y - pos_y) * (y - pos_y))
                
                if check_radius <= radius:
                    self.is_lmb_magnet_pressed = True
                    self.dragged_magnet = magnet
                    break

        elif button == 0 and state == 1:    # LMB released            
            self.is_lmb_magnet_pressed = False
        
    def _mouse_move_action(self, win_x, win_y):
        if self.is_lmb_magnet_pressed:
            x, y = self._win_to_world_coords(win_x, win_y)
        
            if self.dragged_magnet != None:
                self.dragged_magnet.pos_x = x
                self.dragged_magnet.pos_y = y

                glutPostRedisplay()
            
    def _render_image(self):
        self.basins_generator.calculate_basins([0, 0], 30, 0.2, 30)   
        self._generate_texture()
        glutPostRedisplay()   

    def _control_points_action(self):
        if self.is_control_points:
            self.is_control_points = False
        else:
            self.is_control_points = True
             
        glutPostRedisplay()

    def _generate_texture(self):    
        width, height, self.pixels = DataConverter().generate_pixel_data(self.basins_generator, self.number_of_magnets)
 
        texture = glGenTextures(1)
        glPixelStorei(GL_UNPACK_ALIGNMENT,1)
        glBindTexture(GL_TEXTURE_2D, texture)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.pixels)

    def draw(self):     
        glClear(GL_COLOR_BUFFER_BIT)
        
        self._setup_antialiasing()
        self._setup_texture()
        self._draw_plane()
        
        if self.is_control_points:
            self._draw_control_points()
        
        self._disable()
        
        glFlush()       
        
    def _setup_antialiasing(self):   
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) 
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
                
        glEnable(GL_POLYGON_SMOOTH)      
        glEnable(GL_BLEND)
        
    def _setup_texture(self):
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

    def _draw_plane(self):
        glColor3f(0.0, 0.0, 0.0)
        
        xvals = (-self.span, -self.span, self.span, self.span)
        yvals = (self.span, -self.span, -self.span, self.span)
        
        svals = (0, 0, 1, 1)
        tvals = (0, 1, 1, 0)
        
        glBegin(GL_POLYGON);
        for i in range(4):
            glVertex2f(xvals[i], yvals[i])
            glTexCoord2f(svals[i], tvals[i])

        glEnd()  
        
    def _draw_control_points(self):
        magnets = self.basins_generator.pendulum_model.magnets
        
        glColor3f(1.0, 1.0, 1.0)
        
        for magnet in magnets:            
            x, y, z, radius = magnet.pos_x, magnet.pos_y, 0.5, 0.05
            
            glBegin(GL_TRIANGLE_FAN)
            
            glVertex3f(x, y, z)
            for angle in range(360):
                rad = angle * math.pi / 180
                glVertex3f(x + math.sin(rad) * radius, y + math.cos(rad) * radius, z)
     
            glEnd()   
        
    def _disable(self):   
        glDisable(GL_POLYGON_SMOOTH)      
        glDisable(GL_BLEND)
        glDisable(GL_MULTISAMPLE)
        glDisable(GL_TEXTURE_2D)
            
        
if __name__ == "__main__":
        print "Hit ESC key to quit, 'r' to render, and 'c' to hide / display control points"
        print "Drag magnet control points to change magnets position"
        print ""
        
        magnet1 = MagnetModel(1.0, 0.0, 0.5)
        magnet2 = MagnetModel(-1.0, -1.0, 0.5)
        magnet3 = MagnetModel(-1.0, 1.0, 0.5)

        magnets = [magnet1, magnet2, magnet3] 
        
        pendulum = PendulumModel(0.2, 0.5, 0.1)
        pendulum.magnets = magnets 

        integrator = EulerIntegrator(0.01)  
        
        basins_generator = BasinsGenerator(5, SCREEN_SIZE)
        basins_generator.pendulum_model = pendulum
        basins_generator.integrator = integrator

        visualizer = OpenGLvisualizer(basins_generator, len(magnets))  
        
        
        