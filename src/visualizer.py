from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from OpenGL.GL import shaders
from OpenGL.arrays import vbo
from OpenGLContext.arrays import *

import numpy

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600

import colorsys 

from models import MagnetModel, PendulumModel
from integrators import EulerIntegrator
from basins import BasinsGenerator



class WindowView(object):

    def __init__(self, result_data, track_length, number_of_colors):
        glutInit()
        glutInitWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT)
        glutInitWindowPosition(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
        glutCreateWindow("Magnetic Pendulum Basins Visualizer")
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA)
        glutDisplayFunc(self.draw)
        #glutTimerFunc(30, self.timer, 30)

        VERTEX_SHADER = shaders.compileShader("""#version 410 compatibility
            varying vec4 vertex_color;
             
            void main() {
                gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
                vertex_color = gl_Color;
            }""", GL_VERTEX_SHADER)
        
        FRAGMENT_SHADER = shaders.compileShader("""#version 330
            varying vec4 vertex_color;
            
            void main() {
                gl_FragColor = vertex_color;
            }""", GL_FRAGMENT_SHADER)
        
        self.shader = shaders.compileProgram(VERTEX_SHADER, FRAGMENT_SHADER)


        self.texture = self._generate_texture(result_data, track_length, number_of_colors)        
 
        self._init_gl()
        glutMainLoop()

    def _init_gl(self):
        glClearColor(1.0, 1.0, 1.0, 0.0)
        glColor3f(0.0, 0.0, 0.0)
        glPointSize(2.0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(-1.0, 1.0, -1.0, 1.0)

    def _generate_texture(self, result_data, track_length, number_of_colors):
        self._generate_color_list(number_of_colors)

        width, height = result_data.shape   
        
        self.pixels = [0, 0, 0, 255] * numpy.ones((width, height, 4), dtype=numpy.uint8)
        self._colorize_pixels(result_data, track_length) 
        
        texture = glGenTextures(1)
        glPixelStorei(GL_UNPACK_ALIGNMENT,1)
        glBindTexture(GL_TEXTURE_2D, texture)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.pixels)
        
        return texture

    def _setup_texture(self):
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_TEXTURE_2D)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

    def _draw_plane(self):
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(-1, -1)
        glTexCoord2f(1, 0)
        glVertex2f(1, -1)
        glTexCoord2f(1, 1)
        glVertex2f(1, 1)
        glTexCoord2f(0, 1)
        glVertex2f(-1, 1)
        glEnd()

    def draw(self):
        #shaders.glUseProgram(self.shader)
        
        glClear(GL_COLOR_BUFFER_BIT)
 
        self._setup_texture()
        self._draw_plane()
        
        glFlush()     
        
    ''' Some functions in my first prototype which presumably will be deleted as work progresses '''      
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
    ''' End some functions '''
        
if __name__ == "__main__":
        magnet1 = MagnetModel(1.0, 0.0, 0.5)
        magnet2 = MagnetModel(-1.0, -1.0, 0.5)
        magnet3 = MagnetModel(-1.0, 1.0, 0.5)
 
        magnets = [magnet1, magnet2, magnet3] 
        
        pendulum = PendulumModel(0.2, 0.5, 0.1)
        pendulum.magnets = magnets 

        integrator = EulerIntegrator(0.01)  
        
        basins_generator = BasinsGenerator(5, 600)
        basins_generator.pendulum_model = pendulum
        basins_generator.integrator = integrator

        basins_generator.calculate_basins([0, 0], 30, 0.2, 15)   

        view = WindowView(basins_generator.result_data, basins_generator.track_length, len(magnets))  
    
              
