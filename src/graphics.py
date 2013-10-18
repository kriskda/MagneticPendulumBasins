import Image, ImageDraw
import colorsys  
import numpy



class ImageGenerator(object):
    
    RGB_COLOR_SIZE = 255.0
 
    def __init__(self, r, g, b):
        self.size = 0   
        self.draw_grid = False
        self.antialiasing = False   # if True image will be 2x smaller
        self.no_data_color = [0, 0, 0, 255]
        self.base_hsv = colorsys.rgb_to_hsv(r / self.RGB_COLOR_SIZE, g / self.RGB_COLOR_SIZE, b / self.RGB_COLOR_SIZE) # base color 

    def generate_image(self, file_name, result_data, track_length, number_of_colors):
        self._generate_color_list(number_of_colors)

        width, height = result_data.shape        

        print "  Adding pixels...",
        
        self.pixels = self.no_data_color * numpy.ones((width, height, 4), dtype=numpy.uint8)
        self._colorize_pixels(result_data, track_length)  
        
        image = Image.fromarray((self.pixels).astype(numpy.uint8))   
                
        print "done"
                
        if self.draw_grid:   
            image = GridGenerator.add_grid(self.size, image)  
                    
        if self.antialiasing:
            image = image.resize((width / 2, height / 2), Image.ANTIALIAS)
        
        print "  Saving image...",        
        image.rotate(90).save(file_name + ".png", "PNG")
        print "done"

    def _colorize_pixels(self, result_data, track_value):  
        pass
    
    '''
        Takes base color converted to hsv color space and changes hue of this color 
        based on number of magnets (i.e. colors) and current magnet (color) 
    '''
    def _generate_color_list(self, number_of_colors):
        self.color_list = []

        golden_ratio = 0.618033988749895
        hue, saturation, value = self.base_hsv

        for i in range(number_of_colors):            
            rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)
            r, g, b = self._correct_rgb_color(rgb_color)

            self.color_list.append([r, g, b, 255])
            
            hue = hue + golden_ratio
            hue = hue % 1
            
    def _correct_rgb_color(self, rgb_color):
        return map(lambda x: int(self.RGB_COLOR_SIZE * x), rgb_color)
    
     
class BasicImageGenerator(ImageGenerator):

    def _colorize_pixels(self, result_data, track_length):      
        number_of_colors = len(self.color_list)
        
        for i in range(number_of_colors): 
            indices = numpy.where(result_data == i)  
            self.pixels[indices] = self.color_list[i]


class AdvancedImageGenerator(ImageGenerator):    

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

                            
class GridGenerator(object):
    
    ''' We assume here that input image is square i.e. aspect ratio is one '''
    @staticmethod
    def add_grid(size, image):
        draw = ImageDraw.Draw(image) 
        
        width = image.size[0]
        height = image.size[1]
        
        if width != height:
            print "  Image supposed to have aspect ration equal one"
             
            return image   
                 
        white_color = "rgb(255,255,255)"
        #gray_color = "rgb(100,100,100)"        
 
        grid_size = width / size
        cursor_plus = width / 2
        cursor_minus = cursor_plus
        
        ''' Vertical & horizontal cross lines '''
        draw.line((0, width / 2, height, width / 2), fill = white_color)
        draw.line((height / 2, 0, height / 2, width), fill = white_color) 
                
        while (cursor_plus < width):
            cursor_plus = cursor_plus + grid_size
            cursor_minus = cursor_minus - grid_size
            
            ''' Grid vertical lines '''
            draw.line((0, cursor_plus, height, cursor_plus), fill = white_color)
            draw.line((0, cursor_minus, height, cursor_minus), fill = white_color)
            
            ''' Grid horizontal lines '''
            draw.line((cursor_plus, 0, cursor_plus, width), fill = white_color) 
            draw.line((cursor_minus, 0, cursor_minus, width), fill = white_color)        

        return image
                                


