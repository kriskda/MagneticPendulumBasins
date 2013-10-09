import Image
import colorsys  
import numpy
  

class ImageGenerator(object):
    
    RGB_COLOR_SIZE = 255.0

    def __init__(self, r, g, b):
        self.antialiasing = False   # if True image will be 2x smaller
        self.no_data_color = int('ff000000', 16)   # black color
        self.base_hsv = colorsys.rgb_to_hsv(r / self.RGB_COLOR_SIZE, g / self.RGB_COLOR_SIZE, b / self.RGB_COLOR_SIZE) # base color 

    def generate_image(self, file_name, result_data, track_length, number_of_colors):
        self._generate_color_list(number_of_colors)

        width = len(result_data)
        height = len(result_data[0])

        print "  Adding pixels...",

        vect = numpy.vectorize(self._colorize_pixel, otypes=[numpy.uint32])
        pixels = vect(result_data, track_length)

        image = Image.frombuffer('RGBA', (width, height), pixels, 'raw', 'RGBA', 0, 1)        
        
        print "done"
        
        if self.antialiasing:
            image = image.resize((width / 2, height / 2), Image.ANTIALIAS)
        
        print "  Saving image...",        
        image.transpose(Image.FLIP_TOP_BOTTOM).save(file_name + ".png", "PNG")
        print "done"

    def _colorize_pixel(self, color_number, track_value):
        pass
    
    '''
        Takes base color converted to hsv color space and changes hue of this color 
        based on number of magnets (i.e. colors) and current magnet (color) 
    '''
    def _generate_color_list(self, number_of_colors):
        self.color_list = []
      
        for i in range(0, number_of_colors):
            base_hue, base_saturation, base_value = self.base_hsv      
            
            new_hue = (base_hue + ((240 / number_of_colors) * i % 240)) / 240
            new_rgb_color = colorsys.hsv_to_rgb(new_hue, base_saturation, base_value)
            r, g, b = self._correct_rgb_color(new_rgb_color)

            int_color = int('ff%02x%02x%02x' % (b, g, r), 16)

            self.color_list.append(int_color)

    def _correct_rgb_color(self, rgb_color):
        return map(lambda x: int(self.RGB_COLOR_SIZE * x), rgb_color)
    
     
class BasicImageGenerator(ImageGenerator):

    def _colorize_pixel(self, color_number, track_value):        
        if color_number == -1:
            return self.no_data_color
        else:
            return self.color_list[color_number]


class AdvancedImageGenerator(ImageGenerator):
    
    def __init__(self, r, g, b):
        super(AdvancedImageGenerator, self).__init__(r, g, b)
        self.max_tracks_length = []
    
    def generate_image(self, file_name, result_data, track_length, number_of_colors):
        self._calculate_max_track_length(result_data, track_length, number_of_colors)       
        super(AdvancedImageGenerator, self).generate_image(file_name, result_data, track_length, number_of_colors)
    
    def _calculate_max_track_length(self, result_data, track_length, number_of_colors):
        for color in range(number_of_colors):
            non_color_indices = numpy.where(result_data.flatten() != color)[0]   # indices of elements which will be set to zero
            
            color_track_length = track_length.flatten()
            color_track_length[non_color_indices] = 0
            
            max_length = numpy.amax(color_track_length)

            self.max_tracks_length.append(max_length)    

    def _colorize_pixel(self, color_number, track_value):
        if color_number == -1:
            return self.no_data_color
        else:
            return self._get_color_value(self.color_list[color_number], track_value, self.max_tracks_length[color_number])

    def _get_color_value(self, color, track_value, max_track):
        scalefactor = 1 - (track_value / max_track)

        b = int( ((color >> 16) & 0xFF) * scalefactor ) << 16
        g = int( ((color >> 8) & 0xFF) * scalefactor ) << 8
        r = int( (color & 0xFF) * scalefactor )

        return 4278190080L + b + g + r  #  alpha + blue + green + red 
        
                                
