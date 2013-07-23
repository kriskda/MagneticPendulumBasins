import Image
import colorsys  
import random
   

class BasicImageGenerator(object):

    RGB_COLOR_SIZE = 255.0

    def __init__(self, r, g, b):
        self.no_data_color = (0, 0, 0)
        self.base_hsv = colorsys.rgb_to_hsv(r / self.RGB_COLOR_SIZE, g / self.RGB_COLOR_SIZE, b / self.RGB_COLOR_SIZE) # base color 

    def generate_image(self, file_name, result_data, number_of_colors):
        width = len(result_data)
        height = len(result_data[0])

        image = Image.new("RGB", (width, height))
        pixels = image.load()

        self._generate_color_list(number_of_colors)

        for i, row in enumerate(result_data):
            for j, color_number in enumerate(row):
                pixels[j, i] = self._colorize_pixel(color_number)

        image.save(file_name + ".png", "PNG")
        

    def _colorize_pixel(self, color_number):

        if color_number == -1:
            return self.no_data_color
        else:
            return self.color_list[color_number]
    
    '''
        Takes base color converted to hsv color space and changes hue of this color 
        based on number of magnets (i.e. colors) and current magnet (color) 
    '''
    def _generate_color_list(self, number_of_colors):
        self.color_list = []
        
        for i in range(0, number_of_colors):
            base_hue = self.base_hsv[0]
            base_saturation = self.base_hsv[1]
            base_value = self.base_hsv[2]        
            
            new_hue = base_hue + ((240 / number_of_colors) * i % 240)
            new_rgb_color = colorsys.hsv_to_rgb(new_hue / 240, base_saturation, base_value)
    
            r = int(self.RGB_COLOR_SIZE * new_rgb_color[0])
            g = int(self.RGB_COLOR_SIZE * new_rgb_color[1])
            b = int(self.RGB_COLOR_SIZE * new_rgb_color[2])
            
            self.color_list.append((r, g, b))




        
                                