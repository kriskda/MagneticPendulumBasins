import pycuda.driver as cuda
 

class BasinsGenerator(object):
    
    def __init__(self, size, resolution, cuda_device_number = 0):
        self.size = size
        self.resolution = resolution
        self.cuda_device_number = cuda_device_number        
        self.pendulum_model = None
        self.integrator = None
        self.image_generator = None
        
        self.result_data = []
        
    def calculate_basins(self, sim_time):
        pass
    
    def _do_cuda_calculation(self, pos0, vel0, sim_time):
        pass
    
    def _initalize_cuda(self):
        cuda.init() #init pycuda driver
        current_dev = cuda.Device(self.cuda_device_number) #device we are working on
        
        self.cuda_context = current_dev.make_context() #make a working context
        self.cuda_context.push() #let context make the lead
    
    def _deactivate_cuda(self):  
        self.cuda_context.pop() #deactivate again
        self.cuda_context.detach() #delete it
        
    def _save_data(self):
        pass
    
    def draw_basins(self):
        pass