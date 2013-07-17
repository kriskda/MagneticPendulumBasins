import numpy 
import pycuda.driver as cuda

from pycuda.compiler import SourceModule 
    

class BasinsGenerator(object):
    
    constants_source_template = """
        __const__ float dt = %sf;
        __const__ float simTime = %sf;              
    """
    
    main_source_template = """
            __global__ void basins(float *cudaResult, float *posx0, float *posy0) {
                const int idx = blockIdx.y  * gridDim.x  * blockDim.z * blockDim.y * blockDim.x + 
                                blockIdx.x  * blockDim.z * blockDim.y * blockDim.x + 
                                threadIdx.z * blockDim.y * blockDim.x + 
                                threadIdx.y * blockDim.x + 
                                threadIdx.x;

                float x = posx0[idx];
                float y = posy0[idx];
                float vx = %sf;
                float vy = %sf;
                float t = 0.0f;
                
                do {                
                    calculateStep(x, y, vx, vy);
                    t += dt;  
                } while (t <= simTime);

                cudaResult[idx] = determineMagnet(x, y, %sf);
            }
        """
    
    def __init__(self, size, resolution, cuda_device_number = 0):
        self.size = size
        self.resolution = resolution
        self.cuda_device_number = cuda_device_number        
        self.pendulum_model = None
        self.integrator = None
        self.image_generator = None
        
        self.gpu_source = ""
        self.result_data = []
        
    def calculate_basins(self, vel0, sim_time, delta):
        print "> Calculating basins"
        
        self.prepare_gpu_source(vel0, sim_time, delta)

        scale = self.size / float(self.resolution)  # after modification change grid & block sizes in GPU !!!

        self.n_array = numpy.arange(-self.size / 2.0, self.size / 2.0, scale)
  
        posx0 = []
        posy0 = []

        for i in self.n_array:
            for j in self.n_array:
                posx0.append(i)
                posy0.append(j)
        
        posx0 = numpy.array(posx0).astype(numpy.float32)
        posy0 = numpy.array(posy0).astype(numpy.float32)        
        
        self._do_cuda_calculation([posx0, posy0])
    
    def prepare_gpu_source(self, vel0, sim_time, delta):
        self.pendulum_model.prepare_gpu_source()
        
        constants_source = self.constants_source_template % (self.integrator.time_step, float(sim_time))
        main_source = self.main_source_template % (float(vel0[0]), float(vel0[1]), float(delta)) 
        self.gpu_source = constants_source + self.pendulum_model.gpu_source + self.integrator.gpu_source + main_source

    def _do_cuda_calculation(self, pos0):
        cuda_result = numpy.zeros_like(pos0)
        
        self._initalize_cuda()        

        mod = SourceModule(self.gpu_source)

        do_basins = mod.get_function("basins")
        do_basins(cuda.Out(cuda_result), 
                  cuda.In(pos0[0]), 
                  cuda.In(pos0[1]), 
                  block = (16, 16, 1), 
                  grid = (40, 40))
        
        self._deactivate_cuda()        
        self._save_data(cuda_result)
    
    def _initalize_cuda(self):
        cuda.init() 
        current_dev = cuda.Device(self.cuda_device_number) 
        
        self.cuda_context = current_dev.make_context() 
        self.cuda_context.push()
    
    def _deactivate_cuda(self):  
        self.cuda_context.pop() 
        self.cuda_context.detach() 
        
    def _save_data(self, cuda_result):
        is_nodata_pixels = -1 in cuda_result[0]
        
        if is_nodata_pixels:
            print "  WARNING: some pixels could not be assignet to magnet"
        
        self.result_data = numpy.reshape(cuda_result[0], (self.resolution, self.resolution))
        
        #print self.gpu_source   
        #print self.result_data        
                
    def draw_basins(self, file_name):
        print "> Generating image"
        
        self.image_generator.generate_image(file_name, self.result_data, len(self.pendulum_model.magnets))
    
    
    