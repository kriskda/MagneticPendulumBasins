  
         
class EulerIntegrator(object):
    
    gpu_source_template = """
            __device__ inline void calculateStep(float &x, float &y, float &vx, float &vy) {
                float ax, ay;
                
                diff_eq(ax, ay, x, y, vx, vy);
        
                vx = vx + ax * dt;
                vy = vy + ay * dt;
                
                x = x + vx * dt;
                y = y + vy * dt;                
            }
        """
    
    def __init__(self, time_step):
        self.time_step = time_step
        self.gpu_source = self.gpu_source_template