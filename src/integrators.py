

class EulerIntegrator(object):
    
    gpu_source_template = """
            __device__ inline void calculateStep(float t, float &x, float &y, float &vx, float &vy) {
                float dt = %sf;
                float nx, ny, nvx, nvy;
                
                diff_eq(t, nx, ny, nvx, nvy, x, y, vx, vy);
        
                vx = vx + nvx * dt;
                vy = vy + nvy * dt;
                
                x = x + nx * dt;
                y = y + ny * dt;                
            }
        """
    
    def __init__(self, time_step):
        self.time_step = time_step
        self.gpu_source = self.gpu_source_template % time_step 