from src.functions import CommonFunctions    
     
class MagnetModel(object): 
    
    def __init__(self, pos_x, pos_y, magnetic_strength):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.magnetic_strength = magnetic_strength


class PendulumModel(object):
    
    gpu_source_template = """     
            __device__ inline void diff_eq(float &nx, float &ny, float &nvx, float &nvy, float x, float y, float vx, float vy) { 
                %s
                float amx = 0.0f;
                float amy = 0.0f;
                             
                for (int i = 0 ; i < n ; i++) {
                    float deltaX = xm[i] - x;
                    float deltaY = ym[i] - y;
                    
                    float dist = sqrtf(deltaX * deltaX + deltaY * deltaY + d2);
                    float distPow3 = dist * dist *dist;
                
                    amx += km[i] * deltaX / distPow3;
                    amy += km[i] * deltaY / distPow3;
                }
                      
                nvx = -kf * vx - kg * x + amx;
                nvy = -kf * vy - kg * y + amy;
                
                nx = vx;
                ny = vy;
            }
            
            __device__ int determineMagnet(float x, float y, float delta) {
                %s
                
                return -1;        
            }
            """
    
    def __init__(self, friction, gravity_pullback, plane_distance):
        self.magnets = []
        self.gravity_pullback = gravity_pullback
        self.friction = friction
        self.plane_distance = plane_distance
        self.gpu_source = ""
        
    def prepare_gpu_source(self):
        pendulum_constants = """
                const float kf = %sf;
                const float kg = %sf;
                const float d2 = %sf * %sf;                
            """ % (self.friction, self.gravity_pullback, self.plane_distance, self.plane_distance)        
        
        magnets_pos_x = CommonFunctions.array_to_float_carray([magnet.pos_x for magnet in self.magnets])
        magnets_pos_y = CommonFunctions.array_to_float_carray([magnet.pos_y for magnet in self.magnets])
        magnets_strength = CommonFunctions.array_to_float_carray([magnet.magnetic_strength for magnet in self.magnets])
        
        magnets_constants = """
                const int n = %s;
                
                const float xm[n] = %s;
                const float ym[n] = %s;
                const float km[n] = %s;        
            """ % (len(self.magnets), magnets_pos_x, magnets_pos_y, magnets_strength)
            
        determine_magnets = ""
        for i, magnet in enumerate(self.magnets):
            determine_magnets += """
                bool m%sdx = ((%sf - delta) <= x) && (x <= (%sf + delta));
                bool m%sdy = ((%sf - delta) <= y) && (y <= (%sf + delta));
   
                if (m%sdx && m%sdy) {
                    return %s;
                } 
            """ % (i, magnet.pos_x, magnet.pos_x, i, magnet.pos_y, magnet.pos_y, i, i, i)
 
        self.gpu_source = self.gpu_source_template % (pendulum_constants + magnets_constants, determine_magnets)
   
   
   
   
   
    