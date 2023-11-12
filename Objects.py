import taichi as ti

@ti.data_oriented
class ClothSystem:
    def __init__(self, number, position, velocity, forces, k_s = 5e4,
                        gamma_s = 3e3,
                        k_d = 5e4 / ti.sqrt(2),
                        gamma_d = 3e3 / ti.sqrt(2),
                        k_b = 5e4/2,
                        gamma_b = 3e3/2,
                        damp = 5.0,
                        l = None,time = 0.0,):
        
        self.number = number
        N = number
        self.position = position
        self.velocity = velocity
        self.forces = forces
        self.vertices = ti.Vector.field(3, dtype=float, shape=N * N)
        for i, j in ti.ndrange(N, N):
            self.vertices[i * N + j] = position[i, j]
        self.ball_mass = 1e4
        self.ball_radius = 0.2
        self.ball_center = ti.Vector.field(3, dtype=float, shape=(1,))
        self.ball_center[0] = ti.Vector([0.5, self.ball_radius, 0.4])
        self.ball_v = ti.Vector.field(3, dtype=float, shape=(1,))
        self.ball_v[0] = ti.Vector([0.0, 0.0, 0.0])
        self.inertia = 2/5*self.ball_mass*self.ball_radius*self.ball_radius
        self.time = time
        self.k_s  = k_s 
        self.gamma_s  = gamma_s 
        self.k_d = k_d
        self.gamma_d = gamma_d
        self.k_b  = k_b 
        self.gamma_b  = gamma_b 
        if l is None:
            self.l = 1/float(self.number)
        else:
            self.l = l
        self.damp = damp

    @ti.func
    def self_evolve(self,h):
        self.ApplySpring()
        self.ApplyDamping()
    
    @ti.func
    def ClearForces(self):
        for i in range(self.number):
            for j in range(self.number):
                self.forces[i, j] = ti.Vector([0.0,0.0,0.0])

    @ti.func
    def ApplySpring(self):
        k_s  = self.k_s 
        gamma_s  = self.gamma_s 
        k_d = self.k_d
        gamma_d = self.gamma_d
        k_b  = self.k_b 
        gamma_b  = self.gamma_b 
        l = self.l
        for i, j in ti.ndrange(self.number, self.number):
                if(i>0):
                    self.forces[i, j] += (k_s*(ti.Vector.norm(self.position[i-1, j]-self.position[i, j])-l)+gamma_s*ti.Vector.dot(self.velocity[i-1, j]-self.velocity[i, j], self.position[i-1, j]-self.position[i, j])//ti.Vector.norm(self.position[i-1, j]-self.position[i, j]))*(self.position[i-1, j]-self.position[i, j])/ti.Vector.norm(self.position[i-1, j]-self.position[i, j])
                if(i<self.number-1):
                    self.forces[i, j] += (k_s*(ti.Vector.norm(self.position[i+1, j]-self.position[i, j])-l)+gamma_s*ti.Vector.dot(self.velocity[i+1, j]-self.velocity[i, j], self.position[i+1, j]-self.position[i, j])/ti.Vector.norm(self.position[i+1, j]-self.position[i, j]))*(self.position[i+1, j]-self.position[i, j])/ti.Vector.norm(self.position[i+1, j]-self.position[i, j])
                if(j>0):
                    self.forces[i, j] += (k_s*(ti.Vector.norm(self.position[i, j-1]-self.position[i, j])-l)+gamma_s*ti.Vector.dot(self.velocity[i, j-1]-self.velocity[i, j], self.position[i, j-1]-self.position[i, j])/ti.Vector.norm(self.position[i, j-1]-self.position[i, j]))*(self.position[i, j-1]-self.position[i, j])/ti.Vector.norm(self.position[i, j-1]-self.position[i, j])
                if(j<self.number-1):
                    self.forces[i, j] += (k_s*(ti.Vector.norm(self.position[i, j+1]-self.position[i, j])-l)+gamma_s*ti.Vector.dot(self.velocity[i, j+1]-self.velocity[i, j], self.position[i, j+1]-self.position[i, j])/ti.Vector.norm(self.position[i, j+1]-self.position[i, j]))*(self.position[i, j+1]-self.position[i, j])/ti.Vector.norm(self.position[i, j+1]-self.position[i, j])
                if(i>0 and j>0):
                    self.forces[i, j] += (k_d*(ti.Vector.norm(self.position[i-1, j-1]-self.position[i, j])-l*ti.sqrt(2))+gamma_d*ti.Vector.dot(self.velocity[i-1, j-1]-self.velocity[i, j], self.position[i-1, j-1]-self.position[i, j])/ti.Vector.norm(self.position[i-1, j-1]-self.position[i, j]))*(self.position[i-1, j-1]-self.position[i, j])/ti.Vector.norm(self.position[i-1, j-1]-self.position[i, j])
                if(i>0 and j<self.number-1):
                    self.forces[i, j] += (k_d*(ti.Vector.norm(self.position[i-1, j+1]-self.position[i, j])-l*ti.sqrt(2))+gamma_d*ti.Vector.dot(self.velocity[i-1, j+1]-self.velocity[i, j], self.position[i-1, j+1]-self.position[i, j])/ti.Vector.norm(self.position[i-1, j+1]-self.position[i, j]))*(self.position[i-1, j+1]-self.position[i, j])/ti.Vector.norm(self.position[i-1, j+1]-self.position[i, j])
                if(i<self.number-1 and j>0):
                    self.forces[i, j] += (k_d*(ti.Vector.norm(self.position[i+1, j-1]-self.position[i, j])-l*ti.sqrt(2))+gamma_d*ti.Vector.dot(self.velocity[i+1, j-1]-self.velocity[i, j], self.position[i+1, j-1]-self.position[i, j])/ti.Vector.norm(self.position[i+1, j-1]-self.position[i, j]))*(self.position[i+1, j-1]-self.position[i, j])/ti.Vector.norm(self.position[i+1, j-1]-self.position[i, j])
                if(i<self.number-1 and j<self.number-1):
                    self.forces[i, j] += (k_d*(ti.Vector.norm(self.position[i+1, j+1]-self.position[i, j])-l*ti.sqrt(2))+gamma_d*ti.Vector.dot(self.velocity[i+1, j+1]-self.velocity[i, j], self.position[i+1, j+1]-self.position[i, j])/ti.Vector.norm(self.position[i+1, j+1]-self.position[i, j]))*(self.position[i+1, j+1]-self.position[i, j])/ti.Vector.norm(self.position[i+1, j+1]-self.position[i, j])
                if(i>1):
                    self.forces[i, j] += (k_b*(ti.Vector.norm(self.position[i-2, j]-self.position[i, j])-2*l)+gamma_b*ti.Vector.dot(self.velocity[i-2, j]-self.velocity[i, j], self.position[i-2, j]-self.position[i, j])/ti.Vector.norm(self.position[i-2, j]-self.position[i, j]))*(self.position[i-2, j]-self.position[i, j])/ti.Vector.norm(self.position[i-2, j]-self.position[i, j])
                if(i<self.number-2):
                    self.forces[i, j] += (k_b*(ti.Vector.norm(self.position[i+2, j]-self.position[i, j])-2*l)+gamma_b*ti.Vector.dot(self.velocity[i+2, j]-self.velocity[i, j], self.position[i+2, j]-self.position[i, j])/ti.Vector.norm(self.position[i+2, j]-self.position[i, j]))*(self.position[i+2, j]-self.position[i, j])/ti.Vector.norm(self.position[i+2, j]-self.position[i, j])
                if(j>1):
                    self.forces[i, j] += (k_b*(ti.Vector.norm(self.position[i, j-2]-self.position[i, j])-2*l)+gamma_b*ti.Vector.dot(self.velocity[i, j-2]-self.velocity[i, j], self.position[i, j-2]-self.position[i, j])/ti.Vector.norm(self.position[i, j-2]-self.position[i, j]))*(self.position[i, j-2]-self.position[i, j])/ti.Vector.norm(self.position[i, j-2]-self.position[i, j])
                if(j<self.number-2):
                    self.forces[i, j] += (k_b*(ti.Vector.norm(self.position[i, j+2]-self.position[i, j])-2*l)+gamma_b*ti.Vector.dot(self.velocity[i, j+2]-self.velocity[i, j], self.position[i, j+2]-self.position[i, j])/ti.Vector.norm(self.position[i, j+2]-self.position[i, j]))*(self.position[i, j+2]-self.position[i, j])/ti.Vector.norm(self.position[i, j+2]-self.position[i, j])

    @ti.func
    def ApplyDamping(self):
        damp = self.damp
        for i in range(self.number):
            for j in range(self.number):
                self.forces[i, j] -= damp*self.velocity[i, j]

    @ti.func
    def Kinematics(self,h:ti.f32):
        N = self.number
        for i, j in ti.ndrange(N, N):
            self.position[i, j] += self.velocity[i, j]*h
            self.velocity[i, j] += self.forces[i, j]*h
            self.vertices[i * N + j] = self.position[i, j]
        pass
   
@ti.data_oriented
class Ball:
    def __init__(self):
        pass


@ti.data_oriented
class FreeBallOnGround(Ball):
    def __init__(self, ball_v=None,  time=0.0, ball_mass = 1e4,ball_center = None, ball_radius = 0.2):
        super().__init__()
        self.ball_v = ball_v
        self.ball_mass = 1e4
        self.ball_radius = 0.2
        if ball_center is None:
            self.ball_center = ti.Vector.field(3, dtype=float, shape=(1,))
            self.ball_center[0] = ti.Vector([0.5, self.ball_radius, 0.4])
        else:
            self.ball_center = ball_center
        self.ball_v = ti.Vector.field(3, dtype=float, shape=(1,))
        self.ball_v[0] = ti.Vector([0.0, 0.0, 0.0])
        self.inertia = 2/5*self.ball_mass*self.ball_radius*self.ball_radius
        self.time = time
    
    @ti.func
    def Kinematics(self,h:ti.f32):
        self.ball_center[0] += self.ball_v[0] * h
    
    





@ti.data_oriented
class FixedBall(Ball):
    def __init__(self,ball_center = None,ball_radius=0.2):
        super().__init__()
        self.ball_radius = 0.2
        if ball_center is None:
            self.ball_center = ti.Vector.field(3, dtype=float, shape=(1,))
            self.ball_center[0] = ti.Vector([0.5, 0.6, 0.65])
        else:
            self.ball_center = ball_center

@ti.data_oriented
class Pole_x():
    def __init__(self,y:float,z:float,r:float,n=2000):
        self.y = y
        self.z = z
        self.r = r
        self.cylinder_center = ti.Vector.field(3, dtype=float, shape=(n, ))
        for i in range(n):
            self.cylinder_center[i] = [1.5*i/n-0.3,y,z]



@ti.data_oriented
class Ground:
    def __init__(self,height):
        self.height = height

