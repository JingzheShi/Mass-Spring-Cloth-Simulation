import taichi as ti

@ti.data_oriented
class ClothSystem:
    def __init__(self, number, position, velocity, forces, time = 0.0):
        self.number = number
        self.position = position
        self.velocity = velocity
        self.forces = forces
        self.time = time

    @ti.func
    def ClearForces(self):
        for i in range(self.number):
            for j in range(self.number):
                self.forces[i, j] = ti.Vector([0.0,0.0,0.0])
    
    @ti.func
    def ApplyGravity(self):
        for i in range(self.number):
            for j in range(self.number):
                self.forces[i, j] += ti.Vector([0.0,-9.8,0.0])

    @ti.func
    def ApplySpring(self, k_s:ti.f32, gamma_s:ti.f32, k_d:ti.f32, gamma_d:ti.f32, k_b:ti.f32, gamma_b:ti.f32, l:ti.f32):
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
    def ApplyDamping(self, damp:ti.f32):
        for i in range(self.number):
            for j in range(self.number):
                self.forces[i, j] -= damp*self.velocity[i, j]

    @ti.func
    def GroundCollision(self, i:int, j:int, ground:ti.f32):
        if(self.position[i, j][1] <= ground):
            self.position[i, j][1] = ground
            self.velocity[i, j][1] -= 1.3*ti.Vector.dot(self.velocity[i, j], ti.Vector([0.0, 1.0, 0.0]))

    @ti.func
    def BallCollision(self, i:int, j:int, xc:ti.f32, yc:ti.f32, zc:ti.f32, r:ti.f32):
        center = ti.Vector([xc, yc, zc])
        if(ti.Vector.norm(self.position[i, j] - center) <= r):
            self.position[i, j] = center + r*(self.position[i, j] - center)/ti.Vector.norm(self.position[i, j] - center)
            self.velocity[i, j] -= 1.5*ti.Vector.dot(self.velocity[i, j], self.position[i, j] - center)*(self.position[i, j] - center)/ti.Vector.norm(self.position[i, j] - center)

    @ti.func
    def PoleCollision_z(self, i:int, j:int, xc:ti.f32, yc:ti.f32, r:ti.f32):
        if(ti.sqrt((self.position[i, j][0]-xc) * (self.position[i, j][0]-xc) + (self.position[i, j][1]-yc) * (self.position[i, j][1]-yc))<r):
            self.position[i, j][0] = xc + r*ti.cos(ti.atan2(self.position[i, j][1]-yc, self.position[i, j][0]-xc))
            self.position[i, j][1] = yc + r*ti.sin(ti.atan2(self.position[i, j][1]-yc, self.position[i, j][0]-xc))
            self.velocity[i, j] -= 1.5*ti.Vector.dot(self.velocity[i, j], self.position[i, j] - ti.Vector([xc, yc, self.position[i, j][2]]))*(self.position[i, j] - ti.Vector([xc, yc, self.position[i, j][2]]))/ti.Vector.norm(self.position[i, j] - ti.Vector([xc, yc, self.position[i, j][2]]))

    @ti.func
    def PoleCollision_x(self, i:int, j:int, yc:ti.f32, zc:ti.f32, r:ti.f32):
        if(ti.sqrt((self.position[i, j][1]-yc) * (self.position[i, j][1]-yc) + (self.position[i, j][2]-zc) * (self.position[i, j][2]-zc))<r):
            self.position[i, j][1] = yc + r*ti.cos(ti.atan2(self.position[i, j][2]-zc, self.position[i, j][1]-yc))
            self.position[i, j][2] = zc + r*ti.sin(ti.atan2(self.position[i, j][2]-zc, self.position[i, j][1]-yc))
            self.velocity[i, j] -= 1.5*ti.Vector.dot(self.velocity[i, j], self.position[i, j] - ti.Vector([self.position[i, j][0], yc, zc]))*(self.position[i, j] - ti.Vector([self.position[i, j][0], yc, zc]))/ti.Vector.norm(self.position[i, j] - ti.Vector([self.position[i, j][0], yc, zc]))
                