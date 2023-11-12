import taichi as ti
from Objects import *

class SubPhysicsSystem:
    def __init__(self,object_list):
        self.object_list = object_list

class ClearForcesSystem(SubPhysicsSystem):
    def __init__(self,object_list):
        super().__init__(object_list)
    def evolve(self,h):
        for object in self.object_list:
            object.ClearForces()


        
        
class GravitySystem(SubPhysicsSystem):
    def __init__(self, object_list, g:ti.f32 = 9.8):
        super().__init__(object_list)
        self.gravity_force = ti.Vector([0.0, -g, 0.0])
    
    
    def evolve(self,h:ti.f32):
        for obj in self.object_list:
            self.apply_gravity(obj)
    
    @ti.func
    def apply_gravity(self,object):
        if type(object) == ClothSystem:
            self.apply_gravity_ClothSystem(object)
    
    def apply_gravity_ClothSystem(self,cloth:ClothSystem):
        N = cloth.number
        self._apply_gravity_ClothSystem(N,cloth)
        
    @ti.func
    def _apply_gravity_ClothSystem(self,N,cloth):
        
        for i,j in ti.ndrange(N,N):
            cloth.forces[i,j] += self.gravity_force


class CollideSystem(SubPhysicsSystem):
    def __init__(self, object_list):
        super().__init__(object_list)
        self.num_objects = len(object_list)
        # pop out the ground
        ground_item = None
        for idx, item in enumerate(self.object_list):
            if type(item) == Ground:
                # print("! Ground founded !")
                self.object_list.pop(idx)
                ground_item = item
                break
        if ground_item is not None:
            self.object_list = [ground_item] + self.object_list
                
    
    
    def evolve(self,h):
        for i in range(self.num_objects):
            for j in range(i + 1, self.num_objects):
                self.collide_between_objects(self.object_list[i], self.object_list[j],h)
    
    def collide_between_objects(self, obj1, obj2,h):
        # print(type(obj1),type(obj2))
        if type(obj1) == ClothSystem and (type(obj2) == FixedBall or type(obj2) == FreeBallOnGround):
            self.collide_between_ClothSystemAndBall(obj1,obj2,h)
        elif (type(obj1) == FixedBall or type(obj1) == FreeBallOnGround) and type(obj2) == ClothSystem:
            self.collide_between_ClothSystemAndBall(obj2,obj1,h)
        elif type(obj1) == ClothSystem and type(obj2) == Ground:
            self.collide_between_ClothSystemAndGround(obj1,obj2,h)
        elif type(obj1) == Ground and type(obj2) == ClothSystem:
            self.collide_between_ClothSystemAndGround(obj2,obj1,h)
        elif type(obj1) == ClothSystem and type(obj2) == Pole_x:
            self.collide_between_ClothSystemAndPole_x(obj1,obj2,h)
        elif type(obj1) == Pole_x and type(obj2) == ClothSystem:
            self.collide_between_ClothSystemAndPole_x(obj2,obj1,h)
        
    
    def collide_between_ClothSystemAndBall(self,cloth: ClothSystem,ball:Ball,h):
        if type(ball) == FreeBallOnGround:
            self.collide_between_ClothSystemAndFreeBallOnGround(cloth,ball,h)
        elif type(ball) == FixedBall:
            # print("!")
            self.collide_between_ClothSystemAndFixedBall(cloth,ball,h)
            
    @ti.func
    def collide_between_ClothSystemAndFreeBallOnGround(self, cloth, ball, h:ti.f32):
        N = cloth.number
        for i, j in ti.ndrange(N, N):
            impulse = ti.Vector([0.0, 0.0, 0.0])
            if(ti.Vector.norm(cloth.position[i, j] - ball.ball_center[0]) <= ball.ball_radius):
                R = cloth.position[i, j] - ball.ball_center[0]
                r  = cloth.position[i, j] - ti.Vector([ball.ball_center[0][0], 0.0, ball.ball_center[0][2]])
                v2 = cloth.velocity[i, j]
                v1 = ball.ball_v[0]
                impulse = -(ti.Vector.dot(v2, R) - ti.Vector.dot(v1, R)) / (R.norm() + 1 / ball.inertia * (ti.Vector.dot(r,r)*R.norm() - ti.Vector.dot(R,r)**2 / R.norm()))
                cloth.velocity[i, j] += impulse * R / R.norm()
                ball.ball_center[0] += ball.ball_v[0] * h
                moment_of_impulse = ti.Vector.cross(ti.Vector([-ball.ball_center[0][0]+cloth.position[i,j][0],cloth.position[i,j][1],-ball.ball_center[0][2]+cloth.position[i,j][2]]),impulse)
                delta_omega = moment_of_impulse / ball.inertia
                ball.ball_v[0] += ti.Vector.cross(delta_omega, ti.Vector([0.0, ball.ball_radius, 0.0]))
                ball.ball_v[0][1] = 0.0
                ball.ball_center[0][1] = ball.ball_radius
    
    @ti.func
    def collide_between_ClothSystemAndFixedBall(self,cloth,ball,h:ti.f32):
        N = cloth.number
        center = ball.ball_center[0]
        r = ball.ball_radius
        for i, j in ti.ndrange(N,N):
            if(ti.Vector.norm(cloth.position[i, j] - center) <= r):
                cloth.position[i, j] = center + r*(cloth.position[i, j] - center)/ti.Vector.norm(cloth.position[i, j] - center)
                cloth.velocity[i, j] -= 1.0*ti.Vector.dot(cloth.velocity[i, j], cloth.position[i, j] - center)*(cloth.position[i, j] - center)/ti.Vector.norm(cloth.position[i, j] - center)
    
    @ti.func
    def collide_between_ClothSystemAndGround(self,cloth,ground,h:ti.f32):
        N = cloth.number
        for i, j in ti.ndrange(N,N):
            if (cloth.position[i,j][1] <= ground.height):
                # cloth.position[i,j][1] = ground.height
                cloth.velocity[i,j] = ti.Vector([0.0,0.0,0.0])
    
    @ti.func
    def collide_between_ClothSystemAndPole_x(self,cloth,pole, h:ti.f32):
        N = cloth.number
        yc = pole.y
        zc = pole.z
        r = pole.r
        for i,j in ti.ndrange(N,N):
            if(ti.sqrt((cloth.position[i, j][1]-yc) * (cloth.position[i, j][1]-yc) + (cloth.position[i, j][2]-zc) * (cloth.position[i, j][2]-zc))<r):
                cloth.position[i, j][1] = yc + r*ti.cos(ti.atan2(cloth.position[i, j][2]-zc, cloth.position[i, j][1]-yc))
                cloth.position[i, j][2] = zc + r*ti.sin(ti.atan2(cloth.position[i, j][2]-zc, cloth.position[i, j][1]-yc))
                cloth.velocity[i, j] -= 1.5*ti.Vector.dot(cloth.velocity[i, j], cloth.position[i, j] - ti.Vector([cloth.position[i, j][0], yc, zc]))*(cloth.position[i, j] - ti.Vector([cloth.position[i, j][0], yc, zc]))/ti.Vector.norm(cloth.position[i, j] - ti.Vector([cloth.position[i, j][0], yc, zc]))

class KinematicsSystem(SubPhysicsSystem):
    def __init__(self,object_list):
        super().__init__(object_list)
    def evolve(self,h:ti.f32):
        for object in self.object_list:
            object.Kinematics(h)
        
        











class InternalPhysicsSystem(SubPhysicsSystem):
    def __init__(self,object_list):
        super().__init__(object_list)
    def evolve(self,h:ti.f32):
        for object in self.object_list:
            object.self_evolve(h)

class TotalPhysicsSystem:
    def __init__(self,
                 clear_force_system:ClearForcesSystem = None,
                 kinematicsSystem: KinematicsSystem = None,
                 collisionSystem: CollideSystem = None,
                 physics_system_list: list = None,
                 h:ti.f32=0.00007):
        self.clear_force_system = clear_force_system
        self.kinematicsSystem = kinematicsSystem
        self.collisionSystem = collisionSystem
        self.physics_system_list = physics_system_list
        self.h = h
        
    def evolve(self):
        self.clear_force_system.evolve(self.h)
        
        for physics_system in self.physics_system_list:
            physics_system.evolve(self.h)
        self.kinematicsSystem.evolve(self.h)
        self.collisionSystem.evolve(self.h)
        
        
