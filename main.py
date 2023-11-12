import taichi as ti
from Objects import *
from Physics import *
ti.init(arch = ti.cpu)
# ti.init(arch=ti.cuda, device_memory_GB=2.0)

N = 128
position = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))
velocity = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))
forces = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))
k_s = 5e4
gamma_s = 3e3
k_d = 5e4 / ti.sqrt(2)
gamma_d = 3e3 / ti.sqrt(2)
k_b = 5e4/2
gamma_b = 3e3/2
damp = 5.0

num_triangles = (N - 1) * (N - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=N * N)
colors = ti.Vector.field(3, dtype=float, shape=N * N)

# Initialize
@ti.kernel
def initialize():
    for i, j in ti.ndrange(N, N):
            position[i, j] = ti.Vector([i / float(N) + ti.random()*0.02, 2.0 + ti.random()*0.02, j/float(N) + ti.random()*0.02])
            velocity[i, j] = ti.Vector([0.0, 0.0, 0.0])
            forces[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(N - 1, N - 1):
        quad_id = (i * (N - 1)) + j
        # 1st triangle of the square
        indices[quad_id * 6 + 0] = i * N + j
        indices[quad_id * 6 + 1] = (i + 1) * N + j
        indices[quad_id * 6 + 2] = i * N + (j + 1)
        # 2nd triangle of the square
        indices[quad_id * 6 + 3] = (i + 1) * N + j + 1
        indices[quad_id * 6 + 4] = i * N + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * N + j

    for i, j in ti.ndrange(N, N):
        if (i // 4 + j // 4) % 2 == 0:
            colors[i * N + j] = (0.22, 0.72, 0.52)
        else:
            colors[i * N + j] = (1, 0.334, 0.52)

@ti.kernel
def initialize_vertices():
    for i, j in ti.ndrange(N, N):
        vertices[i * N + j] = position[i, j]

initialize()
initialize_mesh_indices()
initialize_vertices()




clothSystem = ClothSystem(N, position, velocity, forces, k_s,
                     gamma_s,k_d,gamma_d,k_b,gamma_b,damp)
freeBallOnGround = FreeBallOnGround()
fixedBall = FixedBall()
pole_x = Pole_x(0.6,0.65,0.08)
ground = Ground(0.0)

clearForceSystem = ClearForcesSystem([clothSystem])
internalPhysicsSystem = InternalPhysicsSystem([clothSystem])
gravitySystem = GravitySystem([clothSystem])
collideSystem = CollideSystem([clothSystem,fixedBall,ground,pole_x,freeBallOnGround])
kinematicsSystem = KinematicsSystem([clothSystem,freeBallOnGround])


physicsSystem = TotalPhysicsSystem(clearForceSystem,kinematicsSystem,collideSystem,[gravitySystem,internalPhysicsSystem],h=0.00007)



@ti.kernel
def substep():
    physicsSystem.evolve()


window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()



while window.running:
    for i in range(40):
        substep()
    camera.position(2.5, 1.0, 2.5)
    camera.lookat(0.5, 1.0, 0.5)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(clothSystem.vertices,
               indices=indices,
               per_vertex_color=colors,
               two_sided=True)
    scene.particles(fixedBall.ball_center, radius=fixedBall.ball_radius*0.93, color=(0.5, 0.42, 0.8))
    scene.particles(pole_x.cylinder_center, radius=pole_x.r*0.93, color=(0.4, 0.62, 0.7))
    scene.particles(freeBallOnGround.ball_center, radius=freeBallOnGround.ball_radius*0.93, color=(0.6, 0.32, 0.6))
    canvas.scene(scene)
    window.show()