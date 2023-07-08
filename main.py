import taichi as ti
import numpy as np
from Objects import *

ti.init(arch=ti.cpu)

N = 64
position = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))
velocity = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))
forces = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))
k_s = 4e3
gamma_s = 3e3
k_d = 4e3 / ti.sqrt(2)
gamma_d = 3e3 / ti.sqrt(2)
k_b = 4e3/2
gamma_b = 3e3/2

num_triangles = (N - 1) * (N - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=N * N)
colors = ti.Vector.field(3, dtype=float, shape=N * N)

# Initialize
@ti.kernel
def initialize():
    for i in range(N):
        for j in range(N):
            position[i, j] = ti.Vector([i / float(N), 2.0, j/float(N)])
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

system = ClothSystem(N, position, velocity, forces)

@ti.kernel
def substep(h:ti.f32):
    system.ClearForces()
    system.ApplyGravity()
    system.ApplySpring(k_s, gamma_s, k_d, gamma_d, k_b, gamma_b, 1 / float(N))
    system.ApplyDamping(6.0)
    for i in range(N):
        for j in range(N):
            system.position[i, j] += system.velocity[i, j]*h
            system.velocity[i, j] += system.forces[i, j]*h
            vertices[i * N + j] = system.position[i, j]
            system.GroundCollision(i,j,0.0)
            system.BallCollision(i,j,0.32,0.6,0.32,0.3)
    #system.time += h

window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
ball_center[0] = [0.32, 0.6, 0.32]

while window.running:
    for i in range(90):
        substep(0.000078)
    camera.position(0, 1.0, 3.0)
    camera.lookat(0.5, 1.0, 0.5)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(vertices,
               indices=indices,
               per_vertex_color=colors,
               two_sided=True)
    scene.particles(ball_center, radius=0.3*0.93, color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    window.show()