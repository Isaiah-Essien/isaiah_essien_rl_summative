import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

grid_size = 10
cell_size = 1


def draw_grid():
    glColor3f(0.8, 0.8, 0.8)
    glBegin(GL_LINES)
    for i in range(grid_size + 1):
        glVertex3f(i * cell_size, 0, 0)
        glVertex3f(i * cell_size, grid_size * cell_size, 0)
        glVertex3f(0, i * cell_size, 0)
        glVertex3f(grid_size * cell_size, i * cell_size, 0)
    glEnd()


def draw_patient(patient_pos, seizure):
    if seizure:
        glColor3f(1.0, 0.0, 0.0)  # Red if seizure
    else:
        glColor3f(0.0, 1.0, 0.0)  # Green if no seizure
    glPointSize(10)
    glBegin(GL_POINTS)
    glVertex3f(patient_pos[0] + 0.5, patient_pos[1] + 0.5, 0)
    glEnd()


def draw_camera(camera_pos):
    glColor3f(0.0, 0.0, 1.0)  # Blue for camera
    glPointSize(15)
    glBegin(GL_POINTS)
    glVertex3f(camera_pos[0] + 0.5, camera_pos[1] + 0.5, 0)
    glEnd()


def visualize_environment(patient_pos, camera_pos, seizure):
    pygame.init()
    display = (600, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluOrtho2D(0, grid_size, 0, grid_size)

    glClearColor(1, 1, 1, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    draw_grid()
    draw_patient(patient_pos, seizure)
    draw_camera(camera_pos)

    pygame.display.flip()

    # Keep window open for static visualization
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    pygame.quit()


if __name__ == "__main__":
    patient_pos = np.array([4, 6])
    camera_pos = np.array([5, 5])
    seizure = True  # or False to visualize both cases
    visualize_environment(patient_pos, camera_pos, seizure)
