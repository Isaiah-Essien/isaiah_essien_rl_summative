import pygame
import numpy as np
import imageio

# Grid settings
grid_size = 10
cell_size = 70
window_size = grid_size * cell_size

colors = {
    'bg': (25, 25, 112),
    'grid': (70, 130, 180),
    'patient_normal': (60, 179, 113),
    'patient_seizure': (220, 20, 60),
    'drone': (255, 215, 0),
    'shadow': (47, 79, 79),
}


def draw_grid(surface):
    for x in range(grid_size):
        for y in range(grid_size):
            rect = pygame.Rect(x*cell_size, y*cell_size, cell_size, cell_size)
            pygame.draw.rect(surface, colors['grid'], rect, 2, border_radius=8)


def draw_patient(surface, patient_pos, seizure):
    patient_color = colors['patient_seizure'] if seizure else colors['patient_normal']
    patient_pos_pix = ((patient_pos + 0.5) * cell_size).astype(int)
    pygame.draw.circle(surface, patient_color, patient_pos_pix, cell_size//3)


def draw_drone(surface, drone_pos):
    drone_pos_pix = ((drone_pos + 0.5) * cell_size).astype(int)
    pygame.draw.rect(surface, colors['drone'], (*drone_pos_pix -
                     cell_size//4, cell_size//2, cell_size//2), border_radius=10)

    for angle in [45, 135, 225, 315]:
        offset = np.array([np.cos(np.radians(angle)), np.sin(
            np.radians(angle))]) * cell_size//2.5
        rotor_pos = drone_pos_pix + offset.astype(int)
        pygame.draw.circle(surface, colors['drone'], rotor_pos, cell_size//10)


def visualize_environment():
    pygame.init()
    surface = pygame.Surface((window_size, window_size))

    frames = []

    drone_pos = np.array([5, 5])
    patient_pos = np.array([4, 6])

    for _ in range(100):
        seizure = np.random.rand() < 0.05

        patient_pos += np.random.choice([-1, 0, 1], size=2)
        patient_pos = np.clip(patient_pos, 0, grid_size-1)

        drone_pos += np.random.choice([-1, 0, 1], size=2)
        drone_pos = np.clip(drone_pos, 0, grid_size-1)

        surface.fill(colors['bg'])
        draw_grid(surface)
        draw_patient(surface, patient_pos, seizure)
        draw_drone(surface, drone_pos)

        frame = pygame.surfarray.array3d(surface)
        frames.append(np.transpose(frame, (1, 0, 2)))

    imageio.mimsave('environment_simulation.gif', frames, duration=0.1)
    pygame.quit()


if __name__ == "__main__":
    visualize_environment()
