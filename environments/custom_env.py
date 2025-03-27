import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class EpilepsyDetectionEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode='rgb_array'):
        super().__init__()
        self.grid_size = 10
        self.window_size = 700
        self.cell_size = self.window_size // self.grid_size
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            0, 10, shape=(5,), dtype=np.float32)

        self.reset()

        if render_mode == 'rgb_array':
            pygame.init()
            self.window = pygame.Surface((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.camera_pos = np.array([5, 5])
        self.patient_pos = np.random.randint(0, self.grid_size, size=2)
        self.seizure = False
        self.time_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.camera_pos[0], self.camera_pos[1], 1,
                         self.patient_pos[0], self.patient_pos[1]], dtype=np.float32)

    def step(self, action):
        reward = -0.1
        self.time_step += 1

        self.patient_pos += np.random.choice([-1, 0, 1], size=2)
        self.patient_pos = np.clip(self.patient_pos, 0, self.grid_size-1)
        self.seizure = np.random.rand() < 0.05

        if action == 0:
            self.camera_pos[0] -= 1
        elif action == 1:
            self.camera_pos[0] += 1
        elif action == 4:
            self.camera_pos[1] -= 1
        elif action == 5:
            self.camera_pos[1] += 1
        self.camera_pos = np.clip(self.camera_pos, 0, self.grid_size-1)

        if self.seizure and np.array_equal(self.camera_pos, self.patient_pos):
            reward += 10
        elif self.seizure:
            reward -= 10
        elif not self.seizure and np.array_equal(self.camera_pos, self.patient_pos):
            reward -= 5

        done = self.time_step >= 200
        return self._get_obs(), reward, done, False, {}

    def render(self):
        colors = {
            'bg': (25, 25, 112),
            'grid': (70, 130, 180),
            'patient_normal': (60, 179, 113),
            'patient_seizure': (220, 20, 60),
            'drone': (255, 215, 0),
            'shadow': (47, 79, 79),
        }
        self.window.fill(colors['bg'])

        # Enhanced grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(
                    x*self.cell_size, y*self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(
                    self.window, colors['grid'], rect, 2, border_radius=8)

        # Patient shadow (adds depth)
        shadow_pos = (
            (self.patient_pos + [0.65, 0.75]) * self.cell_size).astype(int)
        pygame.draw.circle(
            self.window, colors['shadow'], shadow_pos, self.cell_size//5)

        # Patient (clearly visible)
        patient_pos = ((self.patient_pos + 0.5) * self.cell_size).astype(int)
        patient_color = colors['patient_seizure'] if self.seizure else colors['patient_normal']
        pygame.draw.circle(self.window, patient_color,
                           patient_pos, self.cell_size//3)

        # Drone shadow (enhanced depth)
        drone_shadow = (
            (self.camera_pos + [0.55, 0.8]) * self.cell_size).astype(int)
        pygame.draw.ellipse(
            self.window, colors['shadow'], (*drone_shadow, self.cell_size//2, self.cell_size//5))

        # Drone body (clearly drone-like)
        drone_pos = ((self.camera_pos + 0.5) * self.cell_size).astype(int)
        pygame.draw.rect(self.window, colors['drone'], (
            *drone_pos-self.cell_size//4, self.cell_size//2, self.cell_size//2), border_radius=10)

        # Drone rotors (realistic drone representation)
        for angle in [45, 135, 225, 315]:
            offset = np.array([np.cos(np.radians(angle)), np.sin(
                np.radians(angle))]) * self.cell_size//2.5
            rotor_pos = drone_pos + offset.astype(int)
            pygame.draw.circle(
                self.window, colors['drone'], rotor_pos, self.cell_size//10)

        if self.render_mode == 'rgb_array':
            return np.transpose(pygame.surfarray.array3d(self.window), axes=(1, 0, 2))

    def close(self):
        pygame.quit()
