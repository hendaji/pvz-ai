import socket
import json
import numpy as np
import time
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

class PureNetworkEnv(gym.Env):
    def __init__(self):
        super(PureNetworkEnv, self).__init__()
        self.action_space = spaces.Discrete(45)
        self.observation_space = spaces.Box(low=0, high=800, shape=(51,), dtype=np.float32)
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 5005))
        self.sock.settimeout(0.2)
        print("[ИИ СЕРВЕР] Чистый цифровой мозг запущен. Ожидание старта боя...")

    def reset(self, seed=None, options=None):
        return np.zeros(51, dtype=np.float32), {}

    def step(self, action):
        obs_array = np.zeros(51, dtype=np.float32)
        reward = 0.1
        terminated = False
        
        try:
            # Читаем данные от игры
            data, addr = self.sock.recvfrom(4096)
            game_data = json.loads(data.decode('utf-8'))
            
            obs_array = np.array(game_data["grid"] + game_data["zombies"] + [game_data["sun"]], dtype=np.float32)
            
            # ОТПРАВЛЯЕМ КОМАНДУ ОБРАТНО В ИГРУ ПО СЕТИ
            # Никаких кликов по экрану, просто шлем JSON пакет
            cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            cmd_message = json.dumps({"action": int(action)}).encode('utf-8')
            cmd_sock.sendto(cmd_message, ("127.0.0.1", 5006))
            cmd_sock.close()
            
            reward += 1.0 # Похвала за отправку команды
            
            if np.any(np.array(game_data["zombies"]) <= 50):
                terminated = True
                
        except socket.timeout:
            pass

        return obs_array, reward, terminated, False, {}

if __name__ == "__main__":
    env = PureNetworkEnv()
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=32)
    model.learn(total_timesteps=15000)
