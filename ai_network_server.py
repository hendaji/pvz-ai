import socket
import json
import numpy as np
import time
import os
import re
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure


class ForceTensorBoardCallback(BaseCallback):
    def __init__(self, base_log_dir, verbose=0):
        super(ForceTensorBoardCallback, self).__init__(verbose)
        self.episode_reward = 0.0
        self.base_log_dir = base_log_dir

        # Определяем максимальный номер существующих папок PPO_*
        existing = [d for d in os.listdir(base_log_dir)
                    if d.startswith("PPO_") and os.path.isdir(os.path.join(base_log_dir, d))]
        max_num = 0
        for name in existing:
            match = re.match(r"PPO_(\d+)", name)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num
        self.run_idx = max_num + 1

        # Создаём логгер ОДИН РАЗ для всей сессии
        self.current_run_name = f"PPO_{self.run_idx}"
        self.tb_logger = configure(
            os.path.join(self.base_log_dir, self.current_run_name),
            ["stdout", "tensorboard"]
        )
        print(f"[ИИ СЕРВЕР] Создан трей: {self.current_run_name}")

    def _on_step(self) -> bool:
        reward = self.locals["rewards"] if isinstance(self.locals["rewards"], (list, np.ndarray)) else self.locals["rewards"]
        self.episode_reward += float(np.sum(reward))

        if self.n_calls % 10 == 0:
            self.tb_logger.record("ai_metrics/current_accumulated_reward", self.episode_reward)

        if self.locals["dones"] if isinstance(self.locals["dones"], (list, np.ndarray)) else self.locals["dones"]:
            self.tb_logger.record("ai_metrics/final_match_reward", self.episode_reward)
            print(f"\n[ИИ СЕРВЕР] Эпизод завершён. Награда: {self.episode_reward:.2f}")
            self.episode_reward = 0.0
        return True


class PureNetworkEnv(gym.Env):
    def __init__(self):
        super(PureNetworkEnv, self).__init__()
        self.action_space = spaces.Discrete(91)
        self.observation_space = spaces.Box(low=0, high=800, shape=(51,), dtype=np.float32)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 5005))
        self.sock.settimeout(0.3)

        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.prev_plants_count = 0
        self.last_update_time = time.time()
        self.prev_zombie_count = 0
        self.repeat_count = 0
        self.last_action = None

        print("[ИИ СЕРВЕР] Запущен. Жду подключения игры...")
        print("[ИИ СЕРВЕР] Действия: 0-44 горох, 45-89 подсолнух, 90 — ОЖИДАНИЕ")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.prev_plants_count = 0
        self.last_update_time = time.time()
        self.prev_zombie_count = 0
        self.repeat_count = 0
        self.last_action = None

        try:
            data, addr = self.sock.recvfrom(4096)
            game_data = json.loads(data.decode('utf-8'))
            obs_array = np.array(game_data["grid"] + game_data["zombies"] + [game_data["sun"]], dtype=np.float32)
        except:
            obs_array = np.zeros(51, dtype=np.float32)
        return obs_array, {}

    def step(self, action):
        obs_array = np.zeros(51, dtype=np.float32)
        reward = 0.0
        terminated = False

        try:
            data, addr = self.sock.recvfrom(4096)
            game_data = json.loads(data.decode('utf-8'))

            if "status" in game_data and game_data["status"] == "RESET":
                print("[ИИ СЕРВЕР] Получен RESET-сигнал! TERMINATED")
                terminated = True
                return obs_array, 0.0, terminated, False, {}

            if "grid" not in game_data:
                print("[ИИ СЕРВЕР] Ошибка: нет ключа 'grid', завершаем эпизод")
                terminated = True
                return obs_array, 0.0, terminated, False, {}

            grid_flat = np.array(game_data["grid"], dtype=np.float32)
            zombies = np.array(game_data["zombies"], dtype=np.float32)
            sun = float(game_data["sun"])

            current_grid = grid_flat.copy()
            obs_array = np.concatenate([grid_flat, zombies, [sun]])

            current_plants_count = int(np.sum(grid_flat == 1.0))
            if self.prev_plants_count > 2 and current_plants_count == 0:
                print("[ИИ СЕРВЕР] Перезапуск игры (детект по растениям)! TERMINATED")
                terminated = True
                self.prev_plants_count = 0
                return obs_array, reward, terminated, False, {}
            self.prev_plants_count = current_plants_count

            if np.any(zombies < 80):
                print("[ИИ СЕРВЕР] Зомби прорвались! Проигрыш. TERMINATED")
                terminated = True
                reward -= 50.0
                self.prev_zombie_count = 0
                return obs_array, reward, terminated, False, {}

            # ========== ОЖИДАНИЕ ==========
            if action == 90:
                if sun < 50:
                    reward += 1.0
                    print(f"[ИИ] Ожидание, копим солнце: +1.0")
                elif sun < 100:
                    reward += 0.5
                    print(f"[ИИ] Ожидание, солнца достаточно: +0.5")
                else:
                    reward -= 2.0
                    print(f"[ИИ] Ожидание при солнце {sun:.0f}: -2.0")
                
                self.repeat_count = 0
                reward = np.clip(reward, -50.0, 100.0)
                return obs_array, reward, terminated, False, {}

            # ========== ПОВТОРЫ ==========
            if self.last_action == action:
                self.repeat_count += 1
            else:
                self.repeat_count = 0
                self.last_action = action

            if self.repeat_count > 3:
                reward -= 30.0
                print(f"[ИИ] Штраф за повтор ({self.repeat_count} раз): -30")

            # ========== ОТПРАВКА ДЕЙСТВИЯ ==========
            cmd_message = json.dumps({"action": int(action)}).encode('utf-8')
            self.cmd_sock.sendto(cmd_message, ("127.0.0.1", 5006))

            time.sleep(0.02)

            try:
                data2, addr2 = self.sock.recvfrom(4096)
                game_data2 = json.loads(data2.decode('utf-8'))

                if "grid" not in game_data2:
                    print("[ИИ СЕРВЕР] Ошибка: нет ключа 'grid' в ответе, завершаем эпизод")
                    terminated = True
                    return obs_array, 0.0, terminated, False, {}

                new_grid = np.array(game_data2["grid"], dtype=np.float32)
                new_sun = float(game_data2["sun"])

                target_cell = action if action < 45 else (action - 45)
                was_empty = (current_grid[target_cell] == 0)
                is_occupied = (new_grid[target_cell] == 1)
                success = was_empty and is_occupied

                # ========== ПОСАДКА РАСТЕНИЙ ==========
                grid_2d = new_grid.reshape(5, 9)
                # Считаем, сколько ВСЕГО растений сейчас на поле
                total_plants = int(np.sum(new_grid == 1.0)) 

                if action < 45:  # ГОРОХОСТРЕЛ (Защита)
                    if success:
                        if sun >= 100:
                            # Огромная награда в долгую, чтобы горох был выгоднее двух подсолнухов
                            reward += 25.0 
                            print(f"[ИИ] Посажен горох (Инвестиция в оборону): +25")
                        else:
                            reward -= 10.0
                    else:
                        reward -= 1.0

                else:  # ПОДСОЛНУХ (Экономика)
                    if success:
                        if sun >= 50:
                            # Контрим жадный спам: если солнца уже дофига (копим на горох),
                            # а подсолнухов слишком много, режем награду за них в ноль
                            if total_plants > 3 and sun > 250: 
                                reward += 1.0  
                                print(f"[ИИ] Посажен подсолнух (Лимит экономики достигнут): +1")
                            else:
                                reward += 8.0  # Обычная награда (меньше, чем за горох)
                                print(f"[ИИ] Посажен подсолнух: +8")
                        else:
                            reward -= 10.0
                    else:
                        reward -= 1.0

                # ========== НАГРАДА ЗА ВЫЖИВАНИЕ ==========
                # Если зомби далеко, плавно поощряем за удержание позиций
                if np.all(zombies > 150):
                    reward += 0.5
                else:
                    reward += 0.05 # Если прижали к дому, за "просто сидение" очков почти не даем

                # ========== БОНУС ЗА УБИЙСТВО ЗОМБИ ==========
                current_zombie_count = len(zombies)
                killed = self.prev_zombie_count - current_zombie_count
                if killed > 0:
                    # Поднимаем ценность убийства, чтобы окупить затраты на горохострелы
                    reward += killed * 40.0 
                    print(f"[ИИ] Убито {killed} зомби: +{killed * 40}")

                # ========== АНАЛИЗ РЯДОВ ==========
                row_counts = np.sum(grid_2d, axis=1)

                # Хвалим, если ИИ додумался закрыть все 5 линий хотя бы по одному растению
                if np.all(row_counts >= 1):
                    reward += 10.0 
                    print(f"[ИИ] Все ряды заняты: +10")

                # Штраф за стакинг (если лепит больше 4 растений в один ряд, забывая про другие)
                max_in_row = np.max(row_counts)
                if max_in_row > 4:
                    penalty = (max_in_row - 4) * 8.0
                    reward -= penalty
                    print(f"[ИИ] Стакинг в одном ряду: -{penalty}")

                # ========== ОБНОВЛЕНИЕ СЧЁТЧИКОВ ==========
                self.prev_zombie_count = current_zombie_count
                self.prev_plants_count = int(np.sum(new_grid == 1.0))


            except socket.timeout:
                pass

        except socket.timeout:
            reward -= 1.0
        except Exception as e:
            print(f"[ИИ СЕРВЕР] Ошибка: {e}")
            reward -= 2.0
            print("[ИИ СЕРВЕР] Принудительное завершение эпизода из-за ошибки!")
            terminated = True

        reward = np.clip(reward, -50.0, 100.0)
        return obs_array, reward, terminated, False, {}


if __name__ == "__main__":
    raw_env = PureNetworkEnv()
    env = Monitor(raw_env)
    base_log_dir = "./pvz_tensorboard_logs/"
    
    custom_callback = ForceTensorBoardCallback(base_log_dir=base_log_dir)

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/",
        name_prefix="pvz_ppo_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    callback = CallbackList([custom_callback, checkpoint_callback])

    try:
        model = PPO.load("pvz_ppo_brain_v1", env=env)
        print("[ИИ СЕРВЕР] Модель загружена.")
    except:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=512,
            gamma=0.999,
            batch_size=64,
            ent_coef=0.02,
        )
        print("[ИИ СЕРВЕР] Создана новая модель.")

    model.set_logger(custom_callback.tb_logger)

    print("[ИИ СЕРВЕР] Начинаем бесконечное обучение (до прерывания Ctrl+C)...")
    try:
        model.learn(total_timesteps=int(1e9), callback=callback)
    except KeyboardInterrupt:
        print("\n[СТОП] Обучение прервано пользователем. Сохраняем модель...")
    finally:
        model.save("pvz_ppo_brain_v1")
        print("[ИИ СЕРВЕР] Модель сохранена.")
