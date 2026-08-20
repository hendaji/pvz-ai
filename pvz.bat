@echo off
title AI MAIN CONTROLLER

echo [1/3] Запуск TensorBoard...
start "TensorBoard" cmd /k "cd H:\Python_pvz\PythonPlantsVsZombies-master && py -m tensorboard.main --logdir=pvz_tensorboard_logs"

timeout /t 2

echo [2/3] Запуск Сервера Нейросети...
start "AI Network Server" cmd /k "cd H:\Python_pvz\PythonPlantsVsZombies-master && py "ai_network_server.py""

timeout /t 2

echo [3/3] Запуск Игры (Pygame)...
start "PvZ Game Engine" cmd /k "cd H:\Python_pvz\PythonPlantsVsZombies-master && py main.py"

echo Все окна запущены! Удачного обучения нейронки.
pause
