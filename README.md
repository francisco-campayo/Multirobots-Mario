# Super Mario Bros — Deep Reinforcement Learning Agent (PPO)

An autonomous Reinforcement Learning agent trained with **Proximal Policy Optimization (PPO)** using **Stable-Baselines3** and **PyTorch (CUDA)** to play *Super Mario Bros* (NES) directly from raw pixel observations.

## Overview & Architecture

* **Algorithm:** PPO (`CnnPolicy`) from Stable-Baselines3.
* **Environment:** `SuperMarioBros-1-1-v0` (`gym-super-mario-bros` + `nes-py`) with `JoypadSpace` (`RIGHT_ONLY` action space).
* **Parallelization:** 12 vectorized environments running concurrently via `SubprocVecEnv` (~417 FPS on GPU).
* **Observation Preprocessing:**
  * Conversion from RGB (`240×256×3`) to grayscale (`84×84`) and pixel normalization to `[0, 1]` using OpenCV.
  * **Frame Stacking (`CustomStackFrames`):** Stacks 4 consecutive frames into a `(4, 84, 84)` tensor to provide temporal context (velocity and motion direction).

## Custom Reward Shaping

Implemented via a custom Gymnasium `Wrapper` (`CustomReward`) to stabilize convergence and prevent the agent from getting stuck:

* **Forward progress & obstacle clearing:** Continuous incentives for advancing in X, jumping while moving forward, and attempting vertical movement when blocked by an obstacle.
* **Score & Flag completion:** Scaled reward for in-game score increments (defeating enemies/items) and a `+300.0` terminal bonus for reaching the flag.
* **Penalties:** `-10.0` for stagnation (5 consecutive steps without meaningful X progress), `-30.0` for losing a life, and `-30.0` for failing to complete the level.

## Training Callbacks & Results

* **Custom Callbacks:**
  * `TrainAndLogCallback`: Periodic model checkpointing.
  * `SaveOnFlagCallback`: Automatically saves the policy whenever any parallel environment reaches the flag.
  * `SaveBestModelCallback`: Tracks and saves the checkpoint achieving the highest episode reward.
* **Performance:** Trained for >1.8M timesteps on CUDA, achieving a rolling mean episode reward (`ep_rew_mean`) of **740** and a peak single-episode reward of **1363.92**.

## Tech Stack

* **Python 3.8** (Conda environment)
* **PyTorch** (CUDA) & **Stable-Baselines3**
* **Gymnasium**, `gym-super-mario-bros`, `nes-py`
* **OpenCV** & **NumPy**

## Usage

1. Open `[mario_rl_M4.ipynb]` in Jupyter / VS Code using the configured Conda kernel.
2. Run the environment setup and wrapper cells (`CustomStackFrames`, `CustomReward`).
3. Execute the training cell to train from scratch, or load the pre-trained weights (`mario_final_model.zip`) in the evaluation cell to watch the agent play.
