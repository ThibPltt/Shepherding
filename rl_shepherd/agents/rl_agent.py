from stable_baselines3 import PPO,A2C,TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
import time
import os

def train_rl_agent_ppo_mlp(env,eval_env, timesteps=2000000,checkpoint_dir=None,criculam_learning=False):
    # model = PPO("MlpPolicy", env, verbose=1,device='cpu', tensorboard_log="./ppo_shepherd_logs/")

    run_name = f"sheep{env.n_sheep}_obst{int(env.obstacle_radius*10)}_{time.strftime('%Y%m%d-%H%M%S')}"
    log_dir = f"./logs/ppo/{run_name}/"
    os.makedirs(log_dir, exist_ok=True)

    print(f"Logging and saving best models to {log_dir}")

    if checkpoint_dir is not None:
        print(f"Loading checkpoint from {checkpoint_dir}...")
        model=PPO.load(checkpoint_dir,
                env=env,
                n_steps=2048,
                ent_coef=0.003,
                learning_rate=3e-4,
                gamma=0.99,
                verbose=1,
                tensorboard_log=log_dir
                )
        if criculam_learning:
            # Reinitialize optimizer and PPO internals
            print("Reinitializing optimizer and PPO internals for curriculum learning...")
            model._setup_model()
            model.learning_rate = 1e-4    
    else:
        print("No checkpoint provided, training from scratch.")
        model = PPO("MlpPolicy",
                    env=env,
                    n_steps=2048,
                    ent_coef=0.003,
                    learning_rate=3e-4,
                    gamma=0.99,
                    verbose=1,
                    tensorboard_log=log_dir
                    )
    
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                log_path=log_dir, eval_freq=20000,n_eval_episodes=20,
                                deterministic=True, render=False)
    model.learn(total_timesteps=timesteps,callback=eval_callback)
    return model

def train_rl_agent_td3_mlp(env, eval_env, timesteps=2000000,checkpoint_dir=None,criculam_learning=False):
    run_name = f"sheep{env.n_sheep}_obst{int(env.obstacle_radius*10)}_{time.strftime('%Y%m%d-%H%M%S')}"
    log_dir = f"./logs/td3/{run_name}/"
    os.makedirs(log_dir, exist_ok=True)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = TD3("MlpPolicy", env, verbose=1,tensorboard_log=log_dir, action_noise=action_noise)
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                log_path=log_dir, eval_freq=20000,n_eval_episodes=20,
                                deterministic=True, render=False)
    model.learn(total_timesteps=timesteps,callback=eval_callback)
    return model

def train_rl_agent_sac_mlp(env, eval_env, timesteps=2000000, checkpoint_dir=None, criculam_learning=False):
    run_name = f"sheep{env.n_sheep}_obst{int(env.obstacle_radius*10)}_{time.strftime('%Y%m%d-%H%M%S')}"
    log_dir = f"./logs/sac/{run_name}/"
    os.makedirs(log_dir, exist_ok=True)

    print(f"Logging and saving best models to {log_dir}")

    if checkpoint_dir is not None:
        print(f"Loading checkpoint from {checkpoint_dir}...")
        model = SAC.load(checkpoint_dir,
                env=env,
                verbose=1,
                tensorboard_log=log_dir
                )
        if criculam_learning:
            print("Reinitializing optimizer and SAC internals for curriculum learning...")
            model.lr_schedule = lambda _: 3e-4 
            model._setup_model()   
    else:
        print("No checkpoint provided, training from scratch.")
        # SAC n'a pas besoin de bruit d'action externe (NormalActionNoise) car il utilise l'entropie
        model = SAC("MlpPolicy",
                    env=env,
                    learning_rate=3e-4,
                    buffer_size=100000,
                    batch_size=256,
                    learning_starts=1_000,
                    gamma=0.99,
                    tau=0.005,
                    ent_coef='auto', # Important pour SAC : gestion automatique de l'exploration
                    verbose=0,
                    tensorboard_log=log_dir
                    )
    
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                log_path=log_dir, eval_freq=20000, n_eval_episodes=20,
                                deterministic=True, render=False)
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    return model