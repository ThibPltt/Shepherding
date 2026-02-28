from agents.MLP_QN import VectorDQNAgent, train_vector_dqn, VectorDQNAgent, N_ACTIONS
from envs.shepherd_env import ShepherdEnv
from agents.rl_agent import train_rl_agent_ppo_mlp, train_rl_agent_td3_mlp, train_rl_agent_sac_mlp
import torch
import argparse
import time
from datetime import timedelta

# --- Fonction d'estimation du temps ---
def estimate_training_time(env, total_timesteps, algo_name):
    print(f"\n Calcul de l'estimation pour {algo_name} ({total_timesteps} steps)...")
    
    # 1. On mesure la vitesse pure de l'environnement
    start_bench = time.time()
    n_bench_steps = 500 
    
    # On fait un reset et quelques steps bidons
    env.reset()
    for _ in range(n_bench_steps):
        action = env.action_space.sample() if hasattr(env, 'action_space') else 0
        _, _, done, _ = env.step(action)
        if done: env.reset()
        
    bench_duration = time.time() - start_bench
    if bench_duration == 0: bench_duration = 0.001
    
    raw_sps = n_bench_steps / bench_duration 
    
    # 2. Facteur de correction (L'entraînement est ~3.5x plus lent que la simu pure)
    training_factor = 3.5 
    estimated_sps = raw_sps / training_factor
    
    total_seconds = total_timesteps / estimated_sps
    eta = timedelta(seconds=int(total_seconds))
    
    print(f"   -> Vitesse env mesurée : {int(raw_sps)} steps/sec")
    print(f"   -> Vitesse d'entraînement estimée : ~{int(estimated_sps)} steps/sec")
    print(f"   ->  TEMPS ESTIMÉ : {str(eta)} (Heures:Minutes:Secondes)\n")
    print("-" * 50)


# --- Arguments ---
parser = argparse.ArgumentParser(description="Train RL agents for ShepherdEnv.")
parser.add_argument("-a", "--algorithm", type=str,
                    choices=["td3", "dqn", "ppo", "sac", "all"], default="ppo",
                    help="Choose the algorithm to train: 'td3', 'dqn', 'ppo', 'sac', or 'all'. Default is 'ppo'.")

parser.add_argument("-s", "--num_sheep", type=int, default=1, help="Number of sheep.")
parser.add_argument("-m", "--max_steps", type=int, default=500, help="Max steps per episode.")
parser.add_argument("-r", "--obstacle_radius", type=float, default=0.0, help="Obstacle radius.")
parser.add_argument("-g", "--goal_radius", type=float, default=0.7, help="Goal radius.")
parser.add_argument("-c", "--checkpoint_dir", type=str, default=None, help="Checkpoint path.")
parser.add_argument("-cl", "--criculam_learning", type=bool, default=True, help="Curriculum learning.")
parser.add_argument("-v", "--vitesse", type=float, default=0.0, help="Sheep speed.")

args = parser.parse_args()

# --- Init Environnements ---
print("Initialisation des environnements...")
env = ShepherdEnv(n_sheep=args.num_sheep,
                max_steps=args.max_steps,
                obstacle_radius=args.obstacle_radius,
                goal_radius=args.goal_radius,
                sheep_speed=args.vitesse)

eval_env = ShepherdEnv(n_sheep=args.num_sheep,
                        max_steps=args.max_steps,
                        obstacle_radius=args.obstacle_radius,
                        goal_radius=args.goal_radius,
                        sheep_speed=args.vitesse)

global_start_time = time.time()

# ================= DQN =================
if args.algorithm in ["dqn", "all"]:
    try:
        # Estimation approx pour DQN (basée sur episodes * max_steps potentiels)
        estimate_training_time(env, 200 * args.max_steps, "DQN (Approx)")

        print(f"Training with MLP_QN (#sheep: {env.n_sheep})...")
        dqn_start = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        save_name = f"models/dqn_sheep{env.n_sheep}_obst{int(args.obstacle_radius*10)}.pth"

        # Récupération de la dimension du vecteur d'observation
        obs_dim = env.observation_space.shape[0]
        
        agent = VectorDQNAgent(
                    obs_dim=obs_dim, # <--- IMPORTANT: On passe la taille du vecteur
                    n_actions=N_ACTIONS,
                    lr=1e-3,  # On peut augmenter un peu le LR pour du MLP
                    gamma=0.99,
                    eps_start=1.0,
                    device=device
                )
        
        # Logique de chargement de checkpoint spécifique DQN
        if args.checkpoint_dir is not None:
             print(f"Chargement du modèle DQN depuis {args.checkpoint_dir}...")
             agent.q_net.load_state_dict(torch.load(args.checkpoint_dir, map_location=device))
             agent.update_target()
             agent.eps_start = 0.3 # Si on reprend un entrainement, on explore moins
        
        model = train_vector_dqn(
                    env=env,
                    eval_env=eval_env,
                    agent=agent,
                    episodes=1000,
                    batch_size=256,
                    target_update=1000,
                    eval_every=5,
                    eval_episodes=5,
                    train_freq=4,
                    save_path=save_name
                )
        
        print(f" DQN Finished in {str(timedelta(seconds=int(time.time() - dqn_start)))}")
        
    except Exception as e:
        print(f"MLP_QN training failed: {e}")


# ================= PPO =================
if args.algorithm in ["ppo", "all"]:
    try:
        TIMESTEPS = 2000000
        estimate_training_time(env, TIMESTEPS, "PPO")

        print(f"Training with PPO algorithm (#sheep: {env.n_sheep})...")
        ppo_start = time.time()
        
        model = train_rl_agent_ppo_mlp(env, eval_env, timesteps=TIMESTEPS,
                                     checkpoint_dir=args.checkpoint_dir,
                                     criculam_learning=args.criculam_learning)
        
        model.save(f"models/ppo_sheep{env.n_sheep}_obst{int(args.obstacle_radius*10)}")
        print(f" PPO Finished in {str(timedelta(seconds=int(time.time() - ppo_start)))}")
    except Exception as e:
        print(f"PPO training failed: {e}")


# ================= TD3 =================
if args.algorithm in ["td3", "all"]:
    try:
        TIMESTEPS = 2000000
        estimate_training_time(env, TIMESTEPS, "TD3")

        print(f"Training with TD3 algorithm (#sheep: {env.n_sheep})...")
        td3_start = time.time()
        
        model = train_rl_agent_td3_mlp(env, eval_env, timesteps=TIMESTEPS,
                                     checkpoint_dir=args.checkpoint_dir,
                                     criculam_learning=args.criculam_learning)
        
        model.save(f"models/td3_sheep{env.n_sheep}_obst{int(args.obstacle_radius*10)}")
        print(f" TD3 Finished in {str(timedelta(seconds=int(time.time() - td3_start)))}")
    except Exception as e:
        print(f"TD3 training failed: {e}")


# ================= SAC =================
if args.algorithm in ["sac", "all"]:
    try:
        TIMESTEPS = 2000000
        estimate_training_time(env, TIMESTEPS, "SAC")

        print(f"Training with SAC algorithm (#sheep: {env.n_sheep})...")
        sac_start = time.time()
        
        model = train_rl_agent_sac_mlp(env, eval_env, timesteps=TIMESTEPS,
                                     checkpoint_dir=args.checkpoint_dir,
                                     criculam_learning=args.criculam_learning)
        
        model.save(f"models/sac_sheep{env.n_sheep}_obst{int(args.obstacle_radius*10)}")
        print(f" SAC Finished in {str(timedelta(seconds=int(time.time() - sac_start)))}")
    except Exception as e:
        print(f"SAC training failed: {e}")


print("="*40)
print(f"Total Script Duration: {str(timedelta(seconds=int(time.time() - global_start_time)))}")
print("="*40)