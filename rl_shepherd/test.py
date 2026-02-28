import argparse
import numpy as np
import os
import torch

from stable_baselines3 import PPO, A2C, TD3, SAC

from envs.shepherd_env import ShepherdEnv
from agents.rule_based_agent import RuleBasedShepherd,TipsyShepherd,LazyShepherd

from agents.MLP_QN import VectorDQNAgent, train_vector_dqn, VectorDQNAgent, N_ACTIONS, ANGLES



def load_agent(agentType: str, model_name: str, env: ShepherdEnv):
    """Load the selected agent."""
    if agentType == "ruleBase":
        print("A Rule-Based Shepherd Agent ...")
        return RuleBasedShepherd()
    if agentType == "tipsy":
        print("A Tipsy Shepherd Agent ...")
        return TipsyShepherd()
    if agentType == "lazy":
        print("A Lazy Shepherd Agent ...")
        return LazyShepherd()
    if agentType == "DQN" and os.path.exists(model_name):
        print("Using DQN Agent.")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. On récupère la taille du vecteur d'observation depuis l'environnement
        obs_dim = env.observation_space.shape[0]

        agent = VectorDQNAgent(
                            n_actions=N_ACTIONS,
                            obs_dim=obs_dim,
                            lr=1e-4,
                            gamma=0.99,
                            device=device
                        )
        agent.q_net.load_state_dict(torch.load(model_name, map_location=device))
        agent.q_net.eval()
        return agent

    if os.path.exists(model_name):
        print(f"Using {agentType} Agent.")
        return globals()[agentType].load(
            model_name,
            env=env,
            device="cpu",
        )
    else:
        raise ValueError(f"Unsupported model type: {model_name}")


def run_episode(env: ShepherdEnv, agent, model_type: str,display_flag=False) -> tuple:
    """
    Run a single episode and return (final_reward, num_steps, sheep_in_goal)
    """
    obs = env.reset()
    done = False
    reward = 0.0
    max_sheep_in_goal = 0

    while not done:
        if model_type in ["ruleBase","lazy","tipsy"]:
            actions = agent.act(obs)
        elif model_type == "DQN":
            state_tensor = torch.from_numpy(obs).float()
            with torch.no_grad():
                action_idx = agent.select_action(state_tensor)
                actions = [ANGLES[action_idx]]
        else:
            actions, _ = agent.predict(obs, deterministic=True)

        obs, reward, done, _ = env.step(actions)
        
        # Count sheep in goal
        num_sheep_in_goal = sum(1 for s in env.sheep if np.linalg.norm(s - env.goal) < env.goal_radius)
        max_sheep_in_goal = max(max_sheep_in_goal, num_sheep_in_goal)
        
        if display_flag :
            # print(f"Step {env.steps}: Action:{actions}, Reward: {reward:.2f}")
            env.render()

    return reward, env.steps, max_sheep_in_goal


def main():
    parser = argparse.ArgumentParser(description="Shepherd Environment Test Runner")

    parser.add_argument(
        "-a", "--agent_dir",
        type=str,
        default="models/",
        help="Agent model type or path to a saved model. Options: 'ruleBase', 'lazy', 'tipsy', or provide a valid file path.",
    )

    parser.add_argument(
        "-t", "--agentType",
        type=str,
        choices=["ruleBase", "lazy", "tipsy", "PPO", "DQN", "TD3", "SAC"],
        default="ruleBase",
        help="Agent model type. Choose from: 'ruleBase', 'lazy', 'tipsy', 'PPO', 'A2C', 'TD3', 'SAC'.",
    )

    parser.add_argument(
        "-n", "--num_episodes",
        type=int,
        default=1,
        help="Number of episodes to run. If greater than 1, renders are disabled and statistics are reported.",
    )

    parser.add_argument(
        "-s", "--num_sheep",
        type=int,
        default=1,
        help="Number of sheep in the environment.",
    )

    parser.add_argument(
        "-m", "--max_steps",
        type=int,
        default=500,
        help="Maximum number of steps per episode.",
    )

    parser.add_argument(
        "-r", "--obstacle_radius",
        type=float,
        default=0.0,
        help="Radius of obstacles in the environment.",
    )

    parser.add_argument(
        "-g", "--goal_radius",
        type=float,
        default=0.7,
        help="Radius of the goal area in the environment.",
    )
    
    parser.add_argument(
        "-v", "--vitesse",
        type=float,
        default=0.0,
        help="Speed of the sheep. Activate while entering in Level 2",
    )


    args = parser.parse_args()

    rewards = []
    steps_list = []
    sheep_in_goal = []

    for eps in range(1, args.num_episodes + 1):
        env = ShepherdEnv(n_sheep=args.num_sheep,
                        max_steps=args.max_steps,
                        obstacle_radius=args.obstacle_radius,
                        goal_radius=args.goal_radius,
                        sheep_speed=args.vitesse)
        print(
            f"Environment initialized with "
            f"{env.n_sheep} sheep for episode {eps}"
        )

        agent = load_agent(args.agentType,args.agent_dir, env)
        final_reward, num_steps, num_sheep_goal = run_episode(env, agent, args.agentType,display_flag=(args.num_episodes==1))

        rewards.append(final_reward)
        steps_list.append(num_steps)
        sheep_in_goal.append(num_sheep_goal)
        if args.num_episodes == 1:
            print(f"Episode {eps} finished:")
            print(f"  Final reward: {final_reward:.2f}")
            print(f"  Steps taken: {num_steps}")
            print(f"  Sheep in goal: {num_sheep_goal}/{env.n_sheep}")
        env.close()
        
    if args.num_episodes > 1:
        print(f"\nRan {args.num_episodes} episodes.")
        mean_reward = np.mean(rewards)
        mean_steps = np.mean(steps_list)
        mean_sheep_goal = np.mean(sheep_in_goal)
        std_reward = np.std(rewards)
        std_steps = np.std(steps_list)
        ci = 1.96 * (std_reward / np.sqrt(len(rewards)))  # 95% confidence interval
        ci_steps = 1.96 * (std_steps / np.sqrt(len(steps_list)))

        print(f"Average reward: {mean_reward:.2f} ± {ci:.2f} (95% CI)")
        print(f"Average Episode Length: {mean_steps:.1f} ± {ci_steps:.1f} (95% CI)")
        print(f"Sucess rate: {mean_sheep_goal:.2f}/{args.num_sheep}")
        ci_reward = 1.96 * (std_reward / np.sqrt(len(rewards)))
        print(f" 95% Conf. Int (Reward): [{mean_reward - ci_reward:.2f}, {mean_reward + ci_reward:.2f}]")
        print("="*40)


if __name__ == "__main__":
    main()
