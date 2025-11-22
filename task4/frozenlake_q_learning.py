import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import imageio


def create_env(is_slippery: bool):
    """
    Create an 8x8 FrozenLake-v1 environment.

    :param is_slippery: False -> deterministic; True -> stochastic.
    """
    env = gym.make(
        "FrozenLake-v1",
        map_name="8x8",
        is_slippery=is_slippery,
        render_mode="rgb_array"  # for GIF generation
    )
    return env


def train_q_learning(
    env,
    num_episodes=20000,
    max_steps=200,
    alpha=0.8,
    gamma=0.99,
    eps_start=1.0,
    eps_min=0.05,
    eps_decay=0.999,
):
    """
    Tabular Q-learning with reward shaping:
      - 原始环境 reward：只有到达终点时为 1，其余为 0
      - 训练时使用的 shaped_reward：
          shaped_reward = env_reward + 0.1 * (old_dist - new_dist)
        其中 dist 是到终点(7,7)的曼哈顿距离；
        如果 episode 提前结束且 env_reward == 0（掉洞/超步数），再额外 -1 作为惩罚。
      - success rate 仍然只看 env_reward>0（真正到达终点），不受 shaping 影响。
    """

    def manhattan_dist(s: int) -> int:
        row, col = divmod(s, 8)  # 8x8 棋盘
        return abs(7 - row) + abs(7 - col)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=np.float64)

    rewards = []
    successes = []

    epsilon = eps_start

    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        total_reward = 0.0      # 只记录原始 reward，用来算 success
        success = 0

        for step in range(max_steps):
            # ε-greedy policy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state, :]))

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 只要本集出现过 env_reward>0，就算成功一集
            if reward > 0:
                success = 1

            # ---------- reward shaping ----------
            old_d = manhattan_dist(state)
            new_d = manhattan_dist(next_state)
            diff = old_d - new_d               # >0 表示更接近终点

            shaped_reward = reward + 0.1 * diff

            if done and reward == 0:
                # 掉洞或超步数但没到终点，多罚一点
                shaped_reward -= 1.0
            # -----------------------------------

            best_next = np.max(Q[next_state, :])
            if done:
                td_target = shaped_reward
            else:
                td_target = shaped_reward + gamma * best_next

            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state
            total_reward += reward  # 这里仍然用原始 reward

            if done:
                break

        # decay epsilon
        epsilon = max(eps_min, epsilon * eps_decay)

        rewards.append(total_reward)
        successes.append(success)

        # print every 500 episodes
        if episode % 500 == 0:
            last100_success = np.mean(successes[-100:]) if len(successes) >= 100 else np.mean(successes)
            print(
                f"[Ep {episode:5d}] "
                f"last-100 success rate = {last100_success:.3f}, "
                f"epsilon = {epsilon:.3f}"
            )

    return Q, np.array(rewards), np.array(successes)



def greedy_policy_run(env, Q, max_steps=200):
    """
    Roll out one episode following the greedy policy derived from Q,
    returning a list of RGB frames for GIF generation.
    """
    frames = []
    state, info = env.reset()
    for step in range(max_steps):
        # render current frame
        frame = env.render()
        frames.append(frame)

        action = int(np.argmax(Q[state, :]))
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state

        if done:
            # final frame
            frame = env.render()
            frames.append(frame)
            break

    return frames


def plot_learning_curves(
    episodes_det,
    avg_success_det,
    episodes_sto,
    avg_success_sto,
    filename="frozenlake_learning_curve.png",
):
    plt.figure(figsize=(10, 5))
    plt.plot(episodes_det, avg_success_det, label="8x8 Det (is_slippery=False)")
    plt.plot(episodes_sto, avg_success_sto, label="8x8 Sto (is_slippery=True)")
    plt.xlabel("Episode")
    plt.ylabel("Average success rate (last 100)")
    plt.title("Q-learning on FrozenLake 8x8: Success Rate")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved learning curve to {filename}")


def moving_average(x, window=100):
    """
    Simple moving average for plotting.
    """
    if len(x) < window:
        return np.array(x)
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)


def main():
    np.random.seed(0)

    # 1) Deterministic environment
    print("Training on deterministic environment (is_slippery=False)...")
    env_det = create_env(is_slippery=False)
    Q_det, rewards_det, succ_det = train_q_learning(
        env_det,
        num_episodes=20000,
        max_steps=200,
        alpha=0.8,
        gamma=0.99,
        eps_start=1.0,
        eps_min=0.05,
        eps_decay=0.999,
    )

    # 2) Stochastic environment
    print("\nTraining on stochastic environment (is_slippery=True)...")
    env_sto = create_env(is_slippery=True)
    Q_sto, rewards_sto, succ_sto = train_q_learning(
        env_sto,
        num_episodes=20000,
        max_steps=200,
        alpha=0.8,
        gamma=0.99,
        eps_start=1.0,
        eps_min=0.05,
        eps_decay=0.999,
    )

    # Compute moving averages (success rate)
    episodes = np.arange(1, len(succ_det) + 1)
    avg_det = moving_average(succ_det, window=100)
    avg_sto = moving_average(succ_sto, window=100)
    ep_det = episodes[len(episodes) - len(avg_det):]
    ep_sto = episodes[len(episodes) - len(avg_sto):]

    plot_learning_curves(ep_det, avg_det, ep_sto, avg_sto)

    # Generate GIFs for learned policies
    print("Generating GIF for deterministic policy...")
    frames_det = greedy_policy_run(env_det, Q_det)
    imageio.mimsave("policy_det.gif", frames_det, fps=4)
    print("Saved GIF: policy_det.gif")

    print("Generating GIF for stochastic policy...")
    frames_sto = greedy_policy_run(env_sto, Q_sto)
    imageio.mimsave("policy_sto.gif", frames_sto, fps=4)
    print("Saved GIF: policy_sto.gif")

    env_det.close()
    env_sto.close()


if __name__ == "__main__":
    main()

