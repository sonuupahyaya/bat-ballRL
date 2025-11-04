# rl_bat_agent_streamlit.py
import streamlit as st
import numpy as np
import random
import time
import pickle
import io
import math
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

st.set_page_config(page_title="RL Bat Agent (Enhanced)", layout="wide")

# -------------------------
# Environment (modular)
# -------------------------
class BatEnv:
    def __init__(self, screen_w=600, screen_h=400, bat_h=10, bat_y_offset=40, ball_size=15, n_balls=1, angled=False):
        self.SCREEN_W = screen_w
        self.SCREEN_H = screen_h
        self.BAT_H = bat_h
        self.BAT_Y = screen_h - bat_y_offset
        self.BALL_SIZE = ball_size
        self.n_balls = n_balls
        self.angled = angled
        self.reset()

    def reset(self, bat_width=100, lives=5):
        self.bat_width = bat_width
        self.bat_x = (self.SCREEN_W - bat_width) // 2
        self.lives = lives
        self.score = 0
        self.balls = []
        for _ in range(self.n_balls):
            bx = random.randint(0, self.SCREEN_W - self.BALL_SIZE)
            by = random.uniform(-self.SCREEN_H*0.5, 0)
            vx = random.choice([-2, -1, 0, 1, 2]) if self.angled else 0
            vy = random.uniform(2, 4)
            self.balls.append({"x": bx, "y": by, "vx": vx, "vy": vy})
        return self._get_obs()

    def step(self, action, bat_width=None, ball_speed_scale=1.0):
        # action is horizontal change
        if bat_width is None:
            bat_width = self.bat_width
        self.bat_x = int(np.clip(self.bat_x + action, 0, self.SCREEN_W - bat_width))

        reward = 0.0
        info = {"hits": 0, "misses": 0}
        done = False

        for ball in self.balls:
            ball["x"] += ball["vx"]
            ball["y"] += ball["vy"] * ball_speed_scale

            # bounce off side walls
            if ball["x"] < 0:
                ball["x"] = 0
                ball["vx"] *= -1
            if ball["x"] > self.SCREEN_W - self.BALL_SIZE:
                ball["x"] = self.SCREEN_W - self.BALL_SIZE
                ball["vx"] *= -1

            if ball["y"] + self.BALL_SIZE >= self.BAT_Y:
                # collision?
                if (self.bat_x - 10) <= ball["x"] <= (self.bat_x + bat_width + 10):
                    reward += 30
                    self.score += 1
                    self.lives += 5
                    info["hits"] += 1
                else:
                    reward -= 20
                    self.lives -= 1
                    info["misses"] += 1

                # respawn ball
                ball["y"] = random.uniform(-self.SCREEN_H*0.5, 0)
                ball["x"] = random.randint(0, self.SCREEN_W - self.BALL_SIZE)
                ball["vx"] = random.choice([-2, -1, 0, 1, 2]) if self.angled else 0
                ball["vy"] = random.uniform(2, 4)

        # distance penalty (encourage center alignment)
        # compute average distance to closest ball
        dists = [abs((self.bat_x + bat_width/2) - (ball["x"] + self.BALL_SIZE/2)) for ball in self.balls]
        if dists:
            distance_penalty = sum(dists) / (len(dists) * self.SCREEN_W)
            reward -= distance_penalty * 2

        if self.lives <= 0:
            done = True

        obs = self._get_obs()
        return obs, reward, done, info

    def _get_obs(self):
        # return coarse discretized observations
        obs = []
        for ball in self.balls:
            obs.append((int(self.bat_x // 20), int(ball["x"] // 20), int(ball["y"] // 40)))
        # flatten to tuple
        return tuple([item for t in obs for item in t])

    def render_frame(self, bat_width=None):
        if bat_width is None:
            bat_width = self.bat_width
        frame = np.zeros((self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8)
        # draw balls
        for ball in self.balls:
            x1 = int(ball["x"])
            y1 = int(ball["y"])
            x2 = min(self.SCREEN_W, x1 + self.BALL_SIZE)
            y2 = min(self.SCREEN_H, y1 + self.BALL_SIZE)
            if 0 <= y1 < self.SCREEN_H:
                frame[y1:y2, x1:x2] = [0, 255, 0]
        # bat
        frame[int(self.BAT_Y):int(self.BAT_Y)+self.BAT_H, int(self.bat_x):int(self.bat_x)+bat_width] = [255, 0, 0]
        # goal line
        frame[self.SCREEN_H-10:self.SCREEN_H-5, :] = [255,255,255]
        return frame

# -------------------------
# Helper utilities
# -------------------------
def softmax_action(qs, tau=1.0):
    # qs is list
    exp_q = np.exp(np.array(qs) / max(tau, 1e-6))
    probs = exp_q / np.sum(exp_q)
    return np.random.choice(len(qs), p=probs)

def q_state_key(state):
    # turn state tuple into a hashable key (already tuple)
    return tuple(state)

def download_bytes(obj, name):
    b = io.BytesIO()
    pickle.dump(obj, b)
    b.seek(0)
    return b

# -------------------------
# Sidebar / Controls
# -------------------------
st.sidebar.title("⚙️ Controls")
mode = st.sidebar.radio("Mode", ["Train", "Pre-trained", "Manual Play"])
difficulty = st.sidebar.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
bat_width = st.sidebar.slider("Bat Width", 40, 150, 100)
speed_scale = st.sidebar.slider("Game Speed (visual)", 1, 10, 5)
episodes = st.sidebar.slider("Training Episodes", 10, 500, 120)
n_balls = st.sidebar.slider("Number of Balls", 1, 3, 1)
angled = st.sidebar.checkbox("Ball Angles (diagonal fall)", value=False)

policy_choice = st.sidebar.selectbox("Policy", ["epsilon-greedy", "softmax", "greedy"])
alpha = st.sidebar.slider("Learning Rate (alpha)", 0.01, 1.0, 0.3)
gamma = st.sidebar.slider("Discount (gamma)", 0.0, 1.0, 0.9)
init_epsilon = st.sidebar.slider("Start Epsilon (exploration)", 0.0, 1.0, 0.8)
epsilon_decay = st.sidebar.slider("Epsilon Decay per episode", 0.90, 0.999, 0.99)

action_step = st.sidebar.select_slider("Action step (pixels)", options=[10, 20, 30, 40], value=20)
ACTIONS = [-action_step*2, -action_step, 0, action_step, action_step*2]

st.sidebar.markdown("---")
if st.sidebar.button("Restart App"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.experimental_rerun()

# Save / Load Q-table controls
st.sidebar.markdown("### Q-table")
qtable_upload = st.sidebar.file_uploader("Upload Q-table (.pkl)", type=["pkl"])
if st.sidebar.button("Download Q-table") and "q_table" in st.session_state:
    b = download_bytes(st.session_state["q_table"], "q_table.pkl")
    st.sidebar.download_button("Click to Download Q-table", data=b, file_name="q_table.pkl")

# -------------------------
# Initialize session state
# -------------------------
if "q_table" not in st.session_state:
    st.session_state.q_table = {}  # mapping (state, action) -> q
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = []  # list of (score, lives, episodes, timestamp)
if "train_stats" not in st.session_state:
    st.session_state.train_stats = {"rewards": [], "epsilons": [], "scores": [], "actions": Counter()}

# load qtable if uploaded
if qtable_upload is not None:
    try:
        st.session_state.q_table = pickle.load(qtable_upload)
        st.sidebar.success("✅ Q-table loaded")
    except Exception as e:
        st.sidebar.error(f"Error loading Q-table: {e}")

# -------------------------
# Q-learning helpers
# -------------------------
def get_q(s,a):
    return st.session_state.q_table.get((s,a), 0.0)

def set_q(s,a,val):
    st.session_state.q_table[(s,a)] = val

def choose_action(state, eps, policy="epsilon-greedy", tau=1.0):
    s = q_state_key(state)
    if policy == "epsilon-greedy":
        if random.random() < eps:
            a = random.choice(ACTIONS)
            st.session_state.train_stats["actions"][a] += 1
            return a
        qs = [get_q(s,a) for a in ACTIONS]
        arg = np.argmax(qs)
        a = ACTIONS[arg]
        st.session_state.train_stats["actions"][a] += 1
        return a
    elif policy == "greedy":
        qs = [get_q(s,a) for a in ACTIONS]
        arg = np.argmax(qs)
        a = ACTIONS[arg]
        st.session_state.train_stats["actions"][a] += 1
        return a
    elif policy == "softmax":
        qs = [get_q(s,a) for a in ACTIONS]
        idx = softmax_action(qs, tau=tau)
        a = ACTIONS[idx]
        st.session_state.train_stats["actions"][a] += 1
        return a
    else:
        # fallback epsilon
        return choose_action(state, eps, "epsilon-greedy")

def update_q(s,a,r,ns, alpha_val, gamma_val):
    old = get_q(s,a)
    # max Q for next state
    q_next = max(get_q(ns,x) for x in ACTIONS)
    new = old + alpha_val * (r + gamma_val * q_next - old)
    set_q(s,a,new)

# -------------------------
# UI layout
# -------------------------
col1, col2 = st.columns((1,1))
with col1:
    st.header("🏏  Bat Agent")
    st.write("Agent tries to hit falling balls. Hit → +5 lives, +score. Miss → -1 life.")
    st.info("Use the sidebar to control training and modes. Watch live training and visualizations.")
    canvas = st.empty()

with col2:
    st.header("Training & Stats")
    stats_box = st.empty()
    reward_plot_area = st.empty()
    epsilon_plot_area = st.empty()
    action_dist_area = st.empty()
    qheat_area = st.empty()
    save_btn = st.button("Save Q-table to Session")
    if save_btn:
        st.session_state.saved_q_table = dict(st.session_state.q_table)
        st.success("Saved current Q-table in session memory.")

# -------------------------
# Training loop or play
# -------------------------
env = BatEnv(n_balls=n_balls, angled=angled)
# adjust difficulty -> influences initial lives and ball speed scaling
if difficulty == "Easy":
    init_lives = 7
    speed_base = 1.0
elif difficulty == "Medium":
    init_lives = 5
    speed_base = 1.2
else:
    init_lives = 4
    speed_base = 1.5

# Controls for running
run_training = st.button("Run Training") if mode == "Train" else False
run_one_episode = st.button("Run One Episode") if mode in ["Train", "Manual Play"] else False

# Manual controls when in Manual Play
if mode == "Manual Play":
    st.markdown("### Manual Controls")
    manual_move = st.radio("Move Bat", ["Left", "Stay", "Right"])
    manual_step_btn = st.button("Step (manual)")

# Pre-trained mode: if user uploaded Q-table earlier, we will use it
if mode == "Pre-trained" and "q_table" not in st.session_state:
    st.warning("No Q-table found in session. Upload a Q-table via the sidebar or train and save one.")

# Training execution
def train(episodes_local):
    total_rewards = []
    for ep in range(episodes_local):
        obs = env.reset(bat_width=bat_width, lives=init_lives)
        done = False
        ep_reward = 0.0
        eps = max(0.05, init_epsilon * (epsilon_decay ** ep))

        steps = 0
        # increase speed slowly across episodes
        ball_speed_scale = min(3.0, speed_base + ep/200.0)

        while not done:
            s = q_state_key(obs)
            a = choose_action(obs, eps, policy_choice)
            obs2, r, done, info = env.step(a, bat_width=bat_width, ball_speed_scale=ball_speed_scale)
            ns = q_state_key(obs2)
            update_q(s, a, r, ns, alpha, gamma)
            obs = obs2
            ep_reward += r
            steps += 1

            # render frame to canvas
            frame = env.render_frame(bat_width=bat_width)
            canvas.image(frame, channels="RGB")
            stats_box.markdown(f"Episode {ep+1}/{episodes_local} — Step {steps} — Score: {env.score} — Lives: {env.lives} — Eps: {eps:.3f}")
            # control speed visually
            time.sleep(max(0.001, (0.02 / speed_scale)))

            if env.lives >= 10:
                stats_box.success(f"🏆 Agent Wins Episode {ep+1}! Score: {env.score} Lives: {env.lives}")
                break

        total_rewards.append(ep_reward)
        st.session_state.train_stats["rewards"].append(ep_reward)
        st.session_state.train_stats["epsilons"].append(eps)
        st.session_state.train_stats["scores"].append(env.score)

        # update progress display
        bar = st.progress((ep+1)/episodes_local)
        # draw plots every few episodes
        if (ep % max(1, episodes_local//20)) == 0 or ep == episodes_local-1:
            plot_training_metrics()

    # after training, update leaderboard
    st.session_state.leaderboard.append({"score": env.score, "lives": env.lives, "episodes": episodes_local, "time": time.asctime()})
    return total_rewards

def plot_training_metrics():
    # rewards
    rewards = st.session_state.train_stats["rewards"][-200:]
    if rewards:
        fig, ax = plt.subplots()
        ax.plot(rewards)
        ax.set_title("Recent Episode Rewards")
        ax.set_xlabel("Episode (recent)")
        ax.set_ylabel("Reward")
        reward_plot_area.pyplot(fig)
        plt.close(fig)

    # epsilon
    eps = st.session_state.train_stats["epsilons"][-200:]
    if eps:
        fig, ax = plt.subplots()
        ax.plot(eps)
        ax.set_title("Epsilon per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Epsilon")
        epsilon_plot_area.pyplot(fig)
        plt.close(fig)

    # action distribution
    actions_counter = st.session_state.train_stats["actions"]
    if sum(actions_counter.values())>0:
        acts = list(actions_counter.keys())
        counts = [actions_counter[a] for a in acts]
        fig, ax = plt.subplots()
        ax.bar([str(a) for a in acts], counts)
        ax.set_title("Action distribution (aggregated)")
        action_dist_area.pyplot(fig)
        plt.close(fig)

    # Q-heatmap: aggregate q-values by discretized bat_x and ball_x position if available
    # We'll visualize average max-Q per bat_x bin (rough)
    table = st.session_state.q_table
    if table:
        # accumulate by bat_x_bin (state first element)
        agg = defaultdict(list)
        for (s,a),qv in table.items():
            if isinstance(s, (tuple, list)) and len(s)>=1:
                bat_bin = s[0]
                agg[bat_bin].append(qv)
        if agg:
            keys = sorted(agg.keys())
            vals = [np.mean(agg[k]) for k in keys]
            fig, ax = plt.subplots(figsize=(6,2))
            ax.bar([str(k) for k in keys], vals)
            ax.set_title("Avg Q-value by bat_x bin")
            qheat_area.pyplot(fig)
            plt.close(fig)

# Handle run commands
if mode == "Train":
    if run_training:
        st.info("Training started...")
        _ = train(episodes)
        st.success("✅ Training completed.")
    if run_one_episode:
        st.info("Running single episode (training step)")
        _ = train(1)

# Manual Play
if mode == "Manual Play":
    if run_one_episode or manual_step_btn:
        # run one episode but accept manual control steps
        obs = env.reset(bat_width=bat_width, lives=init_lives)
        done = False
        steps = 0
        while not done and steps < 1000:
            # if manual mode: map manual_move to action
            if manual_step_btn or run_one_episode:
                if manual_move == "Left":
                    a = -action_step
                elif manual_move == "Right":
                    a = action_step
                else:
                    a = 0
            else:
                a = 0
            obs2, r, done, info = env.step(a, bat_width=bat_width, ball_speed_scale=speed_base)
            frame = env.render_frame(bat_width=bat_width)
            canvas.image(frame, channels="RGB")
            stats_box.markdown(f"Manual Play — Step {steps} — Score: {env.score} — Lives: {env.lives}")
            steps += 1
            time.sleep(0.02 / speed_scale)
            if env.lives <= 0:
                stats_box.error("💀 Game Over")
                break

# Pre-trained run (use Q-table to play greedy)
if mode == "Pre-trained":
    if st.button("Run Pre-trained Episode"):
        if not st.session_state.q_table:
            st.warning("No Q-table loaded/available.")
        else:
            obs = env.reset(bat_width=bat_width, lives=init_lives)
            done = False
            steps = 0
            while not done and steps < 2000:
                s = q_state_key(obs)
                qs = [get_q(s,a) for a in ACTIONS]
                a = ACTIONS[int(np.argmax(qs))]
                obs, r, done, info = env.step(a, bat_width=bat_width, ball_speed_scale=speed_base)
                frame = env.render_frame(bat_width=bat_width)
                canvas.image(frame, channels="RGB")
                stats_box.markdown(f"Pre-trained Play — Step {steps} — Score: {env.score} — Lives: {env.lives}")
                steps += 1
                time.sleep(0.02 / speed_scale)
                if env.lives <= 0:
                    stats_box.error("💀 Game Over")
                    break
            if env.lives >= 10:
                stats_box.success("🏆 Agent Wins!")

# Leaderboard display
with st.expander("🏅 Leaderboard (session)"):
    if st.session_state.leaderboard:
        for entry in sorted(st.session_state.leaderboard, key=lambda x:-x["score"]):
            st.write(f"Score: **{entry['score']}** | Lives: {entry['lives']} | Episodes: {entry['episodes']} | Time: {entry['time']}")
    else:
        st.write("No leaderboard entries yet. Train and save to populate it.")

# Small tip / explanation panel
with st.expander("🧾 Notes & Tips"):
    st.markdown("""
    - **Policy**: epsilon-greedy explores often at first, then exploits. Softmax samples actions based on relative Q values.
    - **Save/Load**: Use the sidebar to upload or download Q-tables (pickle).
    - **Manual Play**: Use manual mode to compare how a human would move vs. the agent.
    - **Scaling up**: For more complex learning, consider replacing the Q-table with a neural network (DQN).
    """)

# Footer (download Q-table after training convenience)
if mode == "Train" and st.button("Download latest Q-table"):
    if st.session_state.q_table:
        b = download_bytes(st.session_state.q_table, "q_table.pkl")
        st.download_button("Download Q-table file", data=b, file_name="q_table.pkl")
    else:
        st.warning("Q-table empty; nothing to download.")
