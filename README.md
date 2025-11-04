# 🏏 RL Bat Agent (Enhanced)
live:-https://bat-ballrl-frvu6awc3p9gwegjidjqzy.streamlit.app/

A Reinforcement Learning-based **Bat Agent** game implemented in **Python** using **Streamlit**.  
The agent learns to hit falling balls using **Q-learning** while providing interactive manual play, pre-trained modes, and real-time training metrics.

---

## **Features**

- Multiple **ball types**:  
  - Normal (green)  
  - Gold (bonus points/lives)  
  - Fast (higher speed challenge)  

- **Power-ups**:
  - Extra life
  - Bat width increase/decrease
  - Speed modification
  - Poison balls (reduce lives)

- **Combo scoring**: Bonus points for consecutive hits  
- Adjustable **bat width** dynamically  
- **Leaderboard** to track scores in the current session  
- **Training modes**:
  - Train from scratch
  - Pre-trained Q-table
  - Manual play
- **Q-learning policies**:
  - Epsilon-greedy
  - Greedy
  - Softmax
- **Training metrics visualizations**:
  - Reward per episode
  - Epsilon decay
  - Action distribution
  - Average Q-value heatmap
- Adjustable **game difficulty, speed, number of balls, and angles**

---

## **Installation**

1. Clone the repo:

```bash
git clone https://github.com/sonuupahyaya/bat-ballRL.git
cd bat-ballRL
