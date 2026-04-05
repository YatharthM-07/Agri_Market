# Complete Development Prompt for AgriMarket Optimizer RL Environment

## Executive Summary
Build a Reinforcement Learning environment for a crop warehouse management simulation where an AI agent learns optimal selling strategies for perishable agricultural products. The agent must balance freshness decay, market price fluctuations, and market risk signals to maximize revenue.

---

## 1. ENVIRONMENT SPECIFICATIONS

### 1.1 Core Identity
- **Environment Name**: AgriMarket Optimizer v1
- **Framework**: OpenAI Gym (gymnasium)
- **Domain**: Agricultural supply chain optimization
- **Problem Type**: Sequential decision-making under uncertainty
- **Real-World Context**: Simulates the challenge faced by Indian farmers deciding when to sell harvested crops before spoilage

### 1.2 State Space Design
Create a fully observable state vector with 10 dimensions:

**Inventory Levels (3 dimensions)**
- `wheat_quantity`: Integer, 0-100 units
- `corn_quantity`: Integer, 0-80 units  
- `tomato_quantity`: Integer, 0-50 units

**Freshness Timers (3 dimensions)**
- `wheat_freshness`: Days remaining (max: 10 days)
- `corn_freshness`: Days remaining (max: 7 days)
- `tomato_freshness`: Days remaining (max: 3 days)
- Timer decrements by 1 each step
- When reaches 0 with inventory > 0, crop rots

**Market Prices (3 dimensions)**
- `wheat_price`: Float, range $3-$9 per unit
- `corn_price`: Float, range $2-$7 per unit
- `tomato_price`: Float, range $5-$15 per unit
- Prices update each timestep

**Market Signal (1 dimension)**
- `news_feed`: Categorical integer {0, 1, 2}
  - 0 = Normal conditions (70% probability)
  - 1 = Rain expected, prices may rise (15% probability)
  - 2 = Market crash warning, sell immediately (15% probability)

**Implementation**: Return as `np.array` of shape (10,) with dtype `float32`

### 1.3 Action Space Design
Discrete action space with 4 possible actions:

- **Action 0 (Hold)**: Do nothing, wait for better prices
  - No immediate sale
  - Inventory unchanged
  - Freshness continues to decay
  - Risk: crops may rot or prices may crash

- **Action 1 (Sell Wheat)**: Sell entire wheat inventory
  - Immediate reward = current_wheat_price × wheat_quantity
  - wheat_quantity → 0
  - Irreversible within episode

- **Action 2 (Sell Corn)**: Sell entire corn inventory
  - Immediate reward = current_corn_price × corn_quantity
  - corn_quantity → 0
  - Irreversible within episode

- **Action 3 (Sell Tomatoes)**: Sell entire tomato inventory
  - Immediate reward = current_tomato_price × tomato_quantity
  - tomato_quantity → 0
  - Irreversible within episode

**Design Rationale**: All-or-nothing sales simplify the action space for initial learning. Can be extended to partial sales in advanced versions.

---

## 2. REWARD ENGINEERING

### 2.1 Primary Reward Structure

**Sale Rewards (Positive)**
```python
base_reward = current_price × quantity_sold
```

**Price Quality Multipliers**
- Standard sale (any price): 1.0× multiplier
- Peak pricing sale (>80th percentile): 1.5× multiplier (bonus for good timing)
- Crash-day sale (news_feed was 2 yesterday): 0.3× multiplier (penalty for ignoring warning)

**Spoilage Penalties (Negative)**
```python
if freshness <= 0 and quantity > 0:
    reward -= 50  # Per crop type that rots
```

**Risk Management Penalties**
```python
if news_feed == 2 yesterday and agent held:
    if crash_occurs_today:
        reward -= 100  # Severe penalty for ignoring crash warning
```

### 2.2 Reward Shaping Philosophy
- **Sparse vs Dense**: Primarily sparse (rewards on sales), with dense penalty signals (freshness decay)
- **Magnitude Scaling**: Sale rewards range $150-$1500, penalties are $50-$100 to create clear preference
- **Credit Assignment**: Immediate rewards for sales, delayed penalties for rot (teaches urgency)

---

## 3. EPISODE DYNAMICS

### 3.1 Initialization (`reset()`)
```python
# Starting conditions
inventory = {'wheat': 100, 'corn': 80, 'tomatoes': 50}
freshness = {'wheat': 10, 'corn': 7, 'tomatoes': 3}
market_prices = generate_prices()  # Random within ranges
news_feed = 0  # Always start normal
total_profit = 0
day = 0
```

### 3.2 Step Mechanics (`step(action)`)
**Order of operations per timestep:**

1. **Increment day counter**
2. **Decay freshness** (-1 for all crops with inventory > 0)
3. **Execute action** (if sell action, calculate reward)
4. **Check for rot** (apply -50 penalty per rotted crop)
5. **Update market prices** (generate new prices for next step)
6. **Update news feed** (generate next market signal)
7. **Evaluate termination** (check `is_done()`)
8. **Return** (next_state, reward, done, info_dict)

### 3.3 Termination Conditions
Episode ends when ANY of these occur:
- All crops sold (inventory all zeros)
- All crops rotted (all freshness timers ≤ 0)
- Maximum timesteps reached (15 days)

---

## 4. PRICE GENERATION SYSTEM

### 4.1 Task 1: Fixed Prices (Baseline Learning)
```python
prices = {'wheat': 5.0, 'corn': 3.0, 'tomatoes': 8.0}  # Constant
```
**Purpose**: Agent learns basic urgency (sell tomatoes first due to 3-day freshness)

### 4.2 Task 2: Dynamic Prices (Market Timing)
```python
def generate_prices(day, base_price, volatility=0.3):
    trend = np.sin(day * 0.5) * base_price * 0.3  # Sine wave component
    noise = np.random.uniform(-volatility, volatility) * base_price
    return max(min_price, base_price + trend + noise)
```
**Purpose**: Agent learns to wait for high-price days (temporal credit assignment)

### 4.3 Task 3: Crash Events (Risk Management)
```python
if news_feed == 2:  # Crash warning active
    # If agent doesn't sell within 1 step:
    prices = {crop: price * 0.2 for crop, price in prices.items()}  # 80% crash
```
**Purpose**: Agent learns to interpret signals and act defensively

---

## 5. PROGRESSIVE TASK STRUCTURE

### Task 1: Basic Sales (Easy)
**Objective**: Sell all inventory before spoilage
**Environment Config**:
- Fixed prices (no fluctuation)
- No news events
- Pure freshness management

**Success Metric**: 0 rot events in 90% of 100 evaluation episodes

**Expected Learning**: 
- Tomatoes have highest urgency (3-day window)
- Wheat has lowest urgency (10-day window)
- Agent should sell in order: Tomatoes → Corn → Wheat

### Task 2: Profit Maximization (Medium)
**Objective**: Achieve ≥$1,000 total profit per episode
**Environment Config**:
- Dynamic prices (sine wave + noise)
- News feed active (rain signals may boost prices)
- Must balance urgency vs price optimization

**Success Metric**: Reach $1,000 profit in 70% of 100 evaluation episodes

**Expected Learning**:
- Hold crops when prices are in rising phase
- Sell before freshness critical point
- Don't wait too long for perfect price

### Task 3: Risk Management (Hard)
**Objective**: Avoid market crash losses
**Environment Config**:
- Full price dynamics
- Crash warnings appear ~20% of episodes
- 1-day advance warning before 80% price drop

**Success Metric**: Sell within 1 timestep of crash warning in 80% of crash episodes

**Expected Learning**:
- news_feed = 2 → immediate sell signal
- Risk aversion dominates profit maximization
- Emergency liquidation strategy

---

## 6. IMPLEMENTATION ARCHITECTURE

### 6.1 File Structure
```
agrimarket-optimizer/
├── env.py              # Gym environment class
├── agent.py            # Q-Learning or DQN agent
├── train.py            # Training loop and evaluation
├── openenv.yaml        # Hugging Face config
├── README.md           # Environment documentation
├── requirements.txt    # Dependencies
└── notebooks/
    └── demo.ipynb      # Training visualization
```

### 6.2 Core Class Structure

```python
class AgriMarketEnv(gym.Env):
    """Agricultural Market Warehouse Management Environment"""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, task='task1'):
        """
        Args:
            task: 'task1', 'task2', or 'task3' for progressive difficulty
        """
        # Define spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=200, shape=(10,), dtype=np.float32
        )
        
        # Initialize environment parameters
        self.task = task
        self.initial_inventory = {'wheat': 100, 'corn': 80, 'tomatoes': 50}
        self.max_freshness = {'wheat': 10, 'corn': 7, 'tomatoes': 3}
        self.price_range = {
            'wheat': (3, 9), 
            'corn': (2, 7), 
            'tomatoes': (5, 15)
        }
        
    def reset(self):
        """Reset environment to initial state"""
        # Return: np.array(10,)
        
    def step(self, action):
        """Execute one timestep"""
        # Return: (state, reward, done, info)
        
    def _generate_prices(self):
        """Generate market prices based on current task"""
        # Task-specific price logic
        
    def _generate_news(self):
        """Generate news feed signal"""
        # Return: 0, 1, or 2
        
    def is_done(self):
        """Check termination conditions"""
        # Return: bool
        
    def state(self):
        """Construct state vector"""
        # Return: np.array(10,)
        
    def render(self, mode='human'):
        """Optional: Print current state"""
        pass
```

### 6.3 Agent Implementation Options

**Option A: Tabular Q-Learning** (Recommended for Round 1)
- Discretize continuous prices into bins
- Use sparse dictionary for Q-table
- Simple, interpretable, fast training
- Good for demonstrating learning curve

**Option B: Deep Q-Network (DQN)**
- Neural network approximation
- Handles continuous state space naturally
- More sophisticated but slower
- Better for Task 3 complexity

**Hybrid Approach**: Start with Q-Learning for Task 1-2, upgrade to DQN for Task 3

---

## 7. TRAINING PROTOCOL

### 7.1 Hyperparameters
```python
# Q-Learning
alpha = 0.1           # Learning rate
gamma = 0.95          # Discount factor
epsilon_start = 1.0   # Initial exploration
epsilon_min = 0.01    # Minimum exploration
epsilon_decay = 0.995 # Per-episode decay

# Training
num_episodes = 1000   # Task 1-2
num_episodes = 3000   # Task 3 (more complex)
max_steps = 15        # Episode length cap
```

### 7.2 Evaluation Metrics
Track per episode:
- Total reward
- Total profit ($)
- Number of rot events
- Number of timesteps survived
- Epsilon value
- Crash warning response time (Task 3)

### 7.3 Success Criteria
**Task 1**: 
- Rolling 100-episode average: 0 rots in 90+ episodes
- Consistent selling pattern: Tomatoes → Corn → Wheat

**Task 2**:
- Rolling 100-episode average profit: ≥ $1,000 in 70+ episodes
- Price-aware selling visible in logs

**Task 3**:
- Crash response: Sell within 1 step of warning in 80%+ of crash episodes
- No significant profit degradation from Task 2

---

## 8. HUGGING FACE SUBMISSION

### 8.1 openenv.yaml Structure
```yaml
env_id: agrimarket-optimizer-v1
team: team-404
description: >
  A Reinforcement Learning environment simulating crop warehouse 
  management in Indian agriculture. The agent acts as a business 
  manager deciding when to sell perishable crops (wheat, corn, 
  tomatoes) to maximize revenue while managing freshness decay 
  and reacting to market crash signals.

tasks:
  - id: task1_basic_sales
    name: Basic Sales
    difficulty: easy
    objective: Sell all inventory before freshness expires
    success_metric: Zero rot events in 90% of 100 evaluation episodes
    
  - id: task2_profit_max
    name: Profit Maximization
    difficulty: medium
    objective: Achieve $1000 total profit per episode
    success_metric: Reach target in 70% of 100 evaluation episodes
    
  - id: task3_risk_mgmt
    name: Risk Management
    difficulty: hard
    objective: Respond to market crash warnings and avoid losses
    success_metric: Sell within 1 timestep of crash warning in 80% of crash episodes

observation_space:
  type: Box
  shape: [10]
  low: 0
  high: 200
  dtype: float32
  description: 
    - wheat_quantity (units)
    - corn_quantity (units)
    - tomato_quantity (units)
    - wheat_freshness (days)
    - corn_freshness (days)
    - tomato_freshness (days)
    - wheat_price ($/unit)
    - corn_price ($/unit)
    - tomato_price ($/unit)
    - news_feed (categorical: 0=normal, 1=rain, 2=crash_warning)

action_space:
  type: Discrete
  n: 4
  actions:
    - 0: Hold (wait for better prices, risk spoilage)
    - 1: Sell Wheat (liquidate all wheat inventory)
    - 2: Sell Corn (liquidate all corn inventory)
    - 3: Sell Tomatoes (liquidate all tomato inventory)

reward_range: [-150, 1500]
reward_description: >
  Positive: (price × quantity) for sales, with 1.5× bonus for peak-price sales.
  Negative: -50 per crop that rots, -100 for ignoring crash warnings.

version: 1.0.0
license: MIT
tags:
  - agriculture
  - supply-chain
  - reinforcement-learning
  - market-simulation
  - perishable-goods
```

### 8.2 README Requirements
Include these sections:
1. **Overview**: One-paragraph environment description
2. **Installation**: `pip install -r requirements.txt`
3. **Quick Start**: 5-line code snippet to run random agent
4. **State & Action Spaces**: Detailed specifications
5. **Tasks**: Description of all 3 progressive tasks
6. **Training Example**: Command to reproduce your results
7. **Evaluation**: How judges will test submission
8. **Citation**: Team 404 attribution

---

## 9. DEVELOPMENT TIMELINE (7 Days to April 8)

### Day 1: Foundation
- [ ] Set up Git repository
- [ ] Implement `AgriMarketEnv` class skeleton
- [ ] Implement `reset()`, `step()`, `is_done()`, `state()`
- [ ] Test with random agent (no crashes)
- [ ] Verify state shape and action execution

### Day 2: Task 1 + Basic Agent
- [ ] Implement fixed-price mode
- [ ] Create Q-Learning agent
- [ ] Training loop with reward logging
- [ ] Achieve Task 1 success criteria
- [ ] Plot training curve

### Day 3: Task 2
- [ ] Implement dynamic price generation (sine wave)
- [ ] Add profit tracking
- [ ] Train agent to $1,000 target
- [ ] Verify price-aware behavior in logs

### Day 4: Task 3
- [ ] Implement crash warning system
- [ ] Add crash penalty logic
- [ ] Train agent on full environment
- [ ] Verify emergency sell behavior

### Day 5: Packaging
- [ ] Write `openenv.yaml`
- [ ] Write comprehensive README.md
- [ ] Create demo Jupyter notebook
- [ ] Add requirements.txt
- [ ] Clean up code with comments

### Day 6: Deployment
- [ ] Create Hugging Face account/team
- [ ] Push repository to Hugging Face Hub
- [ ] Test submission link accessibility
- [ ] Run final evaluation suite

### Day 7: Buffer & Polish
- [ ] Fix any last bugs
- [ ] Improve documentation
- [ ] Add visualization of agent decisions
- [ ] Submit before midnight

---

## 10. TESTING & VALIDATION

### 10.1 Unit Tests
```python
def test_environment():
    env = AgriMarketEnv()
    
    # Test reset
    state = env.reset()
    assert state.shape == (10,), "State shape incorrect"
    assert env.inventory['wheat'] == 100, "Inventory not reset"
    
    # Test step
    next_state, reward, done, _ = env.step(0)  # Hold action
    assert next_state.shape == (10,), "Next state shape incorrect"
    assert isinstance(reward, float), "Reward not float"
    assert isinstance(done, bool), "Done not bool"
    
    # Test rot penalty
    for _ in range(15):
        env.step(0)  # Hold until rot
    assert env.inventory['tomatoes'] == 0, "Tomatoes didn't rot"
    
    # Test sell action
    env.reset()
    _, reward, _, _ = env.step(1)  # Sell wheat
    assert env.inventory['wheat'] == 0, "Wheat not sold"
    assert reward > 0, "No reward for sale"
```

### 10.2 Integration Tests
```python
def test_full_episode():
    env = AgriMarketEnv(task='task1')
    state = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    
    while not done and step_count < 20:
        action = 3  # Always sell tomatoes first
        state, reward, done, _ = env.step(action)
        total_reward += reward
        step_count += 1
        
        if env.inventory['tomatoes'] == 0:
            action = 2  # Then corn
        if env.inventory['corn'] == 0:
            action = 1  # Then wheat
    
    assert total_reward > 0, "Negative total reward"
    assert done, "Episode didn't terminate"
```

### 10.3 Learning Verification
```python
def test_learning():
    env = AgriMarketEnv(task='task1')
    agent = QLearningAgent(state_size=10, action_size=4)
    
    # Train for 100 episodes
    rewards = []
    for ep in range(100):
        state = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            ep_reward += reward
        rewards.append(ep_reward)
    
    # Check improvement
    early_avg = np.mean(rewards[:20])
    late_avg = np.mean(rewards[-20:])
    assert late_avg > early_avg, "No learning improvement detected"
```

---

## 11. COMMON PITFALLS & SOLUTIONS

### Issue 1: Agent Never Sells
**Symptom**: Agent always chooses Hold (action 0)
**Cause**: Immediate hold reward (0) seems safer than uncertain sale reward
**Solution**: 
- Add small negative reward for holding (-0.1 per step)
- Increase rot penalty magnitude
- Add terminal bonus for selling all inventory

### Issue 2: Q-Table Explosion
**Symptom**: Memory error, slow training
**Cause**: Continuous price values create infinite state space
**Solution**:
- Discretize prices into 5 bins (low, below-avg, avg, above-avg, high)
- Use sparse dictionary instead of numpy array
- Consider DQN for continuous states

### Issue 3: No Learning on Task 3
**Symptom**: Agent ignores crash warnings
**Cause**: Rare event (15% of episodes), insufficient experience
**Solution**:
- Increase crash frequency during training (30%)
- Add explicit crash demonstration episodes
- Increase crash penalty to -500

### Issue 4: Inconsistent Evaluation
**Symptom**: Success rate varies wildly between runs
**Cause**: High variance in price generation
**Solution**:
- Set random seeds: `np.random.seed(42)`
- Use same evaluation episodes across runs
- Report mean ± std dev over 10 runs

---

## 12. EXTENSION IDEAS (Post-Submission)

**After securing baseline submission, consider:**

1. **Multi-agent Competition**: Two warehouses competing for market share
2. **Partial Sales**: Allow selling fraction of inventory
3. **Storage Costs**: Add daily holding fees
4. **Weather System**: Temperature affects freshness decay rate
5. **Price Prediction**: Give agent access to ML price forecaster
6. **Multiple Warehouses**: Spatial dimension with transport costs
7. **Contract System**: Option to lock in future prices
8. **Quality Tiers**: Different freshness levels fetch different prices

---

## 13. JUDGING CRITERIA ALIGNMENT

Ensure your submission excels on these dimensions:

### Real-World Relevance (25%)
- ✅ Directly models Indian agricultural supply chain challenges
- ✅ Perishability is authentic constraint (tomatoes: 3 days)
- ✅ Market volatility matches commodity trading
- ✅ News signals mirror actual information flow

### Technical Soundness (25%)
- ✅ Proper Gym interface implementation
- ✅ Reward function encourages optimal behavior
- ✅ State space is informative but not excessive
- ✅ Action space is meaningful and complete

### Learning Progression (20%)
- ✅ Clear Task 1 → 2 → 3 difficulty curve
- ✅ Agent demonstrably improves over training
- ✅ Epsilon decay visible in logs
- ✅ Different strategies for different tasks

### Code Quality (15%)
- ✅ Clean, commented, PEP-8 compliant
- ✅ No hardcoded magic numbers
- ✅ Modular design (separate env, agent, training)
- ✅ Reproducible (seeded randomness)

### Documentation (15%)
- ✅ Clear README with setup instructions
- ✅ Comprehensive openenv.yaml
- ✅ Demo notebook showing results
- ✅ Inline code comments

---

## 14. FINAL SUBMISSION CHECKLIST

**Code Functionality**
- [ ] Environment runs without errors on fresh Python 3.8+ install
- [ ] `reset()` returns correct state shape (10,)
- [ ] `step()` properly executes all 4 actions
- [ ] Freshness decrements each step
- [ ] Rot penalty fires when freshness ≤ 0
- [ ] Prices update each step
- [ ] News feed generates correctly

**Training Success**
- [ ] Task 1: 90%+ episodes with 0 rots
- [ ] Task 2: 70%+ episodes reach $1,000 profit
- [ ] Task 3: 80%+ crash warnings heeded
- [ ] Reward curves show upward trend
- [ ] Epsilon decays from 1.0 to ~0.01

**Documentation**
- [ ] README.md explains environment clearly
- [ ] openenv.yaml is valid and complete
- [ ] requirements.txt includes all dependencies
- [ ] Demo notebook runs end-to-end
- [ ] Code has meaningful comments

**Hugging Face Deployment**
- [ ] Repository is public
- [ ] Team name "team-404" in config
- [ ] Submission link tested and accessible
- [ ] All files pushed to main branch
- [ ] License file included (MIT recommended)

**Presentation**
- [ ] Training curve plot saved as image
- [ ] Agent decision log for sample episode
- [ ] Performance comparison table (Task 1 vs 2 vs 3)
- [ ] Screenshot of Hugging Face repo page

---

## 15. SAMPLE OUTPUTS

### Expected Console Output (Training)
```
Episode 0 | Avg Reward: -45.23 | Epsilon: 1.000 | Rots: 3
Episode 100 | Avg Reward: 234.56 | Epsilon: 0.366 | Rots: 0
Episode 200 | Avg Reward: 512.89 | Epsilon: 0.134 | Rots: 0
Episode 300 | Avg Reward: 823.45 | Epsilon: 0.049 | Rots: 0
Episode 400 | Avg Reward: 1087.23 | Epsilon: 0.018 | Rots: 0
Episode 500 | Avg Reward: 1243.67 | Epsilon: 0.010 | Rots: 0

Task 2 Success Rate: 78.5% (meets 70% target)
```

### Expected Agent Behavior Log (Task 3)
```
Day 0: State=[100,80,50,10,7,3,5.2,3.1,8.4,0] Action=HOLD
Day 1: State=[100,80,50,9,6,2,6.1,3.8,9.2,0] Action=HOLD
Day 2: State=[100,80,50,8,5,1,5.8,2.9,10.1,2] Action=SELL_TOMATOES ← Crash warning!
Day 3: State=[100,80,0,7,4,0,5.3,3.4,0,2] Action=SELL_CORN ← Still warning
Day 4: State=[100,0,0,6,0,0,4.9,0,0,0] Action=SELL_WHEAT
Day 5: Episode Complete | Total Profit: $1,147 | Avoided crash penalty!
```

---

## 16. TECHNICAL DEPENDENCIES

### requirements.txt
```
gymnasium==0.29.1
numpy==1.24.3
matplotlib==3.7.1
pandas==2.0.2
torch==2.0.1  # If using DQN
huggingface-hub==0.16.4
pyyaml==6.0
jupyter==1.0.0
```

### Python Version
- Minimum: Python 3.8
- Recommended: Python 3.10
- Maximum: Python 3.11 (avoid 3.12 for gym compatibility)

---

## 17. CONTACT & SUPPORT

**Team 404 Members**:
- Yatharth (Environment Logic Lead)
- Purvanshi (Agent Training Lead)

**Hackathon Resources**:
- Live Bootcamp: Today 8:00 PM
- Slack Channel: #team-404
- Mentor Office Hours: Available on request

**Questions to Ask Mentors**:
1. "Is our openenv.yaml format correct for Hugging Face?"
2. "How sensitive are judges to reward function design?"
3. "Should we use Q-Learning or DQN for Round 1?"
4. "Are there example environments we can reference?"

---

## 18. SUCCESS DEFINITION

**Minimum Viable Submission**:
- Environment implements all 3 tasks
- Agent solves Task 1 reliably
- Code runs without errors
- Hugging Face deployment successful

**Competitive Submission**:
- Agent solves all 3 tasks above success thresholds
- Training curves show clear learning
- Code is clean and well-documented
- Visualization/demo is compelling

**Winning Submission**:
- All competitive criteria met
- Novel reward shaping or agent architecture
- Excellent documentation and presentation
- Real-world insights or business recommendations

---

## CONCLUSION

You have a complete blueprint for building AgriMarket Optimizer. The environment balances simplicity (for implementability) with depth (for interesting agent behavior). 

**Key Success Factors**:
1. Build incrementally (Task 1 → 2 → 3)
2. Test frequently (random agent, unit tests)
3. Document as you go (not last minute)
4. Visualize learning (plots convince judges)
5. Ship early (deploy Day 6, polish Day 7)

The deadline is April 8, 2026. You have 7 days. Use them wisely.

**Now execute. Good luck, Team 404!** 🚀
