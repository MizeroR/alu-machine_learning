# Policy Gradient - Reinforcement Learning

This project implements a **Policy Gradient (REINFORCE)** algorithm from scratch using NumPy and OpenAI Gym. The goal is to train an agent to solve the **CartPole-v1** environment by learning a policy directly.


## 📌 Concepts Covered

- Policy-based Reinforcement Learning
- Softmax Policy
- Monte-Carlo Policy Gradient (REINFORCE)
- Discounted Rewards
- Gradient Ascent Optimization

---

## 📁 Files

| File | Description |
|------|-------------|
| `policy_gradient.py` | Implements the softmax policy and gradient computation |
| `train.py` | Full training loop using REINFORCE |
| `0-main.py` | Tests policy function |
| `1-main.py` | Tests policy gradient computation |
| `2-main.py` | Runs training and plots results |
| `3-main.py` | Training with optional animation |

---

## ⚙️ How It Works

1. **Policy Function**
- Computes action probabilities using:
    ```

    softmax(state × weight)

    ```

2. **Action Selection**
- Actions are sampled from the probability distribution.

3. **Gradient Computation**
- Uses:
    ```

    grad = stateᵀ × (one_hot(action) - policy)

    ```

4. **Training (REINFORCE)**
- Collect episode trajectory
- Compute discounted rewards:
    ```

    G_t = r_t + γr_{t+1} + ...

    ```
- Update weights:
    ```

    θ = θ + α × G_t × grad

    ````


## ▶️ Usage

### Train the agent
```bash
./2-main.py
````

### Train with animation

```bash
./3-main.py
```

## 📊 Expected Results

* Initial scores are low (~10–20)
* Gradual improvement over episodes
* Converges toward **200 (solved)**


## ⚠️ Notes

* Training is stochastic; results may vary
* Tune hyperparameters for better performance:

  * `alpha` (learning rate)
  * `gamma` (discount factor)
* Rendering may require:

```python
gym.make('CartPole-v1', render_mode='human')
```


## 🚀 Outcome

By the end of training, the agent learns an optimal policy that balances the pole consistently using only **policy gradients**, without value functions.