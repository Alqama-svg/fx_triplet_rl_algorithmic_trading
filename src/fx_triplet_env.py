#!/usr/bin/env python
# coding: utf-8

# In[109]:


from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import gym
from gym import spaces
import math
from typing import Optional, Tuple, Dict


# ## Config dataclass

# In[111]:


@dataclass
class FXTripletConfig:
    from dataclasses import dataclass
from typing import Optional

@dataclass
class FXTripletConfig:
    # market model params (eqs. 5a-5b)
    xbar_eusd: float = 1.1    # mean level EUR/USD
    xbar_gbusd: float = 1.3   # mean level GBP/USD
    kappa_e: float = 0.5      # kappa0
    kappa_g: float = 0.5      # kappa1
    eta_e_on_g: float = -0.3  # eta1 effect of EUR on GBP
    eta_g_on_e: float = -0.3  # eta0 effect of GBP on EUR
    sigma: float = 0.01       # vol of shocks
    
    # time/horizon
    T: int = 10               # steps in episode
    dt: float = 1.0           # time unit per step
    
    # action constraints
    max_trade: float = 1.0    # max trade in base currency per step
    min_trade: float = -1.0   # min trade per step
    
    # cost / penalty params
    phi_eusd: float = 0.001
    phi_gbusd: float = 0.001
    phi_eurgbp: float = 0.001
    
    # terminal inventory penalties
    alpha_eur: float = 1.0
    alpha_gbp: float = 1.0
    
    # Randomness
    seed: Optional[int] = 0
    
    discrete_action: bool = False
    n_discrete: int = 11


## FX Triplet Environment

# In[113]:


class FXTripletEnv(gym.Env):
    """
    Gym-compatible environment for FX triplet:
    - Observations: [t, Xe_usd, Xgbp_usd, q_eur, q_gbp]
    - Actions: vector of three trades: [a_e_usd, a_gb_usd, a_e_gb]
           a_e_usd : trade in EUR/USD (buy EUR if +)
           a_gb_usd: trade in GBP/USD (buy GBP if +)
           a_e_gb : trade in EUR/GBP (buy EUR using GBP if +)
    - Third exchange rate (EUR/GBP) is implied by no-arbitrage: EUR/GBP = EUR/USD / GBP/USD
    - Reward: per eq. (3) (stepwise value changes + negative quadratic walking-the-book cost).
    """
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, cfg: FXTripletConfig = FXTripletConfig()):
        super().__init__()
        self.cfg = cfg

        # seeding first (must be defined before usage)
        # self._seed = cfg.seed
        
        # RNG
        self._rng = np.random.default_rng(cfg.seed)

        # state placeholders
        self.t = None
        self.Xe = None # EUR/USD
        self.Xg = None # GBP/USD
        self.q_usd = None # USD cash inventory
        self.q_gbp = None # GBP inventory
        self.q_eur = None # EUR inventory

        # observation space: t (int scaled to [0,T]), Xe, Xg, q_eur, q_gbp
        # use Box with reasonable bounds
        obs_low = np.array([0.0, 0.5, 0.5, -10.0, -10.0], dtype = np.float32)
        obs_high = np.array([float(self.cfg.T), 5.0, 5.0, 10.0, 10.0], dtype = np.float32)
        self.observation_space = spaces.Box(low = obs_low, high = obs_high, dtype = np.float32)

        # action space
        self.discrete_action = self.cfg.discrete_action
        if self.cfg.discrete_action:
            self.n_discrete = int(self.cfg.n_discrete)
            self.action_space = spaces.Discrete(self.n_discrete)
            self._action_table = self._make_discrete_action_table(self.n_discrete)
        else:
            # continuous 3-d actions in [-max_trade, max_trade]
            self.action_space = spaces.Box(
                low=np.array([self.cfg.min_trade] * 3, dtype=np.float32),
                high=np.array([self.cfg.max_trade] * 3, dtype=np.float32),
                dtype=np.float32
            )
        # internal bookkeeping
        self.current_step = 0
        self.seed(cfg.seed)

    
    # discrete action table
    def implied_eur_gbp(self):
        # no-arbitrage: EUR/GBP = EUR/USD / GBP/USD
        # guard against division by zero
        if abs(self.Xg) < 1e-12:
            return self.Xe / (self.Xg + 1e-12)
        return self.Xe / self.Xg
    
    def seed(self, seed: Optional[int] = None):
        """Set RNG seed for reproducibility"""
        self._seed = seed
        self._rng = np.random.default_rng(seed if seed is not None else None)
        return [seed]

    def _make_discrete_action_table(self, n: int):
        """Create discrete action lookup table"""
        rng = np.random.default_rng(self.cfg.seed)
        if n <= 25:
            grid = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            combos = np.array([[a, b, c] for a in grid for b in grid for c in grid])
            idx = np.linspace(0, len(combos) - 1, n).astype(int)
            table = combos[idx]
            return table.astype(np.float32)
        else:
            table = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
            return table

    def _get_obs(self):
        return np.array([float(self.t), float(self.Xe), float(self.Xg),
                         float(self.q_eur), float(self.q_gbp)], dtype=np.float32)

    # Gym API
    def reset(self, *, seed: Optional[int] = None,
              return_info: bool = False, options: dict = None):
        if seed is not None:
            self.seed(seed)
        self.current_step = 0
        self.t = 0
        # initialize prices
        self.Xe = float(self.cfg.xbar_eusd)
        self.Xg = float(self.cfg.xbar_gbusd)
        # inventories
        self.q_usd = 0.0
        self.q_gbp = 0.0
        self.q_eur = 0.0

        obs = self._get_obs()
        if return_info:
            return obs, {}
        return obs

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        """Advance environment one step given trade action"""
        if self.discrete_action and isinstance(action, (np.integer, int)):
            action = self._action_table[int(action)]
        action = np.asarray(action, dtype=np.float64)
        assert action.shape == (3,), f"action must be shape (3,), got {action.shape}"

        # Clip actions
        a_eusd = float(np.clip(action[0], self.cfg.min_trade, self.cfg.max_trade))
        a_gbusd = float(np.clip(action[1], self.cfg.min_trade, self.cfg.max_trade))
        a_eurgbp = float(np.clip(action[2], self.cfg.min_trade, self.cfg.max_trade))

        # Store current prices
        Xe_t = float(self.Xe)
        Xg_t = float(self.Xg)
        Xeg_t = float(self.implied_eur_gbp())

        # Price dynamics (eqs. 5a–5b)
        eps_e = float(self._rng.normal(0.0, self.cfg.sigma))
        eps_g = float(self._rng.normal(0.0, self.cfg.sigma))
        Xe_next = Xe_t + self.cfg.kappa_e * (self.cfg.xbar_eusd - Xe_t) \
                  + self.cfg.eta_g_on_e * (self.cfg.xbar_gbusd - Xg_t) + eps_e
        Xg_next = Xg_t + self.cfg.kappa_g * (self.cfg.xbar_gbusd - Xg_t) \
                  + self.cfg.eta_e_on_g * (self.cfg.xbar_eusd - Xe_t) + eps_g

        # implied EUR/GBP
        Xeg_next = Xe_next / (Xg_next + 1e-12)

        # Inventory update
        self.q_usd = float(self.q_usd - Xe_t * a_eusd - Xg_t * a_gbusd)
        self.q_gbp = float(self.q_gbp + a_gbusd - Xeg_t * a_eurgbp)
        self.q_eur = float(self.q_eur + a_eusd + a_eurgbp)

        # Reward per eq. (3)
        q_gbp_before = self.q_gbp - (a_gbusd - Xeg_t * a_eurgbp)
        q_eur_before = self.q_eur - (a_eusd + a_eurgbp)
        expr1 = (q_gbp_before + a_gbusd - Xeg_t * a_eurgbp) * (Xg_next - Xg_t)
        expr2 = (q_eur_before + a_eusd + a_eurgbp) * (Xe_next - Xe_t)
        cost = self.cfg.phi_gbusd * (a_gbusd ** 2) \
             + self.cfg.phi_eusd * (a_eusd ** 2) \
             + self.cfg.phi_eurgbp * (a_eurgbp ** 2)
        reward = float(expr1 + expr2 - cost)

        # advance state
        self.Xe = Xe_next
        self.Xg = Xg_next
        self.current_step += 1
        self.t = self.current_step

        done = (self.current_step >= self.cfg.T)
        if done:
            terminal_pen = -(self.cfg.alpha_eur * (self.q_eur ** 2) +
                             self.cfg.alpha_gbp * (self.q_gbp ** 2))
            reward += terminal_pen

        obs = self._get_obs()
        info = {"Xe": self.Xe, "Xg": self.Xg, "Xeg": self.implied_eur_gbp(),
                "q_usd": self.q_usd, "q_gbp": self.q_gbp, "q_eur": self.q_eur,
                "step": self.current_step}
        return obs, reward, done, info


if __name__ == "__main__":
    env = FXTripletEnv()
    obs = env.reset()
    print("Initial obs:", obs)
    action = np.array([0.1, -0.1, 0.0])
    obs, r, done, info = env.step(action)
    print("Step result:", obs, r, done, info)


# ## discrete action table (simple grid over each action dimension)

# In[ ]:


def _make_discrete_action_table(self, n:int):
    # create a small set of 1D triplet actions where each of the three dims
    # can be in the grid [-1, -0.5, 0, 0.5, 1] (or adaptively from n). For n not matching exact cartesian
    # generate a random but reproducible set

    rng = np.random.default_rng(self.cfg.seed if self.cfg.seed is None else None)
    if n <= 25:
        # generate coarse grid combinations
        grid = np.array([-1.0, -0.5, -0.0, 0.5, 1.0])
        combos= []
        for a in grid:
            for b in grid:
                for c in grid:
                    combos.append([a, b, c])
        combos = np.array(combos)
        # pick first n distinct combos
        idx = np.linspace(0, len(combos) - 1, n).astype(int)
        table = combos[idx]
        return table.astype(np.float32)
    else:
        # generate n random triplets in [-1,1] reproducibly
        table = rng.unifrom(-1.0, 1.0, size=(n,3)).astype(np.float32)
        return table


# ## seeding

# In[ ]:


def seed(self, seed: Optional[int] = None):
    self._seed = seed
    self._rng = np.random.default_rng(seed)
    return [seed]


# ## reset

# In[ ]:


def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: dict = None):
    if seed is not None:
        sel.seed(seed)
    self.current_step = 0
    self.t = 0
    # initialize prices at their mean levels
    self.Xe = float(self.cfg.xbar_eusd + 0.0)
    self.Xg = float(self.cfg.xbar_gbusd + 0.0)
    # initial inventories: start flat (paper used q = 0)
    self.q_usd = 0.0
    self.q_gbp = 0.0
    self.q_eur = 0.0
    obs = self._get_obs()
    if return_info:
        return obs, {}
    return obs


# ## step

# In[ ]:


def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
    if self.discrete_action and isinstance(action, (np.integer, int)):
        action = self._action_table[int(action)]
    action = np.asarray(action, dtype = np.float64)
    assert action.shape == (3,), f"action must be shape (3,), got {action.shape}"

    # clip actions within allowed trade bounds
    a_eusd = float(np.clip(action[0], self.cfg.min_trade, self.cfg.max_trade))
    a_gbusd = float(np.clip(action[1], self.cfg.min_trade, self.cfg.max_trade))
    a_eurgbp = float(np.clip(action[2], self.cfg.min_trade, self.cfg.max_trade))

    # store current prices
    Xe_t = float(self.Xe)
    Xg_t = float(self.Xg)
    Xeg_t = float(self.implied_eur_gbp())

    # Step the fundamental price dynamics (eqs. 5a-5b)
    # Xe_{t+1} = Xe_t + kappa_e*(xbar_e - Xe_t) + eta_g_on_e*(xbar_g - Xg_t) + eps_e
    # Xg_{t+1} = Xg_t + kappa_g*(xbar_g - Xg_t) + eta_e_on_g*(xbar_e - Xe_t) + eps_g
    eps_e = float(self._rng.normal(0.0, self.cfg.sigma))
    eps_g = float(self._rng.normal(0.0, self.cfg.sigma))

    Xe_next = Xe_t + self.cfg.kappa_e * (self.cfg.xbar_eusd - Xe_t) + self.cfg.eta_g_on_e * (self.cfg.xbar_gbusd - Xg_t) + eps_e
    Xg_next = Xg_t + self.cfg.kappa_g * (self.cfg.xbar_gbusd - Xg_t) + self.cfg.eta_e_on_g * (self.cfg.xbar_eusd - Xe_t) + eps_g

    # Ensure third implied rate respects no-arbitrage exactly at next step by computing it from Xe_next and Xg_next
    Xeg_next = Xe_next / Xg_next if abs(Xg_next) > 1e-12 else Xe_next / (Xg_net + 1e-12)

    # Update inventories per trade conventions
    
    # When buy base currency i in pair i/j (a_ij > 0), base inventory increases, quote currency inventory decreases by price * a_ij
    # Paper uses q$ = USD inventory, q£=GBP, qe=EUR
    # Updates:
    # q$_t+1 = q$_t - Xe_t * a_eusd - Xg_t * a_gbusd
    # q£_t+1 = q£_t + a_gbusd - Xeg_t * a_eurgbp
    # qe_t+1 = qe_t + a_eusd + a_eurgbp
    self.q_usd = float(self.q_usd - Xe_t * a_eusd - Xg_t * a_gbusd)
    self.q_gbp = float(self.q_gbp + a_gbusd - Xeg_t * a_eurgbp)
    self.q_eur = float(self.q_eur + a_eusd + a_eurgbp)

    # compute reward according to eq. (3) (step by step)
    # R_t = (q£_t + a£_t - Xeg_t * ae£_t)*(Xg_{t+1}-Xg_t) + (qe_t + ae_t + ae£_t)*(Xe_{t+1}-Xe_t) - sum(phi * a^2)
    term1 = (self.q_gbp - a_gbusd + Xeg_t * a_eurgbp) # careful: q£t in paper denotes inventory before action; we have updated inventories. To match paper eq (3), we reconstruct q before next price move:
    # To be precise, I want (q£_t + a£_t - Xe£_t * ae£_t), where q£_t is the inventory BEFORE applying this step's trades
    # We saved inventories AFTER trades. So reconstruct q_before:
    q_gbp_before = self.q_gbp - (a_gbusd - Xeg_t * a_eurgbp)
    q_eur_before = self.q_eur - (a_eusd + a_eurgbp)
    expr1 = (q_gbp_before + a_gbusd - Xeg_t * a_eurgbp) * (Xg_next - Xg_t)
    expr2 = (q_eur_before + a_eusd + a_eurgbp) * (Xe_next - Xe_t)
    cost = self.cfg.phi_gbusd * (a_gbusd ** 2) + self.cfg.phi_eusd * (a_eusd ** 2) + self.cfg.phi_eurgbp * (a_eurgbp ** 2)
    reward = float(expr1 + expr2 - cost)

    # advance prices to next
    self.Xe = float(Xe_next)
    self.Xg = float(Xg_next)
    # increment time
    self.current_step += 1
    self.t = self.current_step

    done = (self.current_step >= self.cfg.T)
    # terminal penalty if done: -alpha_e*(q_eur)^2 - alpha_gbp*(q_gbp)^2 (convert to USD for paper they are USD units, but here q are in base units so treat params as consistent)
    if done:
        terminal_pen = -(self.cfg.alpha_eur * (self.q_eur ** 2) + self.cfg.alpha_gbp * (self.q_gbp ** 2))
        reward += terminal_pen

    obs = self._get_obs()
    info = {
        "Xe": self.Xe,
        "Xg": self.Xg,
        "Xeg": self.implied_eur_gbp(),
        "q_usd": self.q_usd,
        "q_gbp": self.q_gbp,
        "q_eur": self.q_eur,
        "step": self.current_step
    }
    return obs, reward, done, info


# ## observation getter

# In[ ]:


def _get_obs(self):
    return np.array([float(self.t), float(self.Xe), float(self.Xg), float(self.q_eur), float(self.q_gbp)], dtype=np.float32)


# ## render

# In[ ]:


def render(self, mode="human"):
    print(f"t={self.t}, Xe(EUR/USD)={self.Xe:.6f}, Xg(GBP/USD)={self.Xg:.6f}, Xeg(EUR/GBP)={self.implied_eur_gbp():.6f}")
    print(f"q_eur={self.q_eur:.6f}, q_gbp={self.q_gbp:.6f}, q_usd={self.q_usd:.6f}")


# ## vectorized rollout (for testing / eval)

# In[ ]:


def rollout(self, policy_fn, n_steps: Optional[int] = None):
    """
    policy_fn(obs) -> action
    returns trajectory dict
    """
    if n_steps is None:
        n_steps = self.cfg.T
    obs = self.reset()
    traj = {"obs": [], "actions": [], "rewards": [], "infos": []}
    done = False
    for _ in range(n_steps):
        action = policy_fn(obs)
        obs, r, done, info = self.step(action)
        traj["obs"].append(obs)
        traj["actions"].append(action)
        traj["rewards"].append(r)
        traj["infos"].append(info)
        if done:
            break
    return traj


# ## small unit tests

# In[ ]:


def _test_no_arbitrage_enforced():
    cfg = FXTripletConfig(seed=42)
    env = FXTripletEnv(cfg)
    obs = env.reset()
    # Step with zero noise (set sigma=0) and zero action, prices should revert deterministically
    cfg_zero = FXTripletConfig(seed=1, sigma=0.0)
    env2 = FXTripletEnv(cfg_zero)
    obs2 = env2.reset()
    # store initial implied
    implied0 = env2.implied_eur_gbp()
    # step with zero action
    obs_next, r, done, info = env2.step(np.array([0.0, 0.0, 0.0]))
    # check implied relation holds exactly (within numerical tolerance)
    implied_next = info["Xeg"]
    Xe = info["Xe"]; Xg = info["Xg"]
    implied_calc = Xe / Xg
    assert abs(implied_next - implied_calc) < 1e-12, "No-arbitrage implied rate not enforced exactly"
    print("TEST PASSED: No-arbitrage is enforced (implied_eur_gbp == Xe / Xg).")

def _test_reward_deterministic():
    # deterministic price drift: set dynamics so that Xe increases by +d and Xg increases by +d, sigma=0
    cfg = FXTripletConfig(seed=123, sigma=0.0, kappa_e=0.0, kappa_g=0.0, eta_e_on_g=0.0, eta_g_on_e=0.0)
    # We'll set initial prices and then force externals by directly patching environment next-step values? Simpler:
    # For this deterministic test, set eps via RNG to 0 and use manual small price change by temporarily modifying cfg.
    env = FXTripletEnv(cfg)
    env.reset()
    # monkeypatch prices so Xe_t = 1.0, Xg_t = 2.0
    env.Xe = 1.0
    env.Xg = 2.0
    # I'll pick a simple action: buy 1 EUR via EUR/USD (a_eusd=1), no other trades
    # Set cfg such that next-step Xe increases by +0.01 and Xg by +0.02 by directly overriding the model random draws.
    # Easiest: temporarily set rng.normal to return specified epsilons
    class DummyRNG:
        def __init__(self, eps_e, eps_g):
            self.eps_e = eps_e
            self.eps_g = eps_g
        def normal(self, loc, scale):
            # called twice; return eps_e, then eps_g
            if scale == cfg.sigma:
                return 0.0
            return 0.0
        def __call__(self, *args, **kwargs):
            return 0.0

    # Instead of monkeypatching RNG complexity, craft cfg so drift moves prices deterministically:
    env.cfg.kappa_e = 0.01
    env.cfg.kappa_g = 0.02
    env.cfg.xbar_eusd = env.Xe + 1.0  # forces positive drift for Xe: Xe_next - Xe ~= 0.01 * 1.0
    env.cfg.xbar_gbusd = env.Xg + 1.0  # similarly for Xg
    # run step with action [1,0,0]
    obs_before = env._get_obs().copy()
    obs_next, reward, done, info = env.step(np.array([1.0, 0.0, 0.0]))
    # compute expected step reward manually per eq (3)
    # Compute q_before
    q_gbp_before = 0.0
    q_eur_before = 0.0
    a_eusd = 1.0
    a_gbusd = 0.0
    a_eurgbp = 0.0
    Xe_t = 1.0; Xg_t = 2.0
    # next price estimates
    Xe_next = Xe_t + env.cfg.kappa_e * (env.cfg.xbar_eusd - Xe_t)
    Xg_next = Xg_t + env.cfg.kappa_g * (env.cfg.xbar_gbusd - Xg_t)
    expr1 = (q_gbp_before + a_gbusd - (Xe_t / Xg_t) * a_eurgbp) * (Xg_next - Xg_t)
    expr2 = (q_eur_before + a_eusd + a_eurgbp) * (Xe_next - Xe_t)
    cost = env.cfg.phi_eusd * (a_eusd**2) + env.cfg.phi_gbusd*(a_gbusd**2) + env.cfg.phi_eurgbp*(a_eurgbp**2)
    expected_reward = expr1 + expr2 - cost
    assert abs(reward - expected_reward) < 1e-8, f"Reward mismatch: got {reward}, expected {expected_reward}"
    print("TEST PASSED: Deterministic reward computation matches expected formula.")

if __name__ == "__main__":
    print("Running FXTripletEnv unit tests...")
    _test_no_arbitrage_enforced()
    _test_reward_deterministic()
    print("All tests passed ✅")


# ## run tests

# In[ ]:


"""
def _test_no_arbitrage_enforced():
        cfg = FXTripletConfig(seed=0)
        env = FXTripletEnv(cfg)
        obs = env.reset()
        print("Initial obs:", obs)
        action = np.array([0.2, -0.1, 0.0])
        obs, r, done, info = env.step(action)
def _test_reward_deterministic():
        print("Dummy reward test skipped in this minimal version.")

if __name__ == "__main__":
    _test_no_arbitrage_enforced()
    _test_reward_deterministic()
    """


# In[ ]:




=======
#!/usr/bin/env python
# coding: utf-8

# In[109]:


from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import gym
from gym import spaces
import math
from typing import Optional, Tuple, Dict


# ## Config dataclass

# In[111]:


@dataclass
class FXTripletConfig:
    from dataclasses import dataclass
from typing import Optional

@dataclass
class FXTripletConfig:
    # market model params (eqs. 5a-5b)
    xbar_eusd: float = 1.1    # mean level EUR/USD
    xbar_gbusd: float = 1.3   # mean level GBP/USD
    kappa_e: float = 0.5      # kappa0
    kappa_g: float = 0.5      # kappa1
    eta_e_on_g: float = -0.3  # eta1 effect of EUR on GBP
    eta_g_on_e: float = -0.3  # eta0 effect of GBP on EUR
    sigma: float = 0.01       # vol of shocks
    
    # time/horizon
    T: int = 10               # steps in episode
    dt: float = 1.0           # time unit per step
    
    # action constraints
    max_trade: float = 1.0    # max trade in base currency per step
    min_trade: float = -1.0   # min trade per step
    
    # cost / penalty params
    phi_eusd: float = 0.001
    phi_gbusd: float = 0.001
    phi_eurgbp: float = 0.001
    
    # terminal inventory penalties
    alpha_eur: float = 1.0
    alpha_gbp: float = 1.0
    
    # Randomness
    seed: Optional[int] = 0
    
    discrete_action: bool = False
    n_discrete: int = 11


## FX Triplet Environment

# In[113]:


class FXTripletEnv(gym.Env):
    """
    Gym-compatible environment for FX triplet:
    - Observations: [t, Xe_usd, Xgbp_usd, q_eur, q_gbp]
    - Actions: vector of three trades: [a_e_usd, a_gb_usd, a_e_gb]
           a_e_usd : trade in EUR/USD (buy EUR if +)
           a_gb_usd: trade in GBP/USD (buy GBP if +)
           a_e_gb : trade in EUR/GBP (buy EUR using GBP if +)
    - Third exchange rate (EUR/GBP) is implied by no-arbitrage: EUR/GBP = EUR/USD / GBP/USD
    - Reward: per eq. (3) (stepwise value changes + negative quadratic walking-the-book cost).
    """
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, cfg: FXTripletConfig = FXTripletConfig()):
        super().__init__()
        self.cfg = cfg

        # seeding first (must be defined before usage)
        # self._seed = cfg.seed
        
        # RNG
        self._rng = np.random.default_rng(cfg.seed)

        # state placeholders
        self.t = None
        self.Xe = None # EUR/USD
        self.Xg = None # GBP/USD
        self.q_usd = None # USD cash inventory
        self.q_gbp = None # GBP inventory
        self.q_eur = None # EUR inventory

        # observation space: t (int scaled to [0,T]), Xe, Xg, q_eur, q_gbp
        # use Box with reasonable bounds
        obs_low = np.array([0.0, 0.5, 0.5, -10.0, -10.0], dtype = np.float32)
        obs_high = np.array([float(self.cfg.T), 5.0, 5.0, 10.0, 10.0], dtype = np.float32)
        self.observation_space = spaces.Box(low = obs_low, high = obs_high, dtype = np.float32)

        # action space
        self.discrete_action = self.cfg.discrete_action
        if self.cfg.discrete_action:
            self.n_discrete = int(self.cfg.n_discrete)
            self.action_space = spaces.Discrete(self.n_discrete)
            self._action_table = self._make_discrete_action_table(self.n_discrete)
        else:
            # continuous 3-d actions in [-max_trade, max_trade]
            self.action_space = spaces.Box(
                low=np.array([self.cfg.min_trade] * 3, dtype=np.float32),
                high=np.array([self.cfg.max_trade] * 3, dtype=np.float32),
                dtype=np.float32
            )
        # internal bookkeeping
        self.current_step = 0
        self.seed(cfg.seed)

    
    # discrete action table
    def implied_eur_gbp(self):
        # no-arbitrage: EUR/GBP = EUR/USD / GBP/USD
        # guard against division by zero
        if abs(self.Xg) < 1e-12:
            return self.Xe / (self.Xg + 1e-12)
        return self.Xe / self.Xg
    
    def seed(self, seed: Optional[int] = None):
        """Set RNG seed for reproducibility"""
        self._seed = seed
        self._rng = np.random.default_rng(seed if seed is not None else None)
        return [seed]

    def _make_discrete_action_table(self, n: int):
        """Create discrete action lookup table"""
        rng = np.random.default_rng(self.cfg.seed)
        if n <= 25:
            grid = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            combos = np.array([[a, b, c] for a in grid for b in grid for c in grid])
            idx = np.linspace(0, len(combos) - 1, n).astype(int)
            table = combos[idx]
            return table.astype(np.float32)
        else:
            table = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
            return table

    def _get_obs(self):
        return np.array([float(self.t), float(self.Xe), float(self.Xg),
                         float(self.q_eur), float(self.q_gbp)], dtype=np.float32)

    # Gym API
    def reset(self, *, seed: Optional[int] = None,
              return_info: bool = False, options: dict = None):
        if seed is not None:
            self.seed(seed)
        self.current_step = 0
        self.t = 0
        # initialize prices
        self.Xe = float(self.cfg.xbar_eusd)
        self.Xg = float(self.cfg.xbar_gbusd)
        # inventories
        self.q_usd = 0.0
        self.q_gbp = 0.0
        self.q_eur = 0.0

        obs = self._get_obs()
        if return_info:
            return obs, {}
        return obs

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        """Advance environment one step given trade action"""
        if self.discrete_action and isinstance(action, (np.integer, int)):
            action = self._action_table[int(action)]
        action = np.asarray(action, dtype=np.float64)
        assert action.shape == (3,), f"action must be shape (3,), got {action.shape}"

        # Clip actions
        a_eusd = float(np.clip(action[0], self.cfg.min_trade, self.cfg.max_trade))
        a_gbusd = float(np.clip(action[1], self.cfg.min_trade, self.cfg.max_trade))
        a_eurgbp = float(np.clip(action[2], self.cfg.min_trade, self.cfg.max_trade))

        # Store current prices
        Xe_t = float(self.Xe)
        Xg_t = float(self.Xg)
        Xeg_t = float(self.implied_eur_gbp())

        # Price dynamics (eqs. 5a–5b)
        eps_e = float(self._rng.normal(0.0, self.cfg.sigma))
        eps_g = float(self._rng.normal(0.0, self.cfg.sigma))
        Xe_next = Xe_t + self.cfg.kappa_e * (self.cfg.xbar_eusd - Xe_t) \
                  + self.cfg.eta_g_on_e * (self.cfg.xbar_gbusd - Xg_t) + eps_e
        Xg_next = Xg_t + self.cfg.kappa_g * (self.cfg.xbar_gbusd - Xg_t) \
                  + self.cfg.eta_e_on_g * (self.cfg.xbar_eusd - Xe_t) + eps_g

        # implied EUR/GBP
        Xeg_next = Xe_next / (Xg_next + 1e-12)

        # Inventory update
        self.q_usd = float(self.q_usd - Xe_t * a_eusd - Xg_t * a_gbusd)
        self.q_gbp = float(self.q_gbp + a_gbusd - Xeg_t * a_eurgbp)
        self.q_eur = float(self.q_eur + a_eusd + a_eurgbp)

        # Reward per eq. (3)
        q_gbp_before = self.q_gbp - (a_gbusd - Xeg_t * a_eurgbp)
        q_eur_before = self.q_eur - (a_eusd + a_eurgbp)
        expr1 = (q_gbp_before + a_gbusd - Xeg_t * a_eurgbp) * (Xg_next - Xg_t)
        expr2 = (q_eur_before + a_eusd + a_eurgbp) * (Xe_next - Xe_t)
        cost = self.cfg.phi_gbusd * (a_gbusd ** 2) \
             + self.cfg.phi_eusd * (a_eusd ** 2) \
             + self.cfg.phi_eurgbp * (a_eurgbp ** 2)
        reward = float(expr1 + expr2 - cost)

        # advance state
        self.Xe = Xe_next
        self.Xg = Xg_next
        self.current_step += 1
        self.t = self.current_step

        done = (self.current_step >= self.cfg.T)
        if done:
            terminal_pen = -(self.cfg.alpha_eur * (self.q_eur ** 2) +
                             self.cfg.alpha_gbp * (self.q_gbp ** 2))
            reward += terminal_pen

        obs = self._get_obs()
        info = {"Xe": self.Xe, "Xg": self.Xg, "Xeg": self.implied_eur_gbp(),
                "q_usd": self.q_usd, "q_gbp": self.q_gbp, "q_eur": self.q_eur,
                "step": self.current_step}
        return obs, reward, done, info


if __name__ == "__main__":
    env = FXTripletEnv()
    obs = env.reset()
    print("Initial obs:", obs)
    action = np.array([0.1, -0.1, 0.0])
    obs, r, done, info = env.step(action)
    print("Step result:", obs, r, done, info)


# ## discrete action table (simple grid over each action dimension)

# In[ ]:


def _make_discrete_action_table(self, n:int):
    # create a small set of 1D triplet actions where each of the three dims
    # can be in the grid [-1, -0.5, 0, 0.5, 1] (or adaptively from n). For n not matching exact cartesian
    # generate a random but reproducible set

    rng = np.random.default_rng(self.cfg.seed if self.cfg.seed is None else None)
    if n <= 25:
        # generate coarse grid combinations
        grid = np.array([-1.0, -0.5, -0.0, 0.5, 1.0])
        combos= []
        for a in grid:
            for b in grid:
                for c in grid:
                    combos.append([a, b, c])
        combos = np.array(combos)
        # pick first n distinct combos
        idx = np.linspace(0, len(combos) - 1, n).astype(int)
        table = combos[idx]
        return table.astype(np.float32)
    else:
        # generate n random triplets in [-1,1] reproducibly
        table = rng.unifrom(-1.0, 1.0, size=(n,3)).astype(np.float32)
        return table


# ## seeding

# In[ ]:


def seed(self, seed: Optional[int] = None):
    self._seed = seed
    self._rng = np.random.default_rng(seed)
    return [seed]


# ## reset

# In[ ]:


def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: dict = None):
    if seed is not None:
        sel.seed(seed)
    self.current_step = 0
    self.t = 0
    # initialize prices at their mean levels
    self.Xe = float(self.cfg.xbar_eusd + 0.0)
    self.Xg = float(self.cfg.xbar_gbusd + 0.0)
    # initial inventories: start flat (paper used q = 0)
    self.q_usd = 0.0
    self.q_gbp = 0.0
    self.q_eur = 0.0
    obs = self._get_obs()
    if return_info:
        return obs, {}
    return obs


# ## step

# In[ ]:


def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
    if self.discrete_action and isinstance(action, (np.integer, int)):
        action = self._action_table[int(action)]
    action = np.asarray(action, dtype = np.float64)
    assert action.shape == (3,), f"action must be shape (3,), got {action.shape}"

    # clip actions within allowed trade bounds
    a_eusd = float(np.clip(action[0], self.cfg.min_trade, self.cfg.max_trade))
    a_gbusd = float(np.clip(action[1], self.cfg.min_trade, self.cfg.max_trade))
    a_eurgbp = float(np.clip(action[2], self.cfg.min_trade, self.cfg.max_trade))

    # store current prices
    Xe_t = float(self.Xe)
    Xg_t = float(self.Xg)
    Xeg_t = float(self.implied_eur_gbp())

    # Step the fundamental price dynamics (eqs. 5a-5b)
    # Xe_{t+1} = Xe_t + kappa_e*(xbar_e - Xe_t) + eta_g_on_e*(xbar_g - Xg_t) + eps_e
    # Xg_{t+1} = Xg_t + kappa_g*(xbar_g - Xg_t) + eta_e_on_g*(xbar_e - Xe_t) + eps_g
    eps_e = float(self._rng.normal(0.0, self.cfg.sigma))
    eps_g = float(self._rng.normal(0.0, self.cfg.sigma))

    Xe_next = Xe_t + self.cfg.kappa_e * (self.cfg.xbar_eusd - Xe_t) + self.cfg.eta_g_on_e * (self.cfg.xbar_gbusd - Xg_t) + eps_e
    Xg_next = Xg_t + self.cfg.kappa_g * (self.cfg.xbar_gbusd - Xg_t) + self.cfg.eta_e_on_g * (self.cfg.xbar_eusd - Xe_t) + eps_g

    # Ensure third implied rate respects no-arbitrage exactly at next step by computing it from Xe_next and Xg_next
    Xeg_next = Xe_next / Xg_next if abs(Xg_next) > 1e-12 else Xe_next / (Xg_net + 1e-12)

    # Update inventories per trade conventions
    
    # When buy base currency i in pair i/j (a_ij > 0), base inventory increases, quote currency inventory decreases by price * a_ij
    # Paper uses q$ = USD inventory, q£=GBP, qe=EUR
    # Updates:
    # q$_t+1 = q$_t - Xe_t * a_eusd - Xg_t * a_gbusd
    # q£_t+1 = q£_t + a_gbusd - Xeg_t * a_eurgbp
    # qe_t+1 = qe_t + a_eusd + a_eurgbp
    self.q_usd = float(self.q_usd - Xe_t * a_eusd - Xg_t * a_gbusd)
    self.q_gbp = float(self.q_gbp + a_gbusd - Xeg_t * a_eurgbp)
    self.q_eur = float(self.q_eur + a_eusd + a_eurgbp)

    # compute reward according to eq. (3) (step by step)
    # R_t = (q£_t + a£_t - Xeg_t * ae£_t)*(Xg_{t+1}-Xg_t) + (qe_t + ae_t + ae£_t)*(Xe_{t+1}-Xe_t) - sum(phi * a^2)
    term1 = (self.q_gbp - a_gbusd + Xeg_t * a_eurgbp) # careful: q£t in paper denotes inventory before action; we have updated inventories. To match paper eq (3), we reconstruct q before next price move:
    # To be precise, I want (q£_t + a£_t - Xe£_t * ae£_t), where q£_t is the inventory BEFORE applying this step's trades
    # We saved inventories AFTER trades. So reconstruct q_before:
    q_gbp_before = self.q_gbp - (a_gbusd - Xeg_t * a_eurgbp)
    q_eur_before = self.q_eur - (a_eusd + a_eurgbp)
    expr1 = (q_gbp_before + a_gbusd - Xeg_t * a_eurgbp) * (Xg_next - Xg_t)
    expr2 = (q_eur_before + a_eusd + a_eurgbp) * (Xe_next - Xe_t)
    cost = self.cfg.phi_gbusd * (a_gbusd ** 2) + self.cfg.phi_eusd * (a_eusd ** 2) + self.cfg.phi_eurgbp * (a_eurgbp ** 2)
    reward = float(expr1 + expr2 - cost)

    # advance prices to next
    self.Xe = float(Xe_next)
    self.Xg = float(Xg_next)
    # increment time
    self.current_step += 1
    self.t = self.current_step

    done = (self.current_step >= self.cfg.T)
    # terminal penalty if done: -alpha_e*(q_eur)^2 - alpha_gbp*(q_gbp)^2 (convert to USD for paper they are USD units, but here q are in base units so treat params as consistent)
    if done:
        terminal_pen = -(self.cfg.alpha_eur * (self.q_eur ** 2) + self.cfg.alpha_gbp * (self.q_gbp ** 2))
        reward += terminal_pen

    obs = self._get_obs()
    info = {
        "Xe": self.Xe,
        "Xg": self.Xg,
        "Xeg": self.implied_eur_gbp(),
        "q_usd": self.q_usd,
        "q_gbp": self.q_gbp,
        "q_eur": self.q_eur,
        "step": self.current_step
    }
    return obs, reward, done, info


# ## observation getter

# In[ ]:


def _get_obs(self):
    return np.array([float(self.t), float(self.Xe), float(self.Xg), float(self.q_eur), float(self.q_gbp)], dtype=np.float32)


# ## render

# In[ ]:


def render(self, mode="human"):
    print(f"t={self.t}, Xe(EUR/USD)={self.Xe:.6f}, Xg(GBP/USD)={self.Xg:.6f}, Xeg(EUR/GBP)={self.implied_eur_gbp():.6f}")
    print(f"q_eur={self.q_eur:.6f}, q_gbp={self.q_gbp:.6f}, q_usd={self.q_usd:.6f}")


# ## vectorized rollout (for testing / eval)

# In[ ]:


def rollout(self, policy_fn, n_steps: Optional[int] = None):
    """
    policy_fn(obs) -> action
    returns trajectory dict
    """
    if n_steps is None:
        n_steps = self.cfg.T
    obs = self.reset()
    traj = {"obs": [], "actions": [], "rewards": [], "infos": []}
    done = False
    for _ in range(n_steps):
        action = policy_fn(obs)
        obs, r, done, info = self.step(action)
        traj["obs"].append(obs)
        traj["actions"].append(action)
        traj["rewards"].append(r)
        traj["infos"].append(info)
        if done:
            break
    return traj


# ## small unit tests

# In[ ]:


def _test_no_arbitrage_enforced():
    cfg = FXTripletConfig(seed=42)
    env = FXTripletEnv(cfg)
    obs = env.reset()
    # Step with zero noise (set sigma=0) and zero action, prices should revert deterministically
    cfg_zero = FXTripletConfig(seed=1, sigma=0.0)
    env2 = FXTripletEnv(cfg_zero)
    obs2 = env2.reset()
    # store initial implied
    implied0 = env2.implied_eur_gbp()
    # step with zero action
    obs_next, r, done, info = env2.step(np.array([0.0, 0.0, 0.0]))
    # check implied relation holds exactly (within numerical tolerance)
    implied_next = info["Xeg"]
    Xe = info["Xe"]; Xg = info["Xg"]
    implied_calc = Xe / Xg
    assert abs(implied_next - implied_calc) < 1e-12, "No-arbitrage implied rate not enforced exactly"
    print("TEST PASSED: No-arbitrage is enforced (implied_eur_gbp == Xe / Xg).")

def _test_reward_deterministic():
    # deterministic price drift: set dynamics so that Xe increases by +d and Xg increases by +d, sigma=0
    cfg = FXTripletConfig(seed=123, sigma=0.0, kappa_e=0.0, kappa_g=0.0, eta_e_on_g=0.0, eta_g_on_e=0.0)
    # We'll set initial prices and then force externals by directly patching environment next-step values? Simpler:
    # For this deterministic test, set eps via RNG to 0 and use manual small price change by temporarily modifying cfg.
    env = FXTripletEnv(cfg)
    env.reset()
    # monkeypatch prices so Xe_t = 1.0, Xg_t = 2.0
    env.Xe = 1.0
    env.Xg = 2.0
    # I'll pick a simple action: buy 1 EUR via EUR/USD (a_eusd=1), no other trades
    # Set cfg such that next-step Xe increases by +0.01 and Xg by +0.02 by directly overriding the model random draws.
    # Easiest: temporarily set rng.normal to return specified epsilons
    class DummyRNG:
        def __init__(self, eps_e, eps_g):
            self.eps_e = eps_e
            self.eps_g = eps_g
        def normal(self, loc, scale):
            # called twice; return eps_e, then eps_g
            if scale == cfg.sigma:
                return 0.0
            return 0.0
        def __call__(self, *args, **kwargs):
            return 0.0

    # Instead of monkeypatching RNG complexity, craft cfg so drift moves prices deterministically:
    env.cfg.kappa_e = 0.01
    env.cfg.kappa_g = 0.02
    env.cfg.xbar_eusd = env.Xe + 1.0  # forces positive drift for Xe: Xe_next - Xe ~= 0.01 * 1.0
    env.cfg.xbar_gbusd = env.Xg + 1.0  # similarly for Xg
    # run step with action [1,0,0]
    obs_before = env._get_obs().copy()
    obs_next, reward, done, info = env.step(np.array([1.0, 0.0, 0.0]))
    # compute expected step reward manually per eq (3)
    # Compute q_before
    q_gbp_before = 0.0
    q_eur_before = 0.0
    a_eusd = 1.0
    a_gbusd = 0.0
    a_eurgbp = 0.0
    Xe_t = 1.0; Xg_t = 2.0
    # next price estimates
    Xe_next = Xe_t + env.cfg.kappa_e * (env.cfg.xbar_eusd - Xe_t)
    Xg_next = Xg_t + env.cfg.kappa_g * (env.cfg.xbar_gbusd - Xg_t)
    expr1 = (q_gbp_before + a_gbusd - (Xe_t / Xg_t) * a_eurgbp) * (Xg_next - Xg_t)
    expr2 = (q_eur_before + a_eusd + a_eurgbp) * (Xe_next - Xe_t)
    cost = env.cfg.phi_eusd * (a_eusd**2) + env.cfg.phi_gbusd*(a_gbusd**2) + env.cfg.phi_eurgbp*(a_eurgbp**2)
    expected_reward = expr1 + expr2 - cost
    assert abs(reward - expected_reward) < 1e-8, f"Reward mismatch: got {reward}, expected {expected_reward}"
    print("TEST PASSED: Deterministic reward computation matches expected formula.")

if __name__ == "__main__":
    print("Running FXTripletEnv unit tests...")
    _test_no_arbitrage_enforced()
    _test_reward_deterministic()
    print("All tests passed ✅")


# ## run tests

# In[ ]:


"""
def _test_no_arbitrage_enforced():
        cfg = FXTripletConfig(seed=0)
        env = FXTripletEnv(cfg)
        obs = env.reset()
        print("Initial obs:", obs)
        action = np.array([0.2, -0.1, 0.0])
        obs, r, done, info = env.step(action)
def _test_reward_deterministic():
        print("Dummy reward test skipped in this minimal version.")

if __name__ == "__main__":
    _test_no_arbitrage_enforced()
    _test_reward_deterministic()
    """


# In[ ]:
