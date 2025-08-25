import torch
from torch import nn
import torch.nn.functional as F
from model import Mamba_Encoder, Q_net

class Bamba:
    def __init__(self, state_dim, action_dim, hidden_dim, 
    context_length, d_model, d_state, d_conv, expand, 
    tau = 0.01, alpha = 0.1, gamma = 0.99, policy_freq = 2,
    action_noise = 0.2, action_noise_clip = 0.5, device = "cuda",
    mamba_lr = 3e-4, critic_lr = 3e-4, n_layers = 1, warmup_steps = 1000,
    drop_p = 0.1, beta_s = 1.0, beta_r = 1.0,
):

        # Environment Parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.context_length = context_length

        # Mamba Parameters
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.n_layers = n_layers
        self.warmup_steps = warmup_steps
        self.mamba_lr = mamba_lr
        self.critic_lr = critic_lr
        self.beta_s = beta_s
        self.beta_r = beta_r

        # Hyperparameters
        self.tau = tau
        self.alpha = alpha
        self.gamma = gamma
        self.policy_freq = policy_freq
        self.action_noise = action_noise
        self.action_noise_clip = action_noise_clip
        
        # Device
        self.device = device

        # Model
        self.encoder = Mamba_Encoder(state_dim, context_length, d_model, d_state, d_conv, expand, n_layers, drop_p).to(device)
        self.critic = Q_net(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Q_net(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.action_selector = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        ).to(device)

        self.action_selector_target = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        ).to(device)
        self.action_selector_target.load_state_dict(self.action_selector.state_dict())

        self.next_state_estimator = nn.Sequential(
            nn.Linear(d_model + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        ).to(self.device)

        self.reward_estimator = nn.Sequential(
            nn.Linear(d_model + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

        # Optimizers
        self.mamba_optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.action_selector.parameters()) + list(self.next_state_estimator.parameters()) + list(self.reward_estimator.parameters()), lr=self.mamba_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
        # Warmup schedulers
        self.mamba_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.mamba_optimizer, 
            lambda step: min(1.0, step / warmup_steps)  # 1000 steps 동안 warmup
        )
        self.critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.critic_optimizer, 
            lambda step: min(1.0, step / warmup_steps)  # 1000 steps 동안 warmup
        )

        self.update_num = 0

    def select_action(self, state_latent):
        # state_latent: (batch_size, d_model) - already encoded trajectory representation
        action = self.action_selector(state_latent)
        return action
    
    def update(self, states, actions, rewards, next_states, dones):
        # shapes:
        # states, next_states: (B, K, state_dim)
        # actions: (B, action_dim)
        # rewards, dones: (B, 1)

        rewards = rewards.float()
        dones = dones.float()

        state_latent = self.encoder.encode_trajectory(states)        # (B, H)
        action_preds = self.action_selector(state_latent)            # (B, A)
        current_state = states[:, -1, :]
        next_state = next_states[:, -1, :]

        with torch.no_grad():
            next_state_latent = self.encoder.encode_trajectory(next_states)   # (B, H)

            noise = (torch.randn_like(action_preds) * self.action_noise)\
                    .clamp(-self.action_noise_clip, self.action_noise_clip)

            next_action = (self.action_selector_target(next_state_latent) + noise).clamp(-1, 1)

            next_q1, next_q2 = self.critic_target(next_state, next_action)
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards + self.gamma * (1.0 - dones) * next_q

        current_q1, current_q2 = self.critic(current_state, actions)

        td_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad(set_to_none=True)
        td_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        self.critic_scheduler.step()

        s_target = next_states[:, -1, :]                     # (B, state_dim)

        # aux 입력은 현재 표현 + 현재 action
        sa_feat  = torch.cat([state_latent, actions], dim=-1)
        s_hat    = self.next_state_estimator(sa_feat)        # (B, state_dim)
        r_hat    = self.reward_estimator(sa_feat)            # (B, 1)

        state_loss  = F.mse_loss(s_hat, s_target)
        reward_loss = F.mse_loss(r_hat, rewards)
        encoder_loss = self.beta_s * state_loss + self.beta_r * reward_loss

        # -------------------------
        # Actor + encoder update (policy_freq)
        # -------------------------
        if self.update_num % self.policy_freq == 0:
            Q1_pi = self.critic.Q1(current_state, action_preds)  # (B, 1)
            denom = Q1_pi.abs().mean().clamp(min=1e-6).detach()
            lmbda = self.alpha / denom

            bc_loss   = F.mse_loss(action_preds, actions)
            actor_loss = -lmbda * Q1_pi.mean() + bc_loss

            total_loss = encoder_loss + actor_loss

            self.mamba_optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters())
            + list(self.action_selector.parameters())
            + list(self.next_state_estimator.parameters())
            + list(self.reward_estimator.parameters()),
                max_norm=1.0
            )
            self.mamba_optimizer.step()
            self.mamba_scheduler.step()  
            self.update_target_networks()
        else:
            self.mamba_optimizer.zero_grad(set_to_none=True)
            encoder_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters())
            + list(self.next_state_estimator.parameters())
            + list(self.reward_estimator.parameters()),
                max_norm=1.0
            )
            self.mamba_optimizer.step()
            self.mamba_scheduler.step()

        self.update_num += 1
        return td_loss.item(), encoder_loss.item()
    
    def update_target_networks(self):
        """Target network 업데이트 (에포크마다 호출)"""
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.action_selector.parameters(), self.action_selector_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def get_current_lr(self):
        """현재 learning rate 반환"""
        return {
            'mamba_lr': self.mamba_scheduler.get_last_lr()[0],
            'critic_lr': self.critic_scheduler.get_last_lr()[0]
        }