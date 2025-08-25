import os
import argparse
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings

# 경고 메시지 억제
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gym
import d4rl
from agent import Bamba
from utils import set_seed, evaluate_bamba_on_environment
from d4rl_utils.trajectory_dataset import D4RLTrajectoryDataset
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="hopper")
    parser.add_argument("--dataset_type", type=str, default="medium")
    parser.add_argument("--context_length", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--action_noise", type=float, default=0.2)
    parser.add_argument("--action_noise_clip", type=float, default=0.5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--beta_s", type=float, default=1.0)
    parser.add_argument("--beta_r", type=float, default=1.0)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_state", type=int, default=64)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--drop_p", type=float, default=0.1)

    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--policy_freq", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default="./datasets")

    return parser.parse_args()

def train_epoch(agent, dataloader, device):
    """한 에포크 훈련"""
    total_TD_loss = 0
    total_Mamba_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        # GPU로 데이터 이동
        states = batch['states'].to(device)
        next_states = batch['next_states'].to(device)
        actions = batch['actions'].to(device)
        rewards = batch['rewards'].to(device)
        dones = batch['dones'].to(device)
        
        # 에이전트 업데이트
        TD_loss, Mamba_loss = agent.update(states, actions, rewards, next_states, dones)
        
        total_TD_loss += TD_loss
        total_Mamba_loss += Mamba_loss
        num_batches += 1
    
    # num_batches가 0인 경우 방지
    if num_batches == 0:
        return 0.0, 0.0
    
    return total_TD_loss / num_batches, total_Mamba_loss / num_batches

def save_checkpoint(agent, epoch, 
                   normalization_stats, output_dir, filename=None):
    """체크포인트 저장"""
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pt"
    
    checkpoint = {
        'epoch': epoch,
        'agent_state_dict': {
            'encoder': agent.encoder.state_dict(),
            'critic': agent.critic.state_dict(),
            'critic_target': agent.critic_target.state_dict(),
            'action_selector': agent.action_selector.state_dict(),
        },
        'optimizer_states': {
            'mamba_optimizer': agent.mamba_optimizer.state_dict(),
            'critic_optimizer': agent.critic_optimizer.state_dict(),
        },
        'normalization_stats': normalization_stats,
        'hyperparameters': {
            'state_dim': agent.state_dim,
            'action_dim': agent.action_dim,
            'hidden_dim': agent.hidden_dim,
            'context_length': agent.context_length,
            'd_model': agent.d_model,
            'd_state': agent.d_state,
            'd_conv': agent.d_conv,
            'expand': agent.expand,
            'n_layers': agent.n_layers,
            'warmup_steps': agent.warmup_steps,
            'tau': agent.tau,
            'alpha': agent.alpha,
            'gamma': agent.gamma,
            'action_noise': agent.action_noise,
            'action_noise_clip': agent.action_noise_clip,
            'beta_s': agent.beta_s,
            'beta_r': agent.beta_r,
        }
    }
    
    checkpoint_path = os.path.join(output_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    print(f"✅ 체크포인트 저장됨: {checkpoint_path}")

def load_checkpoint(agent, checkpoint_path, device):
    """체크포인트 로드"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 에이전트 상태 복원
    agent.encoder.load_state_dict(checkpoint['agent_state_dict']['encoder'])
    agent.critic.load_state_dict(checkpoint['agent_state_dict']['critic'])
    agent.critic_target.load_state_dict(checkpoint['agent_state_dict']['critic_target'])
    agent.action_selector.load_state_dict(checkpoint['agent_state_dict']['action_selector'])
    
    # 옵티마이저 상태 복원
    if 'optimizer_states' in checkpoint:
        agent.mamba_optimizer.load_state_dict(checkpoint['optimizer_states']['mamba_optimizer'])
        agent.critic_optimizer.load_state_dict(checkpoint['optimizer_states']['critic_optimizer'])
        print("✅ 옵티마이저 상태 복원됨")
    
    # 정규화 통계 복원
    normalization_stats = checkpoint['normalization_stats']
    
    print(f"✅ 체크포인트 로드됨: {checkpoint_path}")
    return checkpoint['epoch'], normalization_stats

def main():
    args = parse_args()
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    # 시드 설정
    set_seed(args.seed)
    
    # 디바이스 설정
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🚀 훈련 시작 - 환경: {args.env}-{args.dataset_type}-v2, 디바이스: {device}")
    
    # D4RL Trajectory Dataset 초기화
    print("📊 D4RL Trajectory Dataset 초기화 중...")
    dataset = D4RLTrajectoryDataset(
        env=args.env,
        dataset_type=args.dataset_type,
        dataset_dir=args.dataset_dir,
        context_length=args.context_length
    )
    
    # 데이터로더 생성
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,  # Windows 호환성을 위해 0으로 설정
        drop_last=True
    )
    
    # 데이터셋 정보 가져오기
    state_mean, state_std = dataset.get_state_stats()
    normalization_stats = {
        'observation_mean': state_mean,
        'observation_std': state_std
    }
    
    print(f"📈 데이터셋 크기: {len(dataset)} 시퀀스")
    print(f"🔧 상태 차원: {dataset[0]['states'].shape[1]}, 액션 차원: {dataset[0]['actions'].shape[0]}")
    print(f"🔧 데이터셋 통계 - 평균: {state_mean}, 표준편차: {state_std}")
    # 에이전트 생성
    print("🤖 Bamba 에이전트 생성 중...")
    agent = Bamba(
        state_dim=dataset[0]['states'].shape[1],
        action_dim=dataset[0]['actions'].shape[0],
        hidden_dim=args.hidden_dim,
        context_length=args.context_length,
        d_model=args.d_model,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        tau=args.tau,
        alpha=args.alpha,
        gamma=args.gamma,
        action_noise=args.action_noise,
        action_noise_clip=args.action_noise_clip,
        n_layers=args.n_layers,
        device=device,
        warmup_steps=args.warmup_steps,
        policy_freq=args.policy_freq,
        drop_p=args.drop_p,
        beta_s=args.beta_s,
        beta_r=args.beta_r
    )
    
    # 체크포인트에서 복원
    start_epoch = 0
    if args.resume:
        print(f"🔄 체크포인트에서 복원 중: {args.resume}")
        start_epoch, normalization_stats = load_checkpoint(
            agent, args.resume, device
        )
        start_epoch += 1
    
    # 훈련 루프
    print("🎯 훈련 시작!")
    
    best_reward = float('-inf')
    training_logs = []
    
    for epoch in range(start_epoch, args.num_epochs):
        # 한 에포크 훈련
        critic_loss, mamba_loss = train_epoch(agent, dataloader, device)
        
        # 로그 기록
        log_entry = {
            'epoch': epoch,
            'critic_loss': critic_loss,
            'mamba_loss': mamba_loss,
            'timestamp': datetime.now().isoformat()
        }
        training_logs.append(log_entry)
        
        # 10 에포크마다 구분선과 함께 상세 로그 출력
        if epoch % 10 == 0:
            current_lr = agent.get_current_lr()
            print("=" * 60)
            print(f"📊 EPOCH {epoch:4d} 상세 훈련 결과")
            print(f"   Critic Loss: {critic_loss:.6f}")
            print(f"   Mamba Loss: {mamba_loss:.6f}")
            print(f"   Mamba LR: {current_lr['mamba_lr']:.2e}")
            print(f"   Critic LR: {current_lr['critic_lr']:.2e}")
            print("=" * 60)
        
        # 평가 (첫 번째 에포크 제외)
        if epoch > 0 and epoch % args.eval_freq == 0:
            print(f"🔍 Epoch {epoch} 평가 중...")
            eval_results = evaluate_bamba_on_environment(
                model=agent,
                device=device,
                context_len=args.context_length,
                env_name=f"{args.env}-{args.dataset_type}-v2",
                num_eval_ep=5,
                max_test_ep_len=1000,
                state_mean=state_mean,
                state_std=state_std,
                render=False
            )
            
            if eval_results:
                avg_reward = eval_results.get('avg_reward', 0)
                d4rl_score = eval_results.get('d4rl_score', 0)
                log_entry['eval_avg_reward'] = avg_reward
                log_entry['eval_d4rl_score'] = d4rl_score
                print(f"🎯 평가 결과 - 평균 보상: {avg_reward:.2f} | D4RL 스코어: {d4rl_score:.1f}/100")
                
                # 최고 성능 체크포인트 저장
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    save_checkpoint(
                        agent, epoch,
                        normalization_stats, output_dir, "best_checkpoint.pt"
                    )
        
        # 정기 체크포인트 저장
        if epoch % args.save_freq == 0:
            save_checkpoint(
                agent, epoch,
                normalization_stats, output_dir
            )
    
    # 최종 체크포인트 저장
    save_checkpoint(
        agent, args.num_epochs - 1,
        normalization_stats, output_dir, "final_checkpoint.pt"
    )
    
    # 훈련 로그 저장
    with open(output_dir / 'training_logs.json', 'w') as f:
        json.dump(training_logs, f, indent=2)
    
    print("🎉 훈련 완료!")
    print(f"📁 결과 저장 위치: {output_dir}")

if __name__ == "__main__":
    main()
