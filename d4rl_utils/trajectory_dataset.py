#!/usr/bin/env python3
"""
PyTorch Dataset을 상속받는 D4RL Trajectory Dataset
Bamba를 위한 간단한 trajectory 처리
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
import warnings
warnings.filterwarnings('ignore')


class D4RLTrajectoryDataset(Dataset):
    def __init__(self, env: str, dataset_type: str, dataset_dir: str = "./datasets", 
                 context_length: int = 20):
        """
        D4RL Trajectory Dataset 초기화
        
        Args:
            env: 환경 이름 (예: 'hopper', 'walker2d', 'halfcheetah', 'ant')
            dataset_type: 데이터셋 타입 ('random', 'medium', 'expert', 'medium-replay', 'medium-expert')
            dataset_dir: HDF5 데이터셋이 저장된 디렉토리
            context_length: trajectory의 길이
        """
        self.env = env
        self.dataset_type = dataset_type
        self.dataset_dir = Path(dataset_dir)
        self.context_length = context_length
        
        # trajectories 초기화
        self.trajectories = []
        
        # 데이터셋 로드
        self._load_dataset()
        
        # 정규화 통계 계산 및 적용
        self._normalize_data()
        
        print(f"✅ {env}-{dataset_type}-v2: {len(self.trajectories)}개 trajectory 로드 완료")
    
    def _load_dataset(self):
        """HDF5 데이터셋을 trajectory 형태로 변환"""
        import h5py
        
        print(f"🚀 {self.env}-{self.dataset_type}-v2 데이터셋 로딩 시작...")
        
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"데이터셋 디렉토리를 찾을 수 없습니다: {self.dataset_dir}")
        
        # 특정 환경과 데이터셋 타입에 해당하는 HDF5 파일 찾기
        target_filename = f"{self.env}-{self.dataset_type}-v2.hdf5"
        hdf5_file = self.dataset_dir / target_filename
        
        if not hdf5_file.exists():
            raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {target_filename}")
        
        print(f"📁 로딩할 파일: {hdf5_file.name}")
        
        # HDF5 파일 로드
        with h5py.File(hdf5_file, 'r') as f:
            keys = list(f.keys())
            data = {}
            
            for key in keys:
                if key in f and isinstance(f[key], h5py.Dataset):
                    data[key] = f[key][:]
        
        # Trajectory 생성
        self.trajectories = self._create_trajectories(data)
        
        print(f"🎉 {self.env}-{self.dataset_type}-v2 데이터셋 로딩 완료")
    
    def _create_trajectories(self, data: Dict) -> List[Dict]:
        """데이터를 trajectory 형태로 변환"""
        observations = data['observations']
        actions = data['actions']
        rewards = data['rewards']
        next_observations = data['next_observations']
        terminals = data['terminals']
        
        # timeouts가 있는 경우 사용
        use_timeouts = 'timeouts' in data
        if use_timeouts:
            timeouts = data['timeouts']
        
        # episode 단위로 데이터 처리
        N = len(rewards)
        trajectories = []
        
        # episode 경계 찾기
        if use_timeouts:
            episode_ends = np.where(np.logical_or(terminals, timeouts))[0]
        else:
            episode_ends = np.where(terminals)[0]
        
        # episode별로 데이터 분리
        episode_start = 0
        for episode_end in episode_ends:
            # episode 데이터 추출
            episode_data = {
                'observations': observations[episode_start:episode_end + 1],
                'next_observations': next_observations[episode_start:episode_end + 1],
                'actions': actions[episode_start:episode_end + 1],
                'rewards': rewards[episode_start:episode_end + 1],
                'terminals': terminals[episode_start:episode_end + 1]
            }
            
            trajectories.append(episode_data)
            episode_start = episode_end + 1
        
        return trajectories
    
    def _normalize_data(self):
        """데이터 정규화"""
        # 모든 trajectory의 상태를 연결하여 통계 계산
        states = []
        for traj in self.trajectories:
            states.append(traj['observations'])
        
        states = np.concatenate(states, axis=0)
        self.state_mean = np.mean(states, axis=0)
        self.state_std = np.std(states, axis=0) + 1e-6
        
        # 상태 정규화
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std
            traj['next_observations'] = (traj['next_observations'] - self.state_mean) / self.state_std
    
    def get_state_stats(self):
        """상태 정규화 통계 반환"""
        return self.state_mean, self.state_std
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        """단일 trajectory 반환"""
        traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]
        
        if traj_len >= self.context_length:
            # context_length보다 긴 경우 랜덤 슬라이싱
            si = random.randint(0, traj_len - self.context_length)
            
            states = torch.FloatTensor(traj['observations'][si:si + self.context_length])
            next_states = torch.FloatTensor(traj['next_observations'][si:si + self.context_length])
            actions = torch.FloatTensor(traj['actions'][si:si + self.context_length])
            rewards = torch.FloatTensor(traj['rewards'][si:si + self.context_length])
            terminals = torch.FloatTensor(traj['terminals'][si:si + self.context_length])
            
            # 마지막 액션과 보상만 사용 (현재 시점)
            current_action = actions[-1]
            current_reward = rewards[-1].unsqueeze(0)  # (1,)로 차원 맞추기
            current_done = terminals[-1].unsqueeze(0)  # (1,)로 차원 맞추기
            
        else:
            # context_length보다 짧은 경우 패딩
            padding_len = self.context_length - traj_len
            
            # 상태 패딩
            states = torch.FloatTensor(traj['observations'])
            states = torch.cat([states, torch.zeros(padding_len, states.shape[1])], dim=0)
            
            next_states = torch.FloatTensor(traj['next_observations'])
            next_states = torch.cat([next_states, torch.zeros(padding_len, next_states.shape[1])], dim=0)
            
            # 액션 패딩
            actions = torch.FloatTensor(traj['actions'])
            actions = torch.cat([actions, torch.zeros(padding_len, actions.shape[1])], dim=0)
            
            # 보상 패딩 (1차원)
            rewards = torch.FloatTensor(traj['rewards'])
            rewards = torch.cat([rewards, torch.zeros(padding_len)], dim=0)
            
            # done 플래그 패딩 (1차원)
            terminals = torch.FloatTensor(traj['terminals'])
            terminals = torch.cat([terminals, torch.zeros(padding_len)], dim=0)
            
            # 마지막 액션과 보상만 사용 (현재 시점)
            current_action = actions[traj_len - 1] if traj_len > 0 else torch.zeros(actions.shape[1])
            current_reward = rewards[traj_len - 1].unsqueeze(0) if traj_len > 0 else torch.zeros(1)
            current_done = terminals[traj_len - 1].unsqueeze(0) if traj_len > 0 else torch.zeros(1)
        
        return {
            'states': states,                    # (context_length, state_dim)
            'next_states': next_states,          # (context_length, state_dim)
            'actions': current_action,           # (action_dim,)
            'rewards': current_reward,           # (1,)
            'dones': current_done,               # (1,)
            'traj_len': traj_len                # 원본 trajectory 길이
        }


def main():
    """사용 예시"""
    # 데이터셋 초기화
    dataset = D4RLTrajectoryDataset(env="hopper", dataset_type="medium", context_length=20)
    
    # 데이터셋 정보
    state_mean, state_std = dataset.get_state_stats()
    print(f"\n📊 데이터셋 정보:")
    print(f"   Trajectory 수: {len(dataset)}")
    print(f"   Context 길이: {dataset.context_length}")
    print(f"   State 차원: {state_mean.shape[0]}")
    
    # 샘플 데이터 가져오기
    sample = dataset[0]
    print(f"\n🎯 샘플 데이터 형태:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape}")
        else:
            print(f"   {key}: {value}")


if __name__ == "__main__":
    main()
