#!/usr/bin/env python3
"""
유틸리티 함수들
"""

import torch
import numpy as np
import gym
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


def set_seed(seed: int = 42):
    """시드 설정"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def load_checkpoint(checkpoint_path: str, device: str = "auto"):
    """체크포인트 로드"""
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint, device

def evaluate_bamba_on_environment(model, 
                                device, 
                                context_len, 
                                env_name, 
                                num_eval_ep=10, 
                                max_test_ep_len=1000,
                                state_mean=None, 
                                state_std=None, 
                                render=False):
    """
    Bamba 에이전트를 실제 환경에서 평가
    
    Args:
        model: Bamba 에이전트
        device: 사용할 디바이스
        context_len: context 길이
        env: 환경 이름 또는 환경 객체
        num_eval_ep: 평가할 에피소드 수
        max_test_ep_len: 최대 에피소드 길이
        state_mean: 상태 정규화 평균
        state_std: 상태 정규화 표준편차
        render: 렌더링 여부
        
    Returns:
        평가 결과 딕셔너리
    """
    results = {}
    total_reward = 0
    total_timesteps = 0
    
    # 환경 생성
    if isinstance(env_name, str):
        try:
            env = gym.make(env_name)
        except Exception as e:
            print(f"❌ 환경 생성 실패: {e}")
            return {}
    
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # 정규화 통계 설정
    if state_mean is None:
        state_mean = torch.zeros((state_dim,)).to(device)
    else:
        state_mean = torch.from_numpy(state_mean).to(device)
    
    if state_std is None:
        state_std = torch.ones((state_dim,)).to(device)
    else:
        state_std = torch.from_numpy(state_std).to(device)
    
    print(f"🌍 Bamba 환경 평가 시작: {env_name}")
    print(f"   State 차원: {state_dim}, Action 차원: {act_dim}")
    print(f"   Context 길이: {context_len}")
    
    for episode in range(num_eval_ep):
        # 환경 초기화
        running_state = env.reset()
        running_reward = 0
        
        # 상태 히스토리 초기화
        states = torch.zeros((1, max_test_ep_len, state_dim), device=device)
        states[0, 0] = torch.from_numpy(running_state).to(device)
        states[0, 0] = (states[0, 0] - state_mean) / state_std
        
        for t in range(1, max_test_ep_len):
            # context 길이에 따른 상태 선택
            if t < context_len:
                # 초기 단계: 패딩으로 context_length 맞추기
                current_states = states[:, :t+1]
                # 패딩으로 context_length 맞추기
                padding_len = context_len - (t + 1)
                if padding_len > 0:
                    padding = torch.zeros((1, padding_len, state_dim), device=device)
                    current_states = torch.cat([padding, current_states], dim=1)
            else:
                # 슬라이딩 윈도우
                current_states = states[:, t-context_len+1:t+1]
            
            # Bamba 에이전트로 액션 선택
            with torch.no_grad():
                # 상태를 Mamba 인코더로 인코딩
                state_latent = model.encoder.encode_trajectory(current_states)
                # 액션 선택
                action = model.select_action(state_latent)
                action = action[0].detach()  # 첫 번째 배치만 사용
            
            # 환경에서 행동 실행
            running_state, running_reward, done, _ = env.step(action.cpu().numpy())
            
            # 상태 저장
            states[0, t] = torch.from_numpy(running_state).to(device)
            states[0, t] = (states[0, t] - state_mean) / state_std
            
            total_reward += running_reward
            
            if render:
                env.render()
            if done:
                break
        
        total_timesteps += t + 1
    
    env.close()
    
    # 결과 계산
    avg_reward = total_reward / num_eval_ep
    
    # D4RL에서 정확한 성능 기준 가져오기
    try:
        import d4rl
        from d4rl import infos
        
        # 환경 이름에서 D4RL 환경명 추출
        env_name = str(env).split('<')[1].split('>')[0] if '<' in str(env) else str(env)
        
        # D4RL 환경명 매핑
        if 'hopper' in env_name.lower():
            d4rl_env_name = 'hopper-medium-v2'
        elif 'walker2d' in env_name.lower():
            d4rl_env_name = 'walker2d-medium-v2'
        elif 'halfcheetah' in env_name.lower():
            d4rl_env_name = 'halfcheetah-medium-v2'
        elif 'ant' in env_name.lower():
            d4rl_env_name = 'ant-medium-v2'
        else:
            d4rl_env_name = 'hopper-medium-v2'  # 기본값
        
        # D4RL에서 정확한 성능 기준 가져오기
        ref_min_score = infos.REF_MIN_SCORE[d4rl_env_name]
        ref_max_score = infos.REF_MAX_SCORE[d4rl_env_name]
        
        # D4RL 스코어 계산 (0-100 범위)
        d4rl_score = (avg_reward - ref_min_score) / (ref_max_score - ref_min_score) * 100
        d4rl_score = max(0, min(100, d4rl_score))  # 0-100 범위로 제한
        
    except Exception as e:
        print(f"⚠️  D4RL 스코어 계산 실패: {e}")
        d4rl_score = avg_reward
    
    results = {
        'avg_reward': avg_reward,
        'd4rl_score': d4rl_score
    }
    
    print(f"✅ Bamba 환경 평가 완료")
    print(f"   D4RL 스코어: {d4rl_score:.1f}/100")
    
    return results


def save_results(results: Dict, 
                output_dir: str = "results", 
                filename: str = None) -> str:
    """결과를 JSON 파일로 저장"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.json"
    
    filepath = output_path / filename
    
    # numpy 배열을 리스트로 변환 (JSON 직렬화 가능하게)
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, np.integer):
            serializable_results[key] = int(value)
        elif isinstance(value, np.floating):
            serializable_results[key] = float(value)
        else:
            serializable_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"💾 결과 저장됨: {filepath}")
    return str(filepath)


def load_results(filepath: str) -> Dict:
    """JSON 파일에서 결과 로드"""
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results


def print_model_info(model, device):
    """모델 정보 출력"""
    print(f"🔧 모델 정보:")
    print(f"   디바이스: {device}")
    
    if hasattr(model, 'encoder'):
        total_params = sum(p.numel() for p in model.encoder.parameters())
        trainable_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        print(f"   Encoder - 총 파라미터: {total_params:,}, 학습 가능: {trainable_params:,}")
    
    if hasattr(model, 'q_net'):
        total_params = sum(p.numel() for p in model.q_net.parameters())
        trainable_params = sum(p.numel() for p in model.q_net.parameters() if p.requires_grad)
        print(f"   Q-Net - 총 파라미터: {total_params:,}, 학습 가능: {trainable_params:,}")


def main():
    """사용 예시"""
    print("💡 유틸리티 함수들을 사용하려면:")
    print("   - set_seed(42)")
    print("   - checkpoint, device = load_checkpoint('path/to/checkpoint.pt')")
    print("   - results = evaluate_on_environment(model, device, context_len, env)")
    print("   - save_results(results, 'results', 'my_results.json')")


if __name__ == "__main__":
    main()