#!/usr/bin/env python3
"""
D4RL 데이터셋 다운로더
지정된 디렉토리에 D4RL 데이터셋을 다운로드하고 저장합니다.
"""

import os
import argparse
from typing import List, Optional
import gym
from d4rl import offline_env
from d4rl.infos import DATASET_URLS, REF_MIN_SCORE, REF_MAX_SCORE


def download_dataset(env_name: str, dataset_type: str = "medium", dataset_dir: str = "./datasets") -> str:
    """
    D4RL 데이터셋을 다운로드하고 저장합니다.
    
    Args:
        env_name: 환경 이름 (예: 'hopper', 'walker2d', 'halfcheetah')
        dataset_type: 데이터셋 타입 ('random', 'medium', 'expert', 'medium-replay', 'medium-expert')
        dataset_dir: 데이터셋을 저장할 디렉토리 경로
        
    Returns:
        저장된 데이터셋의 경로
    """
    # 데이터셋 디렉토리 생성
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 환경 이름 정규화
    if not env_name.startswith(('hopper', 'walker2d', 'halfcheetah', 'ant')):
        env_name = f"{env_name}-v2"
    
    # 실제로 존재하는 데이터셋인지 확인
    full_env_name = f"{env_name}-{dataset_type}-v2"
    if full_env_name not in DATASET_URLS:
        print(f"❌ {full_env_name} 데이터셋은 존재하지 않습니다.")
        print(f"   사용 가능한 데이터셋을 확인하려면 --list 옵션을 사용하세요.")
        return None
    
    # D4RL 환경 생성 (자동으로 데이터셋 다운로드)
    try:
        env = gym.make(full_env_name)
        print(f"✅ {full_env_name} 데이터셋 다운로드 완료")
        
        # 데이터셋 정보 출력
        dataset = env.get_dataset()
        print(f"📊 데이터셋 크기: {len(dataset['observations'])} 스텝")
        print(f"📊 관찰 차원: {dataset['observations'].shape}")
        print(f"📊 행동 차원: {dataset['actions'].shape}")
        print(f"📊 보상 차원: {dataset['rewards'].shape}")
        
        # 데이터셋을 지정된 디렉토리에 저장
        dataset_path = os.path.join(dataset_dir, f"{full_env_name}.hdf5")
        
        # HDF5 파일로 저장
        import h5py
        with h5py.File(dataset_path, 'w') as f:
            for key, value in dataset.items():
                f.create_dataset(key, data=value)
        
        print(f"💾 데이터셋이 {dataset_path}에 저장되었습니다.")
        
        env.close()
        return dataset_path
        
    except Exception as e:
        print(f"❌ 데이터셋 다운로드 실패: {e}")
        return None


def download_all_datasets(dataset_dir: str = "./datasets") -> List[str]:
    """
    실제로 존재하는 모든 D4RL 데이터셋을 다운로드합니다.
    
    Args:
        dataset_dir: 데이터셋을 저장할 디렉토리 경로
        
    Returns:
        다운로드된 데이터셋 경로들의 리스트
    """
    envs = ['hopper', 'walker2d', 'halfcheetah', 'ant']
    dataset_types = ['random', 'medium', 'expert', 'medium-replay', 'medium-expert']
    
    downloaded_paths = []
    
    for env in envs:
        for dataset_type in dataset_types:
            env_name = f"{env}-{dataset_type}-v2"
            # 실제로 존재하는 데이터셋만 다운로드 시도
            if env_name in DATASET_URLS:
                try:
                    path = download_dataset(env, dataset_type, dataset_dir)
                    if path:
                        downloaded_paths.append(path)
                except Exception as e:
                    print(f"⚠️ {env}-{dataset_type} 다운로드 실패: {e}")
                    continue
            else:
                print(f"⏭️ {env}-{dataset_type} 데이터셋은 존재하지 않아 건너뜁니다.")
    
    return downloaded_paths


def list_available_datasets():
    """사용 가능한 D4RL 데이터셋 목록을 출력합니다."""
    print("📋 사용 가능한 D4RL 데이터셋:")
    print("=" * 50)
    
    # 실제로 존재하는 데이터셋만 필터링
    available_datasets = {}
    
    for env in ['hopper', 'walker2d', 'halfcheetah', 'ant']:
        available_datasets[env] = []
        for dataset_type in ['random', 'medium', 'expert', 'medium-replay', 'medium-expert']:
            env_name = f"{env}-{dataset_type}-v2"
            if env_name in DATASET_URLS:
                available_datasets[env].append((dataset_type, env_name))
    
    # 실제 존재하는 데이터셋만 출력
    for env, datasets in available_datasets.items():
        if datasets:  # 데이터셋이 있는 환경만 출력
            print(f"\n🏃 {env.upper()}:")
            for dataset_type, env_name in datasets:
                print(f"  • {dataset_type}: {env_name}")
                if env_name in REF_MIN_SCORE and env_name in REF_MAX_SCORE:
                    print(f"    참조 점수: {REF_MIN_SCORE[env_name]:.1f} ~ {REF_MAX_SCORE[env_name]:.1f}")
                else:
                    print(f"    참조 점수: 정보 없음")


def main():
    parser = argparse.ArgumentParser(description="D4RL 데이터셋 다운로더")
    parser.add_argument("--env", type=str, help="환경 이름 (예: hopper, walker2d)")
    parser.add_argument("--dataset_type", type=str, default="medium", 
                       help="데이터셋 타입 (random, medium, expert, medium-replay, medium-expert)")
    parser.add_argument("--dataset_dir", type=str, default="./datasets",
                       help="데이터셋을 저장할 디렉토리")
    parser.add_argument("--all", action="store_true", help="모든 데이터셋 다운로드")
    parser.add_argument("--list", action="store_true", help="사용 가능한 데이터셋 목록 출력")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_datasets()
        return
    
    if args.all:
        print("🚀 모든 D4RL 데이터셋 다운로드를 시작합니다...")
        downloaded = download_all_datasets(args.dataset_dir)
        print(f"\n✅ 총 {len(downloaded)}개 데이터셋 다운로드 완료")
        return
    
    if not args.env:
        print("❌ 환경 이름을 지정해주세요. --help로 도움말을 확인하세요.")
        return
    
    print(f"🚀 {args.env}-{args.dataset_type}-v2 데이터셋 다운로드를 시작합니다...")
    download_dataset(args.env, args.dataset_type, args.dataset_dir)


if __name__ == "__main__":
    main()
