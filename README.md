# FastGen Mid-Evaluation Submission

이 디렉터리는 1-NFE와 최대 4-NFE Pokemon 생성 모델의 **평가에 필요한 전체 소스 코드와 고정 체크포인트**를 포함한다. TAs는 이 디렉터리에서 `model.py`와 `evaluate.py`를 그대로 실행할 수 있다.

## 필수 체크포인트

평가기가 요구하는 상대 경로와 파일명을 그대로 유지한다.

| 평가 모드 | 체크포인트 경로 | NFE | 파라미터 수 | Official FID | Valid FID | SHA256 |
|---|---|---:|---:|---:|---:|---|
| 1-NFE (E2b K=2, 2,000 step) | `./checkpoints/one_nfe.ckpt` | 정확히 1 | 48,883,107 | 37.87* | 54.875467 | `a2d57781111ed0e2df8c64ed1ac3ce408a02dc52943e7f6da9ba691f8b5f21a4` |
| Few-NFE | `./checkpoints/few_nfe.ckpt` | 정확히 4 | 48,883,107 | 32.664720 | 47.703815 | `e99cb3a050c21084acb04c6d0ae80d73b225307cd232f6a8322efbedc977445d` |

두 모델은 각각 100M 제한 아래다. 점수 프로토콜: *37.87은 TA `evaluate.py` 기본 설정(batch 32)의 Official FID 보고값이다. E2b의 별도 seed 비교(batch 16, seed 42/1042/2042)는 38.178126 / 38.251802 / 38.754697, 평균 38.394875이며, 두 프로토콜 점수는 서로 섞어 비교하지 않는다. 54.875467은 학습 시 dev FID다. 체크포인트 SHA256은 제출물 내부 파일에서 계산했다. Few-NFE 점수는 기존 기록(Official 32.664720, Valid 47.703815)이다.

## 모델 및 추론 구조

- 입력/출력은 64×64 RGB 이미지이며, 픽셀 값은 `[-1, 1]` 범위다. 조건은 151개 Pokemon 클래스다.
- 두 checkpoint는 같은 48,883,107-parameter class-conditional U-Net 구조를 쓴다: base width 96, channel multipliers `(1, 2, 3, 4)`, level당 residual block 2개, 16×16 해상도 attention, velocity (`v`) prediction.
- 모델 생성 시 [configs/submission_model.json](configs/submission_model.json)을 `model.py`가 자동으로 읽는다. 별도 수동 변환이나 환경 변수 설정은 필요하지 않다.
- 1-NFE는 timestep `[999]`에서 U-Net을 한 번 실행한다. Few-NFE는 `[999, 741, 483, 226]`에서 한 번씩 실행해 총 네 번 호출한다. 설정된 timestep마다 backbone 호출은 하나뿐이다.
- Classifier-free guidance의 조건은 학습된 student에 반영되어 있다. 추론 중 unconditional/conditional 두 번째 호출, teacher, discriminator, encoder, 다른 auxiliary learned network를 실행하지 않는다.
- 출력은 DDIM 형태의 velocity-to-clean 변환과 deterministic update를 거쳐 `[-1, 1]`로 제한한다.

## 학습 및 제한 준수 요약

- 외부 pretrained 생성 모델이나 pretrained weight를 사용하지 않았다. Teacher와 student는 과제 데이터 및 내부에서 scratch 학습한 모델 계보로 만들었다. Inception은 과제 제공 FID 평가 코드의 예외다.
- NFE-1은 E2b K=2 설정으로 DMD-inspired one-step student를 2,000 update 학습한 best-dev checkpoint다. 설정은 generator LR `2e-5`, fake-score LR `2e-4`, endpoint regression weight `0.1`, CFG `2.0`이다. Best checkpoint는 step 2,000이며 이후 step 2,500에서 dev FID가 악화되어 step 2,000을 선택했다. 기록된 peak training VRAM은 4,616,571,392 bytes다.
- NFE-4는 네 단계 student에 실제 train 이미지의 conditional hinge-GAN 신호와 원본 student 출력을 보존하는 anchor를 적용한 1,000-update EMA checkpoint다.
- 제공된 `train_split.txt`에 포함된 데이터만 학습에 사용했다. 일부 선택용 755장 development holdout은 제공된 train split에서 분리한 것으로, 제공된 `val_split.txt` validation/reference 이미지는 학습에 넣지 않았다.
- 학습 중인 NFE-1 generator, real-score model, fake-score model은 각각 48,883,107 parameters다. NFE-4 generator/frozen anchor는 각각 48,883,107, discriminator는 1,746,625 parameters다. 모두 개별 100M 미만이다. 기록된 최대 training VRAM은 NFE-1 E2b DMD **4,616,571,392 bytes**, NFE-4 GAN **2,115,794,944 bytes**로 각각 20GB보다 작다.
- sampling 시 등록되는 learned model은 각 모드의 48,883,107-parameter U-Net 하나다.

## 전체 실행 Bash 예시

아래를 제출 디렉터리에서 실행하면 의존성 설치, 데이터 준비, 두 checkpoint의 FID 평가를 순서대로 수행한다. `PokemonData/`가 이미 준비되어 있으면 기존 파일을 사용하고, 없으면 원본 downloader가 KaggleHub로 내려받는다. 실행 환경에 호환 가능한 PyTorch/CUDA가 이미 있으면 같은 Python 환경을 `PYTHON`으로 지정한다.

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python3}"

"$PYTHON" -m pip install -r requirements.txt

DEVICE=cpu
if "$PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
  DEVICE=cuda
fi

test -f ./checkpoints/one_nfe.ckpt
test -f ./checkpoints/few_nfe.ckpt

# 이미지가 없으면 KaggleHub로 내려받고, 이미 있으면 그대로 사용한다.
"$PYTHON" download_dataset.py \
  --data_root ./data/pokemon-generation-one-22k

for mode in one_nfe few_nfe; do
  if [[ "$mode" == one_nfe ]]; then
    checkpoint=./checkpoints/one_nfe.ckpt
  else
    checkpoint=./checkpoints/few_nfe.ckpt
  fi

  "$PYTHON" evaluate.py \
    --model_checkpoint "$checkpoint" \
    --evaluate_mode "$mode" \
    --data_root ./data/pokemon-generation-one-22k \
    --split_dir ./data/pokemon-generation-one-22k \
    --reference_dir ./results/reference \
    --output_dir ./results/evaluation \
    --batch_size 16 \
    --num_workers 4 \
    --device "$DEVICE"
done
```

평가기는 클래스별 20장, 총 3,020장을 생성한다. 기준 프로토콜은 batch16, bf16, seed=`42 + batch index`다. `kagglehub`는 FID 계산에는 관여하지 않지만, 고정 `dataset.py`가 이미지 디렉터리 부재 시 원본 `download_dataset.py`를 통해 자동 다운로드하므로 requirements에 포함한다. `tqdm`도 고정 `evaluate.py`의 진행률 표시 import에 필요하다. CUDA가 없는 환경에서는 CPU로 실행되며 시간이 더 걸릴 수 있다.

## 파일 구성

```text
.
├── checkpoints/
│   ├── one_nfe.ckpt
│   └── few_nfe.ckpt
├── configs/submission_model.json
├── data/pokemon-generation-one-22k/
│   ├── category_to_id.json
│   ├── train_split.txt
│   └── val_split.txt
├── distillation/          # sampler, DDIM/velocity helpers
├── teacher/               # inference UNet, diffusion schedule, CFG arithmetic
├── model.py               # 제공 base Model 및 두 NFE 제출 subclass
├── dataset.py             # 제공 데이터셋 코드
├── download_dataset.py    # 제공 데이터 다운로드 코드
├── evaluate.py            # 제공 공식 평가 인터페이스
├── requirements.txt       # 과제 제공 런타임 의존성
└── src/utils.py           # 제공 파라미터 계산 코드
```

`dataset.py`, `download_dataset.py`, `evaluate.py`, `src/utils.py`, split 파일은 과제 제공 버전 그대로다. `Model(nn.Module)` base class의 checkpoint loading과 parameter counting도 변경하지 않았다.

## 구현 참고문헌 및 데이터 출처

- Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), NeurIPS 2020 — diffusion objective와 U-Net 기반 모델 계열.
- Ho and Salimans, [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598), 2022 — class/null-condition guidance.
- Song, Meng, and Ermon, [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502), ICLR 2021 — deterministic diffusion sampling update.
- Salimans and Ho, [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512), ICLR 2022 — few-step student 계보.
- Yin et al., [One-step Diffusion with Distribution Matching Distillation](https://arxiv.org/abs/2311.18828), CVPR 2024 — NFE-1 DMD-inspired objective.
- Goodfellow et al., [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661), NeurIPS 2014 — NFE-4 adversarial fine-tuning objective.
- Miyato et al., [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957), ICLR 2018 — NFE-4 fine-tuning discriminator의 spectral normalization.
- [Pokemon Generation One - 22k](https://www.kaggle.com/datasets/bhawks/pokemon-generation-one-22k) — 데이터셋.

위 방법은 참고한 항목을 밝힌 것이며 논문 알고리즘 전체를 재현했다고 주장하지 않는다. 제공된 FastGen `model.py` base interface와 `dataset.py`/`download_dataset.py`/`evaluate.py`/`src/utils.py`를 사용한다. U-Net, sampling 및 학습 loss 조합은 제출 checkpoint에 맞춘 과제용 구현이며, sampling grid, anchor 및 학습 예산은 자체 실험 설정이다.
