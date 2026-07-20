# GR00T Fine-tuning Session Summary (2026-04-02 ~ 04-03)

## 1. 이전 세션에서 이어받은 상태

- Piper 로봇 + GR00T N1.6 파인튜닝 프로젝트
- 이전 파인튜닝은 `piper_config.py` 없이 진행 → `action_configs=None` → arm이 ABSOLUTE로 학습됨 → 추론 시 의미없는 움직임
- 이미지 사이즈 불일치 (256x256 squish vs 341x256 학습), 카메라 매핑 오류, EmbodimentTag 불일치 등 다수 문제 존재

## 2. 핵심 문제 진단

### 이전 파인튜닝 실패 원인
- `piper_config.py` (modality config) 미사용 → `action_configs=None` → arm이 ABSOLUTE로 학습
- ABSOLUTE 학습은 시작 자세가 조금만 달라도 오차 회복 불가 → 의미없는 반복 움직임
- 공식 SO100 예제처럼 arm=RELATIVE, gripper=ABSOLUTE로 해야 함

### HDF5 → MP4 변환 시 BGR/RGB 채널 스왑 문제 (critical)
- EasyTrainer의 HDF5는 이미지를 **BGR** 형식으로 저장함
- 초기 변환 스크립트가 RGB로 가정하고 `cv2.cvtColor(img, COLOR_RGB2BGR)` 적용 → 채널 이중 스왑 → 빨간색이 파란색으로 보임
- **"pick up the red object"인데 모델은 파란색 물체를 보고 학습** → 학습 품질 심각하게 저하
- 수정: `images_to_mp4()`에서 BGR 변환 제거, `writer.write(images[t])` 직접 기록
- **실 데이터를 재변환하고 서버에 다시 업로드 후 학습을 처음부터 다시 해야 함**

### Inference 코드 문제들
- EmbodimentTag 불일치 (GR1 vs NEW_EMBODIMENT)
- State key 불일치 (control 통째 vs arm/gripper 분리)
- 이미지 256x256 강제 리사이즈 (학습은 341x256)
- 카메라 매핑 오류 (top/side 뒤바뀜)

## 3. 새로 만든 파일들

### 생성
| 파일 | 용도 |
|------|------|
| `examples/Piper/piper_config.py` | Piper modality config (arm RELATIVE + gripper ABSOLUTE, 카메라 3대) |
| `scripts/piper_convert_hdf5_to_lerobot.py` | HDF5 → LeRobot v2 변환 스크립트 |

### 수정
| 파일 | 수정 내용 |
|------|----------|
| `scripts/deployment/gr00t_piper_inference_server.py` | piper_config import, state arm/gripper 분리, action 합쳐서 control로 반환, 256x256 리사이즈 제거 |
| `ros2_ws/.../piper_bridge_node.py` | 카메라 키 side/top/wrist로 변경, top/side 매핑 교정, wrist crop 유지 |
| `gr00t/model/modules/eagle_backbone.py` | `local_files_only` 인자 제거 (transformers 호환성) |
| `gr00t/data/dataset/factory.py` | `barrier()` 조건부 처리 (단일 GPU 지원) |
| `gr00t/configs/data/data_config.py` | video_backend: torchcodec → opencv (서버 전용) |
| `gr00t/configs/finetune_config.py` | `extra_dataset_paths`, `extra_mix_ratios` 필드 추가 |
| `gr00t/experiment/launch_finetune.py` | 다중 데이터셋 지원 (datasets_cfg 리스트) |

## 4. Piper Modality Config (`examples/Piper/piper_config.py`)

```python
piper_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["side", "top", "wrist"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=["arm", "gripper"]),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),  # 16-step horizon
        modality_keys=["arm", "gripper"],
        action_configs=[
            ActionConfig(rep=RELATIVE, type=NON_EEF, format=DEFAULT),  # arm 6D
            ActionConfig(rep=ABSOLUTE, type=NON_EEF, format=DEFAULT),  # gripper 1D
        ],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.action.task_description"]),
}
```

## 5. 데이터셋

### 실 데이터 (30 에피소드)
- 경로 (로컬): `/home/wook/Airlab/gr00t_research/data/piper_pick_place/`
- 경로 (서버): `/data1/kowook_gr00t/gr00t_research/data/`
- 10Hz, 201 프레임/에피소드, 총 6,030 프레임
- HDF5 → LeRobot v2 변환 완료 (cv2로 MP4 생성)
- 3개 태스크: red/green/blue object pick and place

### 시뮬 데이터 (457 에피소드)
- 경로 (서버): `/data1/kowook_gr00t/gr00t_research/datasets/sim_red|sim_blue|sim_green/`
- sim_red: 223 ep, sim_blue: 189 ep, sim_green: 45 ep
- 10Hz, 241 프레임/에피소드, 총 110,137 프레임
- Isaac Lab에서 수집, AV1 → H.264 변환 완료

### 데이터셋 공통 modality.json
```json
{
    "state": {"arm": {"start": 0, "end": 6}, "gripper": {"start": 6, "end": 7}},
    "action": {"arm": {"start": 0, "end": 6}, "gripper": {"start": 6, "end": 7}},
    "video": {
        "side": {"original_key": "observation.images.cam_left_wrist"},
        "top": {"original_key": "observation.images.cam_high"},
        "wrist": {"original_key": "observation.images.cam_right_wrist"}
    },
    "annotation": {"human.action.task_description": {"original_key": "task_index"}}
}
```
(실 데이터는 original_key가 `observation.images.side/top/wrist`로 다름)

## 6. 카메라 매핑

### 물리 카메라 → ROS 토픽 → 학습 키
| 카메라 | ROS 토픽 | HDF5 (실물) | 시뮬 | 학습 키 |
|--------|----------|------------|------|---------|
| side view | `/ec_sensor_2` | sensor_2 | cam_left_wrist | side |
| top view | `/ec_sensor_3` | sensor_3 | cam_high | top |
| wrist | `/ec_sensor_4` | sensor_4 | cam_right_wrist | wrist |

- `ec_sensor_1`은 존재하지 않음

### Bridge 매핑 (piper_bridge_node.py)
- image_1 (`/ec_sensor_3`) → `"top"` (수정 완료: 이전에 topic이 ec_sensor_2→side로 변경됨, 현재는 ec_sensor_3→top)
- image_2 (`/ec_sensor_2`) → `"side"`
- image_3 (`/ec_sensor_4`) → `"wrist"` (crop `[:, 104:744, :]` 적용 — 학습 데이터와 동일)

**주의**: bridge의 topic_image 기본값이 세션 중 변경되었을 수 있음. 실행 전 확인 필요.

## 7. 학습 설정

### 현재 진행 중인 학습 (piper_v3_mixed)
```bash
export NUM_GPUS=1 && CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path /data1/kowook_gr00t/gr00t_research/data \
    --extra-dataset-paths \
        /data1/kowook_gr00t/gr00t_research/datasets/sim_red \
        /data1/kowook_gr00t/gr00t_research/datasets/sim_blue \
        /data1/kowook_gr00t/gr00t_research/datasets/sim_green \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/Piper/piper_config.py \
    --num-gpus 1 \
    --output-dir /data1/kowook_gr00t/gr00t_research/outputs/piper_v3_mixed \
    --max-steps 10000 --save-steps 2000 --save-total-limit 5 \
    --global-batch-size 64 --dataloader-num-workers 0 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
```

- mix_ratio 전부 1.0 → 실 데이터 25%, 시뮬 데이터 75% (배치 내)
- 체크포인트: 2000, 4000, 6000, 8000, 10000
- 출력: `/data1/kowook_gr00t/gr00t_research/outputs/piper_v3_mixed/`
- 속도: ~6.77s/step → 전체 약 18.8시간

### 이전 학습 (piper_v2, 실 데이터만)
- 출력: `/data1/kowook_gr00t/gr00t_research/outputs/piper_v2/`
- 실 데이터 30개만 사용, batch_size=32, ~1.94s/step

## 8. Inference 설정

### Inference Server 실행
```bash
uv run python scripts/deployment/gr00t_piper_inference_server.py \
    --model-path outputs/checkpoint-XXXX \
    --host 0.0.0.0 --port 5555 --device cuda
```

- `examples/Piper/piper_config.py`를 자동 import
- State: 7D → arm(6D) + gripper(1D) 분리
- Action: arm + gripper → control(7D) 합쳐서 bridge에 전달
- 이미지: 원본 그대로 전달 (모델 processor가 리사이즈)

### Bridge (piper_bridge_node.py)
- `action_chunk_len=16` (학습 horizon과 동일)
- wrist 이미지: `[:, 104:744, :]` crop 적용 (학습 데이터와 동일)
- 이미지: 640x480 원본 전송 (리사이즈 하지 않음)

## 9. 서버 환경 이슈 및 해결

| 이슈 | 해결 |
|------|------|
| `local_files_only` 에러 | eagle_backbone.py에서 해당 키 제거 |
| `torch.distributed.barrier()` | factory.py에서 조건부 처리 |
| torchcodec 미설치 | video_backend를 opencv로 변경 |
| shared memory 부족 | dataloader-num-workers=0으로 변경 |
| AV1 코덱 디코딩 불가 | ffmpeg으로 H.264 재인코딩 |

## 10. 미해결 / 다음 세션에서 확인할 사항

1. **[긴급] 실 데이터 BGR 채널 문제로 재변환 및 재학습 필요**
   - HDF5 이미지가 BGR인데 RGB로 가정하고 변환 → 빨간색이 파란색으로 보임
   - `scripts/piper_convert_hdf5_to_lerobot.py`는 수정 완료 (BGR 직접 기록)
   - 실 데이터 재변환 → 서버 재업로드 → 학습 재시작 필요
   - **현재 진행 중인 piper_v3_mixed 학습은 잘못된 색상으로 학습 중이므로 중단/폐기해야 함**
2. **추론 테스트** — 재학습 완료 후 실제 로봇 테스트
3. **그리퍼 파지 문제** — 이전 모델에서 그리퍼가 물체를 못 잡는 문제 있었음
   - EMA smoothing (alpha=0.3)이 그리퍼 닫힘을 느리게 함 → alpha 올리거나 그리퍼에 EMA 미적용
   - 그리퍼 범위가 매우 좁음 (0.002~0.0862)
   - Open loop evaluation으로 모델 예측 정확도 확인 필요
4. **bridge의 topic_image 기본값 확인** — 세션 중 여러 번 변경됨, 실행 전 최종 상태 확인 필요
5. **`/dev/shm` 크기 확인** — 충분하면 dataloader-num-workers 늘려서 학습 속도 개선
6. **mix_ratio 조정 실험** — 실 데이터 비율을 높이면 실물 성능 향상 가능
