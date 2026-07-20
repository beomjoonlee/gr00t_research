# Piper GR00T Inference 실행 가이드 (2026-07-02)

7월 카메라 재배치 데이터(wrist/right/left)로 학습한 체크포인트 기준.
**EasyTrainer는 데이터 수집 때 사용한 것을 그대로 사용 — 아무 수정 없음.**

## 구조

```
[호스트]  gr00t_piper_inference_server.py  ← 체크포인트 로드, TCP :5555
    ↑ msgpack socket (127.0.0.1, 컨테이너가 host network라 localhost 통신)
[호스트]  piper_bridge (ROS2 노드, 레포 코드)   ← 2026-07-02 수정본
    ↕ ROS2 토픽 (host network 공유)
[Docker]  EasyTrainer 컨테이너  ← 로봇 드라이버(/ec_robot_1/*) + 카메라(/ec_sensor_*)
```

- 브릿지는 예전처럼 컨테이너 안이 아니라 **호스트에서 실행** (ROS2 Jazzy, 레포 `ros2_ws` 빌드본).
  `/opt/easytrainer/project` 안의 구버전 브릿지는 더 이상 사용하지 않음 (실행만 안 하면 무해).

## 실행 순서

### 터미널 1 — 추론 서버 (호스트)

```bash
cd ~/Airlab/gr00t_research
uv run python scripts/deployment/gr00t_piper_inference_server.py \
    --model-path <체크포인트 경로> \
    --host 0.0.0.0 --port 5555 --device cuda
```

- `Policy loaded. Listening on 0.0.0.0:5555` 로그가 뜨면 준비 완료.
- 서버(aigpu1218)에서 체크포인트를 가져올 경우:
  `scp -r kowook@163.239.98.45:/data1/kowook_gr00t/gr00t_research/outputs/<run>/checkpoint-XXXX ./`

### 터미널 2 — EasyTrainer (수집 때 쓰던 그대로)

평소 데이터 수집할 때처럼 EasyTrainer를 켜고 로봇 팔 + 카메라 3대를 연결한다.
수정할 것 없음. 확인만:

```bash
# 호스트에서 (source /opt/ros/jazzy/setup.bash 후)
ros2 topic list | grep -E "ec_sensor|ec_robot"
```

기대 토픽 (수집 때와 동일한 EasyTrainer라면):

| 역할 | 센서 | 토픽 |
|------|------|------|
| wrist | sensor_7 (D405) | `/ec_sensor_7/camera/color/image_rect_raw` |
| right | sensor_8 (D435) | `/ec_sensor_8/camera/color/image_raw` |
| left | sensor_11 (D435) | `/ec_sensor_11/camera/color/image_raw` |
| 관절 상태 | robot_1 | `/ec_robot_1/joint_states_single` |
| 관절 명령 | robot_1 | `/ec_robot_1/joint_states` |

### 터미널 3 — 브릿지 (호스트)

```bash
source /opt/ros/jazzy/setup.bash
source ~/Airlab/gr00t_research/ros2_ws/install/setup.bash

ros2 run gr00t_piper_bridge piper_bridge --ros-args \
    -p prompt_text:="Pick up the red object and place it on the white plate"
```

- 토픽이 위 기대값과 같으면 prompt만 넘기면 됨. 다르면 `-p topic_image_wrist:=...` 식으로 덮어쓰기.
- 시작 로그로 `Connected to GR00T server 127.0.0.1:5555` 와
  `Camera topics: wrist=..., right=..., left=...` 확인.
- `Latest observation incomplete, waiting (...)` 경고가 계속 나오면 → 빠진 키(joint/wrist/right/left)의 토픽이 안 들어오는 것. 토픽명 확인.

## prompt_text (학습 문구와 정확히 일치해야 함)

| 태스크 | prompt_text |
|--------|-------------|
| blue | `Pick up the blue object from the table and place it on the white plate` |
| red | `Pick up the red object and place it on the white plate` |
| green | `Pick up the green object and place it on the white plate` |

blue만 "from the table"이 들어가는 것에 주의.

## 주요 브릿지 파라미터 (기본값)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `server_host` / `server_port` | `127.0.0.1` / `5555` | 추론 서버 주소 |
| `exec_hz` | `10.0` | 액션 실행 주기 (수집 10Hz와 동일) |
| `action_chunk_len` / `exec_steps` | `16` / `8` | 청크 16 중 앞 8스텝만 실행 후 재추론 |
| `image_w` x `image_h` | `320` x `240` | 학습 데이터와 동일한 squish resize |
| `rotate_{wrist,right,left}_180` | `False` | 수집 시 카메라 rotation을 썼다면 동일하게 설정 |

## 트러블슈팅

- **서버 KeyError 'right' 등**: 구버전 브릿지가 실행된 것. 반드시 레포 빌드본(호스트) 사용.
- **토픽은 있는데 이미지가 안 들어옴**: 브릿지는 raw `Image` 토픽 구독. EasyTrainer 백엔드는 compressed를 쓰지만 realsense 노드는 raw도 함께 발행하므로 정상이라면 둘 다 존재.
- **호스트↔컨테이너 토픽이 안 보임**: ROS_DOMAIN_ID 불일치. 컨테이너 기동 방식에 따라 0 또는 30 → 브릿지 터미널에서 `export ROS_DOMAIN_ID=<값>` 후 재실행.
- **그리퍼 파지 실패**: 이전 세션에서 확인된 이슈 (그리퍼 범위 0.002~0.0862로 좁음). 액션 청크는 `/gr00t/action_chunk` (Float32MultiArray, 16x7)로 발행되므로 `ros2 topic echo`로 모델 예측값 직접 확인 가능.
- **브릿지 코드 수정 후**: `cd ~/Airlab/gr00t_research/ros2_ws && colcon build --packages-select gr00t_piper_bridge` 재빌드 필요.

## 2026-07-02 브릿지 변경사항

`ros2_ws/src/gr00t_piper_bridge/gr00t_piper_bridge/piper_bridge_node.py` (레포 내 1개 파일만 수정):

1. 이미지 키 `top/side/wrist` → `wrist/right/left` (새 piper_config / 새 체크포인트와 일치)
2. 기본 토픽 sensor_2/3/4 → sensor_7/8/11
3. 구 wrist 카메라 전용 crop(`[:, 104:744]`) 제거 (새 학습 데이터는 crop 없음)
4. 학습 기록과 동일한 320x240 cv2 squish resize 추가
5. `exec_hz` 기본값 30 → 10 (수집 주파수와 일치)
6. 기본 prompt를 학습 문구와 정확히 일치하도록 수정

검증 완료: 새 학습 데이터 색상 정상 (BGR 스왑 없음 — red 태스크에서 빨간 블록, blue 태스크에서 파란 블록 집는 것 프레임으로 확인). 브릿지의 RGB 변환 유지가 올바름.
