# Piper GR00T Inference

GR00T 추론은 호스트에서 돌리고, ROS2 브리지는 도커 안에서 돌린다.

## 1. Host

`gr00t` uv 환경이 있는 호스트에서 실행:

```bash
uv run python scripts/deployment/gr00t_piper_inference_server.py --model-path gr00t/model/gr00t_rgb_run/checkpoint-200000 --host 0.0.0.0 --port 5555 --device cuda
```

## 2. Container

도커 안에서 브리지 실행:

```bash
python3 /root/piper_gr00t_ros2_bridge.py
```

## 3. Important

브리지 파일 `/root/piper_gr00t_ros2_bridge.py` 안의 `SERVER_HOST`는 호스트 IP로 맞춰야 한다.

- `127.0.0.1`로 두면 컨테이너 자기 자신을 가리켜서 호스트 inference server에 연결되지 않는다.

## 4. Current Topics

- 입력 이미지: `/ec_sensor_1/camera/color/image_raw`
- 입력 관절 상태: `/ec_robot_1/joint_states_single`
- 출력 관절 명령: `/ec_robot_1/joint_states`

## 5. Current Mapping

현재는 Piper 한 팔 기준 최소 매핑만 사용한다.

- `right_arm[:6]` -> `joint1..joint6`
- `right_hand[0]` -> `gripper`
