"""RoboCasa Panda+Omron modality configuration for GR00T N1.6 fine-tuning.

This config is aligned EXACTLY with the official `robocasa_panda_omron`
definition baked into the pretrained GR00T-N1.6 model (embodiment projector
index 13). Matching the pretrained layout means:
  - the projector starts from a correctly-aligned init (no need to relearn a
    key-order permutation during fine-tuning),
  - the official RoboCasa evaluation pipeline works without modification.

Key-order / type decisions (do NOT reorder):
  - video : [side, side, wrist]   -> wrist (eye-in-hand) MUST be last.
  - state : [ee_pos_rel, ee_rot_rel, gripper, base_pos, base_rot]
  - action: [ee_pos, ee_rot, gripper, base_motion, control_mode]

Action values in the dataset are already stored as deltas (EE-space delta
commands produced by RoboCasa's OSC controller, plus base velocity / discrete
signals), confirmed by norm stats (EE action q01/q99 ~= [-1, 1], mean ~= 0).
They must be used as-is, so all `rep` fields are ABSOLUTE -- setting RELATIVE
would cause a double subtraction (or an EEF 9-d assert failure) during
training preprocessing. All `type` fields are NON_EEF to match the official
config; with rep=ABSOLUTE the type field is unused anyway.

Video / language key NAMES follow this dataset's `meta/modality.json`
(`robot0_*`, `annotation.human.task_description`); only the ORDER is matched
to the official definition.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

robocasa_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            # [side, side, wrist] -- wrist (eye-in-hand) MUST be last.
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "end_effector_position_relative",
            "end_effector_rotation_relative",
            "gripper_qpos",
            "base_position",
            "base_rotation",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),  # 16-step action horizon (GR00T N1.6)
        modality_keys=[
            "end_effector_position",
            "end_effector_rotation",
            "gripper_close",
            "base_motion",
            "control_mode",
        ],
        action_configs=[
            # All ABSOLUTE (data is already delta) + NON_EEF (matches official).
            # end_effector_position (3) - already EE delta
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # end_effector_rotation (3) - already EE delta rotation
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # gripper_close (1) - open/close signal
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # base_motion (4) - already velocity/delta
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # control_mode (1) - discrete switch signal
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(robocasa_config, embodiment_tag=EmbodimentTag.ROBOCASA_PANDA_OMRON)
