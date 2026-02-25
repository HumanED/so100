import os
from typing import Optional, TypedDict

import gymnasium
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium.utils import EzPickle

from so_arm_rl.envs.utils import mujoco_utils

DEFAULT_CAMERA_CONFIG = {
    "distance": 0.5,
    "azimuth": 140.0,
    "elevation": -25.0,
    "lookat": np.array([0.4, -0.6, 0.5]),
}
# Based on keyframe values https://github.com/google-deepmind/mujoco_menagerie/blob/main/trs_so_arm100/so_arm100.xml
INITIAL_ARM_POSITION = {"robot_Rotation": 0, "robot_Pitch": -1.57, "robot_Elbow": 1.57, "robot_Wrist_Pitch": 1.57,
                        "robot_Wrist_Roll": -1.57, "robot_Jaw": 0.0}

class InfoDict(TypedDict):
    is_success: int
    total_timesteps: int
    debug_jaw_top_to_target: float | int
    debug_jaw_bottom_to_target: float | int
    debug_jaw_center_to_object: float | int  # d1 'distance from end effector to cylinder'
    reset_flag: bool
    debug_object_to_target: float | int  # d3 'distance from cylinder to goal'
    debug_jaw_total_dist_to_object: float | int  # d2 'sum of distances of each finger'
    debug_regularisation: float | int
    debug_grasp_reward: float | int
    debug_jaw_angle: float | int
    debug_rew_standard: float | int


class SoFetchEnv(gymnasium.Env, EzPickle):
    """
    Gymnasium environment of the SO-100 arm https://github.com/huggingface/lerobot/blob/main/examples/10_use_so100.md
    """
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ],
        "render_fps": 50,
    }

    def __init__(
            self,
            render_mode: Optional[str] = None,
    ):
        """
        Initialize environment
        Args:
            render_mode (optional string): type of rendering mode, "human" for window rendering and "rgb_array" for offscreen. Defaults to None.
        """
        EzPickle.__init__(self, render_mode)

        # SETTINGS
        # All settings should not be modified at runtime
        # N_SUBSTEPS (integer)              number of MuJoCo simulation total_timesteps per call to step()
        # STEP_BETWEEN_GOALS (integer)      Robot has STEP_BETWEEN_GOALS calls to step() to reach the current goal or the environment is truncated
        # SCREEN_WIDTH                      SCREEN_WIDTH of each rendered frame. Defaults to DEFAULT_SIZE.
        # SCREEN_HEIGHT                     SCREEN_HEIGHT of each rendered frame . Defaults to DEFAULT_SIZE.
        # ROTATION_THRESHOLD (float)        If angular rotation from current orientation to target orientation is below this threshold, goal is considered achieved. Unit is radians
        # RELATIVE_CONTROL (bool)           Set True to actuate hand using relative joint positions (following OpenAI) or False for absolute joint positions (like Pybullet model)
        # RANDOMIZE_INITIAL_ROTATION (bool) Set True to set cube orientation to a random orientation at start of episode.
        # FIXED_GOAL (bool)                 True if getting goal position of "target" body in scene.xml
        # MAX_GOALS                         Maximum number of goals to reach before truncating the environment
        # N_ACTIONS (integer)               size of the action space.
        # N_OBS (integer)                   size of observation space
        # FULLPATH                          Path to Mujoco XML file holding robot hand, floor and cube of the simulation environment

        self.info:InfoDict = None
        self.MAX_TIMESTEPS = 100  # 8 seconds real time. Do NOT rename this attribute.
        self.RELATIVE_CONTROL = False
        self.N_SUBSTEPS = 20
        self.EMA = None
        self.FIXED_GOAL = True
        self.GOAL_MAX = [0.5, 0.5, 0.5]
        self.GOAL_MIN = [0.1, 0.1, 0.1]
        self.FIXED_GRASP_REWARD = 20
        self.grasp_reward = self.FIXED_GRASP_REWARD
        self.FIXED_TARGET_REACHED_REWARD = 30
        self.target_reached_reward = self.FIXED_TARGET_REACHED_REWARD
        self.grasped = False
        self.prev_action = None

        N_ACTIONS = 6
        N_OBS = 28
        self.N_DISCRETE = 64
        self.action_space = gymnasium.spaces.MultiDiscrete(nvec=[self.N_DISCRETE] * N_ACTIONS)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(N_OBS,), dtype=np.float32)

        self.FULLPATH = os.path.join(os.path.dirname(__file__), "../resources", "fetch", "scene.xml")
        self.SCREEN_WIDTH = 1200
        self.SCREEN_HEIGHT = 800
        # END SETTINGS

        self._load_mujoco_robot()
        self.goal = np.zeros(0)
        self.total_timesteps = 0
        # self.info: InfoDict = {
        #     "is_success": 0,
        #     "total_timesteps": 0,
        # }
        self.render_mode = render_mode
        self.mujoco_renderer = MujocoRenderer(
            self.model,
            self.data,
            DEFAULT_CAMERA_CONFIG,
        )

    def _load_mujoco_robot(self):
        """
        Loads XML file containing all information about the cube and hand. Runs only once when gymnasium.make() is called
        """
        self.model = mujoco.MjModel.from_xml_path(self.FULLPATH)
        self.data = mujoco.MjData(self.model)
        self._model_names = mujoco_utils.MujocoModelNames(self.model)
        self.model.vis.global_.offwidth = self.SCREEN_WIDTH
        self.model.vis.global_.offheight = self.SCREEN_HEIGHT

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        """
        Reset the environment, counters, self.info, position of cube and hand, and goal
        """
        # Reset environment state
        super().reset(seed=seed)
        self.total_timesteps = 0

        self._reset_sim()

        # Compute initial goal
        self.goal = self._compute_goal()

        # Reset the once-per episode rewards
        self.info:InfoDict = {
            "is_success": 0,
            "total_timesteps": 0,
            "debug_jaw_top_to_target": 0,
            "debug_jaw_bottom_to_target": 0,
            "debug_jaw_center_to_object": 0, # d1 'distance from end effector to cylinder' in paper
            "reset_flag": True,
            "debug_object_to_target": 0, # d3 'distance from cylinder to goal' in paper
            "debug_jaw_total_dist_to_object": 0, # d2 'sum of distances of each finger' to the cylinder
            "debug_regularisation": 0,
            "debug_grasp_reward": 0,
            "debug_jaw_angle": 0,
            "debug_rew_standard": 0,
        }

        # Return obs and info
        obs, _ = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, self.info

    def _compute_goal(self) -> np.ndarray:
        """Returns goal position [x,y,z]"""
        if self.FIXED_GOAL:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
            new_goal = self.data.xpos[body_id].copy()
        else:
            # Generate a random goal (x,y,z)
            new_goal = self.np_random.uniform(self.GOAL_MIN, self.GOAL_MAX)
        return new_goal

    def _reset_sim(self):
        """Resets simulation and puts cube in fixed initial position depending on settings. Resets arm to fixed position"""
        # TODO: Add option for random initial position and orientation of cube within reach of the arm
        mujoco.mj_resetData(self.model, self.data)

        for name, val in INITIAL_ARM_POSITION.items():
            mujoco_utils.set_joint_qpos(self.model, self.data, name, val)
            actuator_name = name.replace("robot_", "")  # joint "robot_Pitch" → actuator "Pitch"
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            self.data.ctrl[actuator_id] = val
        mujoco.mj_forward(self.model, self.data)

        # Run the simulation for a bunch of timesteps to let everything settle in.
        for _ in range(10):
            try:
                mujoco.mj_step(self.model, self.data, nstep=self.N_SUBSTEPS)
            except Exception:
                return False

    def _rescale_actions(self, action: np.ndarray) -> np.ndarray:
        """
        Converts discrete actions to continuous actions in range [-1, 1] which mujoco expects.
        """
        rescaledAction = -1 + (action * 2) / (self.N_DISCRETE - 1)
        if self.EMA != None:
            rescaledAction = self.EMA.update(rescaledAction)
        return rescaledAction


    def step(self, action: np.ndarray):
        """Run one timestep of the environment's dynamics using the agent actions.

        Args:
            action (np.ndarray): Control action to be applied to the agent and update the simulation. Should be of shape :attr:`action_space`.
            Should be [Rotation (base rotation), Pitch (base up-down), Elbow (up-down), Wrist_Pitch (up-down), Wrist_Rotation (rotation), Jaw (up down)]
            by observing the motors in order from base of the robot to the jaw

        Returns:
            observation (np.ndarray): Next observation due to the agent actions
            reward (float): The reward as a result of taking the action.
            terminated (boolean): Whether the agent reaches the terminal state (cube is dropped)
            truncated (boolean): Whether the agent exceeds maximum time for an episode
            info (dictionary): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
        """
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")
        self.info["reset_flag"] = False

        self.total_timesteps += 1
        self.info["total_timesteps"] = self.total_timesteps

        # Rescale the angle between -1 and 1 for _apply_action(). See action space of https://robotics.farama.org/envs/shadow_dexterous_hand/manipulate_block/
        # See second min-max normalization formula https://en.wikipedia.org/wiki/Feature_scaling

        rescaled_actions = self._rescale_actions(action)
        self.prev_action = rescaled_actions
        self._apply_action(rescaled_actions)

        obs, extra_obs = self._get_obs()

        # Compute reward
        reward = self._compute_reward(obs, extra_obs)

        terminated = truncated = False
        if (self.total_timesteps >= self.MAX_TIMESTEPS):
            truncated = True

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, self.info

    def _compute_reward(self, obs, extra_obs):
        # TODO: Try a 2 staged reward where reward for getting jaw to next to the cube then lessen the cube reward and add reward for cube to target
        # TODO: Try using previous distance - current distance maybe later
        # TODO: Change logging to the composition of 0.75 etc of each
        """Reward function"""
        # d1 distance between jaw center and object
        object_jaw_diff = obs[22:25]
        jaw_center_to_object = np.linalg.norm(object_jaw_diff)
        d1 = jaw_center_to_object
        self.info["debug_jaw_center_to_object"] = d1

        # distance from center of cube to edges = sqrt(1**2 + 1**2)
        cube_diagonal_width = 0.0213
        # d2 distance between jaw fingers and object
        object_pos = obs[12:15]
        jaw_top_object_dist =  np.linalg.norm(object_pos - extra_obs[0:3])
        jaw_bottom_object_dist =  np.linalg.norm(object_pos - extra_obs[3:6])
        d2 = jaw_top_object_dist + jaw_bottom_object_dist - (2 * cube_diagonal_width)

        self.info["debug_jaw_top_to_target"] = jaw_top_object_dist
        self.info["debug_jaw_bottom_to_target"] = jaw_bottom_object_dist
        self.info["debug_jaw_total_dist_to_object"] = d2

        # d3 distance between object and target
        object_target_diff = np.linalg.norm(obs[25:28])
        self.info["debug_object_to_target"] = object_target_diff
        d3 = object_target_diff

        weight = 50
        jaw_angle = obs[5]
        min_angle = 0.220 # angle below which no grasp reward is given. Encourages jaw to be at least min_angle open
        self.info["debug_jaw_angle"] = jaw_angle
        object_jaw_proximity_threshold = 0.035 # jaw center must be at least this close to cube center for grasp reward.
        grasp_reward = weight * max(0, jaw_angle - min_angle) * max(0, object_jaw_proximity_threshold - d1)
        self.info["debug_grasp_reward"] = grasp_reward

        reward = 1 / (1 + (
            d1 * 1
            + d2 * 1
            + d3 * 1
        ))
        self.info["debug_rew_standard"] = reward
        reward += grasp_reward

        if self.prev_action is not None:
            regularisation = np.linalg.norm(self.prev_action) * 10**-3
            self.info["debug_regularisation"] = regularisation
            reward -= regularisation

        return reward

    def _apply_action(self, action: np.ndarray):
        """
        Sends AI action (numpy array of numbers between -1 and 1) to Mujoco simulation and steps simulation to execute the action.
        Action can be relative position or absolute position based on settings
        """
        ctrlrange = self.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.0

        if self.RELATIVE_CONTROL:
            actuation_center = np.zeros_like(action)
            for i in range(self.data.ctrl.shape[0]):
                # In xml, joints start with robot_ but actuators don't
                actuation_center[i] = self.data.get_joint_qpos(
                    self.model.actuator_names[i].lstrip("robot_")
                )
        else:
            actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.0
        self.data.ctrl[:] = actuation_center + action * actuation_range
        self.data.ctrl[:] = np.clip(self.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])
        mujoco.mj_step(self.model, self.data, nstep=self.N_SUBSTEPS)

    def _get_obs(self):
        """
        Removed self.goal from observation 11/2/2026 as information already encoded in object_target_diff
        observation = [robot_qpos, robot_qvel, object_qpos, jaw_pos, object_jaw_diff, object_target_diff]

        Indexing        [0 - 5 (robot_qpos), 6 - 11 (robot_qvel), 12 - 18 (object_qpos), 19 - 21 (jaw_pos), 22 - 24 (object_jaw_diff), 25 - 27 (object_target_diff)]
        extra_observation = [0 - 2 (jaw_top_pos), 3 - 5 (jaw_bottom_pos) ]
        robot_qpos:          6 numbers. Joint angles (radians) of 6 motors. 'robot_Rotation', 'robot_Pitch', 'robot_Elbow', 'robot_Wrist_Pitch', 'robot_Wrist_Roll', 'robot_Jaw' (items starting with robot_ in self._model_names.joint_names)
        robot_qvel:          6 numbers. Joint velocity (radians / sec) of 6 motors
        object_qpos:         7 numbers. Position (x,y,z) then orientation (w, x, y, z) of the cube
        jaw_pos:             3 numbers. Position (x,y,z) of the jaw
        object_jaw_diff:     3 numbers. Difference between position of the object and jaw (object - jaw)
        object_target_diff:  3 numbers. Difference between position of the object and target location (object - target)

        """
        robot_qpos, robot_qvel = mujoco_utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )

        # cube cartesian position (x,y,z) and quaternion orientation (x,y,z,w)
        object_qpos = mujoco_utils.get_joint_qpos(self.model, self.data, "object:joint")

        # Position of the jaw (x,y,z)
        jaw_pos = mujoco_utils.get_site_xpos(self.model, self.data, "jaw_site")

        # Difference between Jaw and Cube position (dx, dy, dz)
        object_jaw_diff = object_qpos[:3] - jaw_pos

        # Difference between Cube and target position (dx, dy, dz)
        object_target_diff = object_qpos[:3] - self.goal

        observation = np.concatenate(
            [robot_qpos, robot_qvel, object_qpos, jaw_pos, object_jaw_diff, object_target_diff])
        assert observation.shape == self.observation_space.shape, f"Expected obs shape {self.observation_space.shape} Actual shape {observation.shape}"

        jaw_top_pos = mujoco_utils.get_site_xpos(self.model, self.data, "jaw_top_site")
        jaw_bottom_pos = mujoco_utils.get_site_xpos(self.model, self.data, "jaw_bottom_site")
        extra_observation = np.concatenate([jaw_top_pos, jaw_bottom_pos])

        return observation, extra_observation

    # --- other utility methods

    def render(self):
        """Render a frame of the Mujoco simulation.

        Returns:
            rgb image (np.ndarray): if render_mode is "rgb_array", return a 3D image array.
        """
        # Assign current state to target object but offset a bit so that the actual object
        # is not obscured.

        render_target = np.concatenate([self.goal, np.array([1, 0, 0, 0])])
        assert render_target.shape == (7,), f"Actual goal shape {render_target.shape}"

        if "object_hidden" in self._model_names.geom_names:
            hidden_id = self._model_names.geom_name2id["object_hidden"]
            self.model.geom_rgba[hidden_id, 3] = 1.0
        mujoco.mj_forward(self.model, self.data)

        return self.mujoco_renderer.render(self.render_mode)

    def close(self):
        """
        Terminates any existing WindowViewer instances in the Gymnaisum MujocoRenderer.
        Call this method to prevent errors when rendering.
        """
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()


