import os
from typing import Optional
import mujoco
import numpy as np
import gymnasium
from gymnasium import spaces
from gymnasium.utils import EzPickle
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from ..utils import rotations
from ..utils import mujoco_utils
from ..utils import ema_util


DEFAULT_CAMERA_CONFIG = {
    "distance": 0.5,
    "azimuth": 140.0,
    "elevation": -25.0,
    "lookat": np.array([0.4, -0.6, 0.5]),
}

INITIAL_ARM_POSITION = {"robot_Rotation":0,"robot_Pitch":-3.10,"robot_Elbow":3.10,"robot_Wrist_Pitch":0.0,"robot_Wrist_Roll":0.0,"robot_Jaw":0.0}

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
        # FIXED_GOAL (bool or list)         Fixed Euler goal. Set to None if we are not using a random goal instead of a fixed goal. Set to list e.g. [1,2,3] for goal of x=1, y=2, z=3 radians.
        # MAX_GOALS                         Maximum number of goals to reach before truncating the environment
        # N_ACTIONS (integer)               size of the action space.
        # N_OBS (integer)                   size of observation space
        # FULLPATH                          Path to Mujoco XML file holding robot hand, floor and cube of the simulation environment

        self.MAX_TIMESTEPS = 100 # 8 seconds real time
        self.RELATIVE_CONTROL = False
        self.N_SUBSTEPS = 20
        self.EMA = None
        self.FIXED_GOAL = [-0.3, -0.2, 0.025]
        self.GOAL_MAX = [0.5, 0.5, 0.5]
        self.GOAL_MIN = [0.1, 0.1, 0.1]
        self.initial_cube_position = np.array([0.3, -0.3, 0.025])
        self.grasp_reward = 20
        self.target_reached_reward = 30


        N_ACTIONS = 6
        N_OBS = 31
        self.action_space = gymnasium.spaces.MultiDiscrete(nvec=[11] * N_ACTIONS)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(N_OBS,), dtype=np.float32)

        self.FULLPATH = os.path.join(os.path.dirname(__file__), "../resources", "fetch", "scene.xml")
        self.SCREEN_WIDTH = 1200
        self.SCREEN_HEIGHT = 800
        # END SETTINGS

        self._load_mujoco_robot()
        self.goal = np.zeros(0)
        self.total_timesteps = 0
        self.info = {
            "is_success": 0,
            "total_timesteps": 0,
            "dt": 0
        }
        self.render_mode = render_mode
        self.mujoco_renderer = MujocoRenderer(
            self.model,
            self.data,
            DEFAULT_CAMERA_CONFIG,
        )

    def _load_mujoco_robot(self):
        # TODO: Done
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
        # TODO: Done
        """
        Reset the environment, counters, self.info, position of cube and hand, and goal
        """
        # Reset environment state
        super().reset(seed=seed)
        self.total_timesteps = 0
        # Time between each frame in rendering
        dt = self.model.opt.timestep * self.N_SUBSTEPS
        self._reset_sim()

        # Compute initial goal
        self.goal = self._compute_goal()

        self.info = {
            "success": 0,
            "dt": dt,
            "total_timesteps": 0,
            "grasp":0,
            "rew_jaw_center_to_object":0,
            "rew_object_to_target":0,
            "rew_jaw_center_to_object_prop": 0,
            "rew_object_to_target_prop": 0,
            "reset_flag": True,
            "rew_other": 0
        }

        # Return obs and info
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, self.info

    def _compute_goal(self) -> np.ndarray:
        # TODO: Done
        """Returns goal position [x,y,z]"""
        if self.FIXED_GOAL:
            new_goal = np.array(self.FIXED_GOAL.copy())
        else:
            # Generate a random goal (x,y,z)
            new_goal = self.np_random.uniform(self.GOAL_MIN, self.GOAL_MAX)
        return new_goal

    def _reset_sim(self):
        """Resets simulation and puts cube in fixed initial position depending on settings. Resets arm to fixed position"""
        # TODO: Add option for random initial position and orientation of cube within reach of the arm
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # Set initial position and orientation (w,x,y,z) of the cube
        initial_cube_quat = np.array([1,0,0,0])
        initial_cube_qpos = np.concatenate([self.initial_cube_position, initial_cube_quat])
        mujoco_utils.set_joint_qpos(self.model, self.data, "object:joint", initial_cube_qpos)

        # Set initial position of the arm
        for joint_name, initial_position in INITIAL_ARM_POSITION.items():
            mujoco_utils.set_joint_qpos(self.model, self.data, joint_name, initial_position)

        # Run the simulation for a bunch of timesteps to let everything settle in.
        for _ in range(10):
            self._apply_action(np.zeros(self.action_space.shape))
            try:
                mujoco.mj_step(self.model, self.data, nstep=self.N_SUBSTEPS)
            except Exception:
                return False

    def step(self, action: np.ndarray):
        """Run one timestep of the environment's dynamics using the agent actions.

        Args:
            action (np.ndarray): Control action to be applied to the agent and update the simulation. Should be of shape :attr:`action_space`.

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
        action = -1 + (action * 2) / 10
        if self.EMA != None:
            action = self.EMA.update(action)
        self._apply_action(action)

        obs = self._get_obs()

        # Compute reward
        reward = self._compute_reward(obs)

        terminated = truncated = False
        if (self.total_timesteps >= self.MAX_TIMESTEPS):
            truncated = True


        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, self.info

    def _compute_reward(self, obs):
        # TODO: Try a 2 staged reward where reward for getting jaw to next to the cube then lessen the cube reward and add reward for cube to target
        # TODO: Try using previous distance - current distance maybe later
        # TODO: Change logging to the composition of 0.75 etc of each
        """Reward function"""
        object_jaw_diff = obs[22:25]
        object_target_diff = obs[25:28]
        reward = 0
        rew_jaw_center_to_object = -np.linalg.norm(object_jaw_diff)
        self.info["rew_jaw_center_to_object"] = rew_jaw_center_to_object
        rew_object_to_target = -np.linalg.norm(object_target_diff)
        self.info["rew_object_to_target"] = rew_object_to_target
        self.info["rew_other"] = 0
        # If not yet grasped the object
        if (self.grasp_reward > 0):
            # When grasped, immediate reward
            if (abs(rew_jaw_center_to_object) < 0.004):
                reward += self.grasp_reward
                self.grasp_reward = 0
                self.info["rew_other"] = self.grasp_reward
            reward += (0.75 * rew_jaw_center_to_object) + (0.25 * rew_object_to_target)
            self.info["rew_jaw_center_to_object_prop"] = (0.75 * rew_jaw_center_to_object)
            self.info["rew_object_to_target_prop"] = (0.25 * rew_object_to_target)
        else:
            reward += (0.25 * rew_jaw_center_to_object) + (0.75 * rew_object_to_target)
            self.info["rew_jaw_center_to_object_prop"] = (0.25 * rew_jaw_center_to_object)
            self.info["rew_object_to_target_prop"] = (0.75 * rew_object_to_target)

        if (abs(rew_object_to_target) < 0.002 and self.target_reached_reward > 0):
            reward += self.target_reached_reward
            self.info["is_success"] = 1
            self.target_reached_reward = 0
            self.info["rew_other"] += self.target_reached_reward
        return reward

    def _apply_action(self, action: np.ndarray):
        """
        Sends AI action (numpy array of numbers between -1 and 1) to Mujoco simulation and steps simulation to execute the action.
        Action can be relative position or absolute position based on settings
        """
        # TODO: Done
        ctrlrange = self.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.0

        if self.RELATIVE_CONTROL:
            actuation_center = np.zeros_like(action)
            for i in range(self.data.ctrl.shape[0]):
                # In xml, joints start with robot_ but actuators don't
                actuation_center[i] = self.data.get_joint_qpos(
                    self.model.actuator_names[i].lstrip("robot_", "")
                )
        else:
            actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.0
        self.data.ctrl[:] = actuation_center + action * actuation_range
        self.data.ctrl[:] = np.clip(self.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])
        mujoco.mj_step(self.model, self.data, nstep=self.N_SUBSTEPS)

    def _get_obs(self):
        """
        observation = [robot_qpos, robot_qvel, object_qpos, jaw_pos, object_jaw_diff, object_target_diff, self.goal]

        Indexing        [0 - 5 (robot_qpos), 6 - 11 (robot_qvel), 12 - 18 (object_qpos), 19 - 21 (jaw_pos), 22 - 24 (object_jaw_diff), 25 - 27 (object_target_diff), 28 - 30 (cube_qvel)]

        robot_qpos:          6 numbers. Joint angles (radians) of 6 motors
        robot_qvel:          6 numbers. Joint velocity (radians / sec) of 6 motors
        object_qpos:         7 numbers. Position (x,y,z) then orientation (w, x, y, z) of the cube
        jaw_pos:             3 numbers. Position (x,y,z) of the jaw
        object_jaw_diff:     3 numbers. Difference between position of the object and jaw (object - jaw)
        object_target_diff:  3 numbers. Difference between position of the object and target location (object - target)
        self.goal:           3 numbers. Position (x,y,z) of the goal
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
            [robot_qpos, robot_qvel,object_qpos, jaw_pos, object_jaw_diff, object_target_diff, self.goal])
        assert observation.shape == self.observation_space.shape, f"Expected obs shape {self.observation_space.shape} Actual shape {observation.shape}"
        return observation

    # --- other utility methods
    def render(self):
        """Render a frame of the Mujoco simulation.

        Returns:
            rgb image (np.ndarray): if render_mode is "rgb_array", return a 3D image array.
        """
        # Assign current state to target object but offset a bit so that the actual object
        # is not obscured.

        render_target = np.concatenate([self.goal, np.array([1, 0,0,0])])
        assert render_target.shape == (7,), f"Actual goal shape {render_target.shape}"

        mujoco_utils.set_joint_qpos(self.model, self.data, "target:joint", render_target)
        mujoco_utils.set_joint_qvel(self.model, self.data, "target:joint", np.zeros(6))

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
