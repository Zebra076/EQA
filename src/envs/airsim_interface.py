import airsim
import json
import msgpackrpc


def _patched_from_msgpack(cls, encoded):
    obj = cls()
    obj.__dict__ = {}
    for key, value in encoded.items():
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        if isinstance(value, dict):
            value = getattr(getattr(obj, key).__class__, "from_msgpack")(value)
        obj.__dict__[key] = value
    return obj


airsim.MsgpackMixin.from_msgpack = classmethod(_patched_from_msgpack)


def _patched_init(self, ip="", port=41451, timeout_value=3600):
    if ip == "":
        ip = "127.0.0.1"
    # 去掉 pack_encoding / unpack_encoding
    self.client = msgpackrpc.Client(
        msgpackrpc.Address(ip, port),
        timeout=timeout_value
    )

airsim.VehicleClient.__init__ = _patched_init

# from line_profiler import profile
import os
import time
import math
import numpy as np
import python_motion_planning as pmp
from scipy.spatial.transform import Rotation as R
import cv2

import binvox_rw


class AirSimInterface:
    def __init__(
        self,
        img_save_path: str = None,
        client_port: int = None,
        vehicle_name: str = "",
        camera_name: str = "cam1",
        external: bool = True,
        warmup: bool = True,
        warmup_attempts: int = 100,
        warmup_interval: float = 0.2,
    ):
        if img_save_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            img_save_path = os.path.join(current_dir, "tmp.png")
        self.img_save_path = img_save_path

        self.client_port = client_port if client_port is not None else 41451
        self.vehicle_name = vehicle_name
        self.camera_name = camera_name
        self.external = external
        self._planning_map = None
        self._planning_bounds = None
        self._planning_map_key = None
        self.client = airsim.MultirotorClient(port=self.client_port)
        self.client.confirmConnection()
        self.external_camera_count = self._get_external_camera_count() if self.external else None
        if warmup:
            self.warmup_cameras(
                attempts=warmup_attempts,
                interval=warmup_interval,
            )

    @staticmethod
    def get_default_settings_path():
        return os.path.expanduser("~/Documents/AirSim/settings.json")

    def _get_external_camera_count(self, settings_path=None, camera_prefix="cam"):
        if settings_path is None:
            settings_path = AirSimInterface.get_default_settings_path()
        settings_path = os.path.expanduser(settings_path)

        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"AirSim settings 文件不存在：{settings_path}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"AirSim settings JSON 解析失败：{settings_path}: {e}") from e
        except OSError as e:
            raise OSError(f"读取 AirSim settings 失败：{settings_path}: {e}") from e

        external_cameras = settings.get("ExternalCameras")
        if not isinstance(external_cameras, dict):
            raise ValueError(f"AirSim settings 缺少 ExternalCameras：{settings_path}")

        camera_count = 0
        while f"{camera_prefix}{camera_count + 1}" in external_cameras:
            camera_count += 1

        if camera_count == 0:
            raise ValueError(
                f"AirSim settings ExternalCameras 中没有连续相机 {camera_prefix}1：{settings_path}"
            )

        return camera_count

    @staticmethod
    def _is_black_image(img_rgb, max_threshold=20, mean_threshold=5.0):
        return img_rgb.max() <= max_threshold or img_rgb.mean() <= mean_threshold

    def _warmup_camera_names(self):
        if not self.external:
            return [self.camera_name]

        return [f"cam{i + 1}" for i in range(self.external_camera_count)]

    def warmup_cameras(
        self,
        attempts: int = 100,
        interval: float = 0.2,
        camera_names=None,
        max_threshold: int = 20,
        mean_threshold: float = 5.0,
    ):
        if attempts <= 0:
            return False

        if camera_names is None:
            camera_names = self._warmup_camera_names()
        else:
            camera_names = list(camera_names)

        if len(camera_names) == 0:
            return False

        requests = [
            airsim.ImageRequest(camera_name, airsim.ImageType.Scene, pixels_as_float=False, compress=False)
            for camera_name in camera_names
        ]

        for attempt in range(attempts):
            try:
                responses = self.client.simGetImages(
                    requests,
                    vehicle_name=self.vehicle_name,
                    external=self.external,
                )
                imgs_rgb = [self._response_to_rgb(response) for response in responses]
                if len(imgs_rgb) == len(camera_names) and all(
                    not self._is_black_image(
                        img_rgb,
                        max_threshold=max_threshold,
                        mean_threshold=mean_threshold,
                    )
                    for img_rgb in imgs_rgb
                ):
                    return True
            except Exception as e:
                if attempt == attempts - 1:
                    print(f"AirSim 相机预热失败：{e}")

            time.sleep(interval)

        print(f"AirSim 相机预热未检测到非黑帧，已丢弃 {attempts} 轮初始图像")
        return False

    def quat_angle_diff(self, q1, q2):
        r1 = R.from_quat([q1[0], q1[1], q1[2], q1[3]])  # 注意 SciPy 的顺序是 [x, y, z, w]
        r2 = R.from_quat([q2[0], q2[1], q2[2], q2[3]])
        r_rel = r1.inv() * r2
        angle_rad = r_rel.magnitude()
        return angle_rad

    # @profile
    def _set_pose(
        self,
        pose,
        vehicle_name: str = None,
        camera_name: str = None,
        external: bool = None,
    ):
        if vehicle_name is None:
            vehicle_name = self.vehicle_name
        if camera_name is None:
            camera_name = self.camera_name
        if external is None:
            external = self.external

        if external:
            self.client.simSetCameraPose(camera_name, pose, vehicle_name=vehicle_name, external=True)
        else:
            self.client.simSetVehiclePose(pose, True, vehicle_name=vehicle_name)
        
        pos = pose.position
        quat = pose.orientation

        if external:
            real_pose = self.client.simGetCameraInfo(
                camera_name,
                vehicle_name=vehicle_name,
                external=True,
            ).pose
        else:
            real_pose = self.client.simGetVehiclePose(vehicle_name=vehicle_name)
        real_pos = real_pose.position
        real_quat = real_pose.orientation

        delta_pos = math.sqrt((pos.x_val - real_pos.x_val)**2 + (pos.y_val - real_pos.y_val)**2 + (pos.z_val - real_pos.z_val)**2)
        delta_quat = self.quat_angle_diff(
            [real_quat.x_val, real_quat.y_val, real_quat.z_val, real_quat.w_val],
            [quat.x_val, quat.y_val, quat.z_val, quat.w_val]
        )

        return delta_pos, delta_quat

    # @profile
    def set_pose(
        self,
        pose,
        max_retries = 100000,
        vehicle_name: str = None,
        camera_name: str = None,
        external: bool = None,
    ):
        # Please use this safe set_pose
        delta_pos, delta_quat = self._set_pose(
            pose,
            vehicle_name=vehicle_name,
            camera_name=camera_name,
            external=external,
        )

        retry_count = 0
        while delta_pos > 1.0 or delta_quat > math.pi / 6:
            if retry_count >= max_retries:
                raise Exception("位姿到齐失败")

            time.sleep(0.0001)

            delta_pos, delta_quat = self._set_pose(
                pose,
                vehicle_name=vehicle_name,
                camera_name=camera_name,
                external=external,
            )
            
            retry_count += 1

        return delta_pos, delta_quat

    @staticmethod
    def _response_to_rgb(response):
        if len(response.image_data_uint8) == 0:
            raise Exception("获取图像失败：数据为空")

        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        pixel_count = response.height * response.width
        if pixel_count <= 0 or img1d.size % pixel_count != 0:
            raise Exception("获取图像失败：图像尺寸无效")

        channel_count = img1d.size // pixel_count
        if channel_count == 3:
            img_bgr = img1d.reshape(response.height, response.width, 3)
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if channel_count == 4:
            img_bgra = img1d.reshape(response.height, response.width, 4)
            return cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGB)

        raise Exception(f"获取图像失败：不支持的通道数 {channel_count}")

    def _get_obs_img_from_camera(
        self,
        vehicle_name: str = None,
        camera_name: str = None,
        external: bool = None,
    ):
        if vehicle_name is None:
            vehicle_name = self.vehicle_name
        if camera_name is None:
            camera_name = self.camera_name
        if external is None:
            external = self.external

        responses = self.client.simGetImages([
            airsim.ImageRequest(camera_name, airsim.ImageType.Scene, pixels_as_float=False, compress=False)
        ], vehicle_name=vehicle_name, external=external)

        return self._response_to_rgb(responses[0])

    @staticmethod
    def _save_image(img_rgb, save_path):
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        if img_rgb.ndim == 3 and img_rgb.shape[2] == 3:
            img_to_save = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        elif img_rgb.ndim == 3 and img_rgb.shape[2] == 4:
            img_to_save = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2BGRA)
        else:
            img_to_save = img_rgb

        ok = cv2.imwrite(save_path, img_to_save)
        if not ok:
            raise IOError(f"保存图像失败：{save_path}")

    @staticmethod
    def _indexed_save_path(save_path, index):
        save_dir, filename = os.path.split(save_path)
        stem, ext = os.path.splitext(filename)
        if not filename or not ext:
            return os.path.join(save_path, f"{index:06d}.png")
        return os.path.join(save_dir, f"{stem}_{index}{ext}")

    @staticmethod
    def _as_list(value, length, name):
        if isinstance(value, str):
            return [value] * length

        values = list(value)
        if len(values) != length:
            raise ValueError(f"{name} 数量必须和 poses 一致")
        return values

    # @profile
    def get_obs_img(
        self,
        pose,
        max_retries = 100000,
        save=False,
        vehicle_name: str = None,
        camera_name: str = None,
        external: bool = None,
    ):
        """
        Set one camera pose and capture a scene image.

        Returns:
            np.ndarray: RGB image suitable for VLM input. If save is True or a
                path string, the saved file is color-converted for normal viewing.
        """
        imgs_rgb = self.get_obs_imgs(
            [pose],
            vehicle_names=None if vehicle_name is None else [vehicle_name],
            max_retries=max_retries,
            save=save,
            camera_names=None if camera_name is None else [camera_name],
            pause_sim=True,
            external=external,
        )
        if imgs_rgb:
            return imgs_rgb[0]

    # @profile
    def get_obs_imgs(
        self,
        poses,
        vehicle_names=None,
        max_retries=100000,
        save=False,
        camera_names=None,
        pause_sim=True,
        external: bool = None,
    ):
        """
        Set multiple external cameras to different poses and capture images.

        Args:
            poses: List of airsim.Pose. For N poses, settings.json must define
                cam1...camN external cameras if camera_names is not provided.
            vehicle_names: AirSim vehicle names. Usually unused in ComputerVision
                external-camera mode.
            max_retries: Retry budget for each pose.
            save: False, True, or a path. True saves beside self.img_save_path
                as indexed files. Saved files are color-converted for normal viewing.
            camera_names: External camera names. Defaults to cam1...camN.
            pause_sim: Pause AirSim while setting poses and capturing images.
            external: True for ExternalCameras, False for vehicle cameras.

        Returns:
            List[np.ndarray]: RGB images in the same order as poses.
        """
        poses = list(poses)
        if len(poses) == 0:
            return []

        if external is None:
            external = self.external

        if camera_names is None:
            if len(poses) == 1:
                camera_names = [self.camera_name]
            else:
                camera_names = [f"cam{i + 1}" for i in range(len(poses))]
        else:
            camera_names = self._as_list(camera_names, len(poses), "camera_names")

        if vehicle_names is None:
            vehicle_names = [self.vehicle_name] * len(poses)
        else:
            vehicle_names = self._as_list(vehicle_names, len(poses), "vehicle_names")

        was_paused = None
        try:
            if pause_sim:
                was_paused = self.client.simIsPause()
                self.client.simPause(True)

            for pose, vehicle_name, camera_name in zip(poses, vehicle_names, camera_names):
                self.set_pose(
                    pose,
                    max_retries=max_retries,
                    vehicle_name=vehicle_name,
                    camera_name=camera_name,
                    external=external,
                )

            imgs_rgb = []
            if external and len(set(vehicle_names)) == 1:
                requests = [
                    airsim.ImageRequest(camera_name, airsim.ImageType.Scene, pixels_as_float=False, compress=False)
                    for camera_name in camera_names
                ]
                responses = self.client.simGetImages(
                    requests,
                    vehicle_name=vehicle_names[0],
                    external=True,
                )
                for index, response in enumerate(responses):
                    img_rgb = self._response_to_rgb(response)
                    if save:
                        base_path = save if isinstance(save, str) else self.img_save_path
                        self._save_image(img_rgb, self._indexed_save_path(base_path, index))
                    imgs_rgb.append(img_rgb)
            else:
                for index, (vehicle_name, camera_name) in enumerate(zip(vehicle_names, camera_names)):
                    img_rgb = self._get_obs_img_from_camera(
                        vehicle_name=vehicle_name,
                        camera_name=camera_name,
                        external=external,
                    )
                    if save:
                        base_path = save if isinstance(save, str) else self.img_save_path
                        self._save_image(img_rgb, self._indexed_save_path(base_path, index))
                    imgs_rgb.append(img_rgb)

            return imgs_rgb

        except Exception as e:
            print(f"批量获取观测图像失败: {e}")

        finally:
            if pause_sim and was_paused is not None:
                self.client.simPause(was_paused)

    @staticmethod
    def xyzyaw_to_pose(x, y, z, yaw):
        roll = 0.0
        pitch = 0.0
        
        quat = airsim.to_quaternion(pitch, roll, yaw)
        pose = airsim.Pose(airsim.Vector3r(x, y, z), quat)

        return pose

    @staticmethod
    def _yaw_from_point_to_point(p0, p1):
        direction = np.array(p1[:3]) - np.array(p0[:3])
        norm = np.linalg.norm(direction)
        if norm == 0:
            return 0.0
        direction = direction / norm
        return math.atan2(direction[1], direction[0])

    @staticmethod
    def _regularize_map_point(point, map_):
        return (
            max(min(point[0], map_.shape[0] - 1), 0),
            max(min(point[1], map_.shape[1] - 1), 0),
            max(min(point[2], map_.shape[2] - 1), 0),
        )

    @staticmethod
    def _load_planning_map(env_config_path, binvox_path, inflate_radius):
        with open(env_config_path, "r", encoding="utf-8") as f:
            env_config = json.load(f)

        x_range = [env_config["range"]["x_min"], env_config["range"]["x_max"]]
        y_range = [env_config["range"]["y_min"], env_config["range"]["y_max"]]
        z_range = [env_config["range"]["z_min"], env_config["range"]["z_max"]]

        x_range, y_range = y_range, x_range
        z_range = [-z_range[1], -z_range[0]]

        resolution = env_config["resolution"]
        bounds = [x_range, y_range, z_range]

        with open(binvox_path, "rb") as f:
            vox = binvox_rw.read_as_3d_array(f).data

        map_ = pmp.Grid(
            bounds=bounds,
            resolution=resolution,
            type_map=vox.astype(np.int8),
        )
        map_.inflate_obstacles(radius=inflate_radius)

        return map_, bounds

    def load_planning_map(self, env_config_path, binvox_path, inflate_radius=1):
        map_key = (env_config_path, binvox_path, inflate_radius)
        if (
            getattr(self, "_planning_map_key", None) == map_key
            and getattr(self, "_planning_map", None) is not None
        ):
            return self._planning_map, self._planning_bounds

        self._planning_map, self._planning_bounds = self._load_planning_map(
            env_config_path,
            binvox_path,
            inflate_radius,
        )
        self._planning_map_key = map_key
        return self._planning_map, self._planning_bounds

    @classmethod
    def _plan_rrt_star(cls, map_, start_w, goal_w, sample_num, stop_func=None):
        start_m = map_.world_to_map((start_w[1], start_w[0], -start_w[2]))
        start_m = cls._regularize_map_point(start_m, map_)
        goal_m = map_.world_to_map((goal_w[1], goal_w[0], -goal_w[2]))
        goal_m = cls._regularize_map_point(goal_m, map_)

        if stop_func is None:
            stop_func = lambda cur, fss, mss: (
                cur >= fss * 10 if fss is not None else False
            ) or (cur >= mss)
        planner_kwargs = {
            "map_": map_,
            "start": start_m,
            "goal": goal_m,
            "stop_func": stop_func,
        }

        try:
            path, path_info = pmp.RRTStar(
                **planner_kwargs,
                max_sample_step=sample_num,
            ).plan()
        except TypeError as e:
            if "max_sample_step" not in str(e):
                raise
            path, path_info = pmp.RRTStar(
                **planner_kwargs,
                sample_num=sample_num,
            ).plan()

        if not path_info["success"]:
            return []

        path_world = map_.path_map_to_world(path)
        return [(pt[1], pt[0], -pt[2]) for pt in path_world]

    @classmethod
    def _plan_segment(
        cls,
        map_,
        start_w,
        goal_w,
        sample_num,
        fallback_to_straight,
        stop_func=None
    ):
        try:
            segment = cls._plan_rrt_star(map_, start_w, goal_w, sample_num, stop_func=None)
            if len(segment) > 0:
                return segment, True
        except Exception as e:
            if not fallback_to_straight:
                raise RuntimeError(
                    f"path planning failed from {start_w} to {goal_w}"
                ) from e

        if fallback_to_straight:
            return [start_w, goal_w], False

        raise RuntimeError(f"path planning failed from {start_w} to {goal_w}")

    def plan_path(
        self,
        traj,
        env_config_path: str = None,
        binvox_path: str = None,
        inflate_radius: int = 1,
        sample_num: int = 30000,
        fallback_to_straight: bool = True,
        stop_func=None
    ):
        """
        Plan a collision-aware path for AirSim world-coordinate trajectory points.

        Args:
            traj: List of (x, y, z) or (x, y, z, yaw) waypoints.
            env_config_path: Environment config path containing range/resolution.
            binvox_path: Occupancy map path generated as map.binvox.
            inflate_radius: Obstacle inflation radius in grid cells.
            sample_num: RRTStar sample budget.
            fallback_to_straight: Return straight segments when planning fails.

        Returns:
            (pose_list, success), where pose_list contains (x, y, z, yaw).
        """
        if len(traj) < 2:
            pose_list = []
            for point in traj:
                if len(point) >= 4:
                    yaw = point[3]
                else:
                    yaw = 0.0
                pose_list.append((point[0], point[1], point[2], yaw))
            return pose_list, True

        if env_config_path is not None and binvox_path is not None:
            map_, _bounds = self.load_planning_map(
                env_config_path,
                binvox_path,
                inflate_radius,
            )
        elif getattr(self, "_planning_map", None) is not None:
            map_, _bounds = self._planning_map, self._planning_bounds
        else:
            raise ValueError(
                "planning map is not loaded; call load_planning_map() first "
                "or pass env_config_path and binvox_path"
            )

        path_segments = []
        success = False
        for i in range(len(traj) - 1):
            start_w = tuple(traj[i][:3])
            goal_w = tuple(traj[i + 1][:3])
            segment, segment_success = self._plan_segment(
                map_,
                start_w,
                goal_w,
                sample_num,
                fallback_to_straight,
                stop_func=None
            )

            segment[0] = tuple(traj[i])
            segment[-1] = tuple(traj[i + 1])
            path_segments.append(segment)
            success = success or segment_success

        full_path = []
        for i, segment in enumerate(path_segments):
            if i > 0 and segment[0] == full_path[-1]:
                full_path.extend(segment[1:])
            else:
                full_path.extend(segment)

        path = full_path
        pose_list = []
        for i, point in enumerate(path[:-1]):
            if len(point) >= 4:
                yaw = point[3]
            else:
                yaw = self._yaw_from_point_to_point(point, path[i + 1])

            pose_list.append((point[0], point[1], point[2], yaw))

        last_point = path[-1]
        if len(last_point) >= 4:
            last_yaw = last_point[3]
        elif pose_list:
            last_yaw = pose_list[-1][3]
        else:
            last_yaw = 0.0
        pose_list.append((last_point[0], last_point[1], last_point[2], last_yaw))

        return pose_list, success


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_save_dir = os.path.join(current_dir, "tmp")

    single_save_path = os.path.join(test_save_dir, "get_obs_img.png")
    batch_save_path = os.path.join(test_save_dir, "get_obs_imgs.png")

    interface = AirSimInterface(img_save_path=single_save_path, client_port=41453)
    test_poses = [
        airsim.Pose(
            airsim.Vector3r(0, 0, -2),
            airsim.to_quaternion(0, 0, 0),
        ),
        airsim.Pose(
            airsim.Vector3r(2, 0, -2),
            airsim.to_quaternion(0, 0, 0),
        ),
        airsim.Pose(
            airsim.Vector3r(-50, 0, -50),
            airsim.to_quaternion(0, 0, 0),
        ),
        airsim.Pose(
            airsim.Vector3r(-50, 0, -50),
            airsim.to_quaternion(0, 0, 1.57),
        ),
        airsim.Pose(
            airsim.Vector3r(-50, 0, -50),
            airsim.to_quaternion(0, 0, 3.14),
        ),
        airsim.Pose(
            airsim.Vector3r(-50, 0, -50),
            airsim.to_quaternion(0, 0, -1.57),
        ),
        airsim.Pose(
            airsim.Vector3r(0, 10, -2),
            airsim.to_quaternion(0, 0, 0),
        ),
        airsim.Pose(
            airsim.Vector3r(0, 20, -2),
            airsim.to_quaternion(0, 0, 0),
        ),
        airsim.Pose(
            airsim.Vector3r(0, 30, -2),
            airsim.to_quaternion(0, 0, 0),
        ),
        airsim.Pose(
            airsim.Vector3r(0, 40, -2),
            airsim.to_quaternion(0, 0, 0),
        ),
    ]

    img = interface.get_obs_img(test_poses[0], save=False)
    # img = interface.get_obs_img(test_poses[0], save=True)
    if img is not None:
        print(
            "get_obs_img returned RGB image: "
            f"shape={img.shape}, saved={AirSimInterface._indexed_save_path(single_save_path, 0)}"
        )

    imgs = interface.get_obs_imgs(test_poses, save=False)
    # imgs = interface.get_obs_imgs(test_poses, save=batch_save_path)
    if imgs is not None:
        print(
            "get_obs_imgs returned RGB images: "
            f"count={len(imgs)}, saved={AirSimInterface._indexed_save_path(batch_save_path, 0)}"
        )
