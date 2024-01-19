import pybullet
import pybullet_data
import h5py
import os
from models.kinematics import ForwardKinematicsURDF
import torch
from scipy.spatial.transform import Rotation

import dataset
from loguru import logger
from utils.util import create_folder


class Display_Debug_Msg:
    def __init__(self, debug_cfg, inferenced_file=None, demonstrate_file=None) -> None:
        self.debug_message = debug_cfg
        self.targets = sorted(
            [
                target
                for target in getattr(dataset, "YumiAll")(root="./data/target/yumi-all")
            ],
            key=lambda target: target.skeleton_type,
        )[0]
        self.disp_life_time = 0.01
        self.human_demonstrate = (
            demonstrate_file if demonstrate_file is not None else None
        )
        self.inferenced_file = inferenced_file if inferenced_file is not None else None
        self.hand_fk = ForwardKinematicsURDF()

    def __calc_hand_forward_kinematics(self, l_arm, r_arm):
        _, _, global_pos = self.hand_fk(
            torch.cat([l_arm, r_arm]),
            self.targets.parent,
            self.targets.offset,
            1,
        )
        global_pos[:, 2] = global_pos[:, 2] + 0.51492
        return global_pos

    def disp_forward_kinematics(self, index=None, l_arm=None, r_arm=None):
        # Plot forward kinematics
        if self.inferenced_file is not None:
            global_pos = self.__calc_hand_forward_kinematics(
                torch.tensor(self.inferenced_file["l_arm"][index]),
                torch.tensor(self.inferenced_file["r_arm"][index]),
            )
        else:
            global_pos = self.__calc_hand_forward_kinematics(
                torch.tensor(l_arm),
                torch.tensor(r_arm),
            )
        pybullet.addUserDebugPoints(
            pointPositions=global_pos.tolist(),
            pointColorsRGB=[[1, 0, 0] for _ in global_pos.tolist()],
            pointSize=10,
            lifeTime=self.disp_life_time,
        )

    def disp_demonstrate_arm(self, index=None, l_arm=None, r_arm=None):
        if index is not None:
            froms = (
                self.human_demonstrate["l_arm"][index][:2] + [0, 1, 1]
            ).tolist() + (
                self.human_demonstrate["r_arm"][index][:2] + [0, 1, 1]
            ).tolist()
            tos = (self.human_demonstrate["l_arm"][index][1:] + [0, 1, 1]).tolist() + (
                self.human_demonstrate["r_arm"][index][1:] + [0, 1, 1]
            ).tolist()
        else:
            froms = (l_arm[:2] + [0, 1, 1]).tolist() + (r_arm[:2] + [0, 1, 1]).tolist()
            tos = (l_arm[1:] + [0, 1, 1]).tolist() + (r_arm[1:] + [0, 1, 1]).tolist()
        for f, t in zip(froms, tos):
            pybullet.addUserDebugLine(
                lineFromXYZ=f,
                lineToXYZ=t,
                lineColorRGB=[1, 0, 1],
                lineWidth=4,
                lifeTime=self.disp_life_time,
            )

    def disp_end_effector_orientation(
        self, index=None, ee_matrix_l=None, ee_matrix_r=None
    ):
        if self.human_demonstrate is not None:
            quaternion1 = Rotation.from_quat(self.human_demonstrate["ee_ori"][index][0])
            quaternion2 = Rotation.from_quat(self.human_demonstrate["ee_ori"][index][1])
            ee_matrix_l = quaternion1.as_matrix().tolist()
            ee_matrix_r = quaternion2.as_matrix().tolist()
        pybullet.addUserDebugLine(
            lineFromXYZ=[0, 0.35, 1],
            lineToXYZ=[
                i + j * 0.3
                for i, j in zip(
                    [0, 0.35, 1],
                    [ee_matrix_l[0][0], ee_matrix_l[1][0], ee_matrix_l[2][0]],
                )
            ],
            lineColorRGB=[0, 0, 1],
            lineWidth=4,
            lifeTime=self.disp_life_time,
        )
        pybullet.addUserDebugLine(
            lineFromXYZ=[0, -0.35, 1],
            lineToXYZ=[
                i + j * 0.3
                for i, j in zip(
                    [0, -0.35, 1],
                    [ee_matrix_r[0][0], ee_matrix_r[1][0], ee_matrix_r[2][0]],
                )
            ],
            lineColorRGB=[0, 0, 1],
            lineWidth=4,
            lifeTime=self.disp_life_time,
        )

    def print_loss_message(self, loss):
        logger.info(
            "EE {:.2f} | Vec {:.2f}| Ori {:.2f} | Fin {:.2f}".format(
                loss[0],
                loss[1],
                loss[2],
                loss[3],
            )
        )

    def print_human_demonstrate_arms(self, index):
        # print demonstrate arm joints' positions
        logger.debug(
            "Human right shoulder-elbow vector: "
            + str(
                self.human_demonstrate["r_arm"][index][1]
                - self.human_demonstrate["r_arm"][index][0]
            )
        )
        logger.debug(
            "Human right shoulder-wrist vector: "
            + str(
                self.human_demonstrate["r_arm"][index][2]
                - self.human_demonstrate["r_arm"][index][0]
            )
        )
        logger.debug(
            "Human left shoulder-elbow vector: "
            + str(
                self.human_demonstrate["l_arm"][index][1]
                - self.human_demonstrate["l_arm"][index][0]
            )
        )
        logger.debug(
            "Human left shoulder-wrist vector: "
            + str(
                self.human_demonstrate["l_arm"][index][2]
                - self.human_demonstrate["l_arm"][index][0]
            )
        )
        # print robot arm joints' positions
        global_pos = self.__calc_hand_forward_kinematics(
            torch.tensor(self.inferenced_file["l_arm"][index]),
            torch.tensor(self.inferenced_file["r_arm"][index]),
        )
        logger.debug(
            "Robot right shoulder-elbow vector: "
            + str((global_pos[11] - global_pos[7]).tolist())
        )
        logger.debug(
            "Robot right shoulder-wrist vector: "
            + str((global_pos[13] - global_pos[7]).tolist())
        )
        logger.debug(
            "Robot left shoulder-elbow vector: "
            + str((global_pos[4] - global_pos[0]).tolist())
        )
        logger.debug(
            "Robot left shoulder-wrist vector: "
            + str((global_pos[6] - global_pos[0]).tolist())
        )

    def disp(
        self,
        index=None,
        l_arm_ang=None,
        r_arm_ang=None,
        l_arm_demonstrate=None,
        r_arm_demonstrate=None,
        ee_matrix_l=None,
        ee_matrix_r=None,
    ):
        if self.debug_message["draw_debug"]:
            self.disp_forward_kinematics(index, l_arm_ang, r_arm_ang)
            self.disp_demonstrate_arm(index, l_arm_demonstrate, r_arm_demonstrate)
            self.disp_end_effector_orientation(index, ee_matrix_l, ee_matrix_r)
        if self.debug_message["print_debug"]:
            self.print_loss_message(self.inferenced_file["loss"][index].tolist())
            self.print_human_demonstrate_arms(index)


class YuMi_Simulation:
    def __init__(
        self,
        debug_message_cfg,
        disp_gui=True,
        save_video_path=None,
        inferenced_file=None,
        demonstrate_file=None,
    ):
        self.inferenced_file = inferenced_file if inferenced_file is not None else None
        # Connect the client
        self.client = (
            pybullet.connect(pybullet.GUI)
            if disp_gui
            else pybullet.connect(pybullet.DIRECT)
        )
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        # Add source path
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Load land
        pybullet.loadURDF("plane.urdf")
        # Load yumi
        self.robot_id = pybullet.loadURDF(
            fileName="data/yumi_urdf_all/yumi_with_hands.urdf",
            flags=pybullet.URDF_USE_SELF_COLLISION
            + pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
        )
        # Set camera
        pybullet.resetDebugVisualizerCamera(
            cameraDistance=1.2,
            cameraYaw=85,
            cameraPitch=-20,
            cameraTargetPosition=[0, 0, 0.93],
        )
        # Get joints available
        self.available_joints_indices = [
            i
            for i in range(pybullet.getNumJoints(self.robot_id))
            if pybullet.getJointInfo(self.robot_id, i)[2] != pybullet.JOINT_FIXED
        ]
        # Set default color
        self.default_color = pybullet.getVisualShapeData(self.robot_id)[
            self.available_joints_indices[0]
        ][7]
        self.red_color = [1, 0, 0, 1]
        # Debug messages
        self.debug_msg_disp = Display_Debug_Msg(
            debug_cfg=debug_message_cfg,
            inferenced_file=self.inferenced_file,
            demonstrate_file=demonstrate_file,
        )
        # Start video capturing
        if (save_video_path is None) or (save_video_path == []):
            self.save_video = False
            self.save_video_path = []
            self.video_id = None
        else:
            self.save_video = True
            self.save_video_path = save_video_path
            self.video_id = pybullet.startStateLogging(
                pybullet.STATE_LOGGING_VIDEO_MP4, save_video_path
            )

    def close_simulation(self):
        if self.save_video:
            pybullet.stopStateLogging(self.video_id)
        pybullet.disconnect(self.client)
        if self.inferenced_file is not None:
            self.inferenced_file.close()

    def step_simulation(self, index=None, given_joints_angles=None):
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SINGLE_STEP_RENDERING)
        if index is not None:
            # right arm, right arm, left arm, left hand
            data_frame = (
                self.inferenced_file["r_arm"][index].tolist()
                + self.inferenced_file["r_hand"][index].tolist()
                + self.inferenced_file["l_arm"][index].tolist()
                + self.inferenced_file["l_hand"][index].tolist()
            )
            # Control joints
            pybullet.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=self.available_joints_indices,
                controlMode=pybullet.POSITION_CONTROL,
                targetPositions=data_frame,
            )
        elif given_joints_angles is not None:
            # Control joints
            pybullet.stepSimulation()
            pybullet.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=self.available_joints_indices,
                controlMode=pybullet.POSITION_CONTROL,
                targetPositions=given_joints_angles,
            )
        pybullet.stepSimulation()
        # Collision check
        self.collision_check()
        # Display debug messages
        self.debug_msg_disp.disp(index)

    def collision_check(self):
        for joint in self.available_joints_indices:
            if (
                len(
                    pybullet.getContactPoints(
                        bodyA=self.robot_id,
                        linkIndexA=joint,
                    )
                )
                > 0
            ):
                for contact in pybullet.getContactPoints(
                    bodyA=self.robot_id, linkIndexA=joint
                ):
                    logger.error(
                        "Collision Occurred between {} & {}".format(
                            str(self.all_joints_names[contact[3]])[2:-1],
                            str(self.all_joints_names[contact[4]])[2:-1],
                        )
                    )
                    if contact[3] in self.available_joints_indices:
                        pybullet.changeVisualShape(
                            self.robot_id, contact[3], rgbaColor=self.red_color
                        )
                    if contact[4] in self.available_joints_indices:
                        pybullet.changeVisualShape(
                            self.robot_id, contact[4], rgbaColor=self.red_color
                        )
            else:
                pybullet.changeVisualShape(
                    self.robot_id, joint, rgbaColor=self.default_color
                )

    @property
    def available_joints_numbers(self):
        return len(self.available_joints_indices)

    @property
    def available_joints_names(self):
        return [
            pybullet.getJointInfo(self.robot_id, i)[1]
            for i in self.available_joints_indices
        ]

    @property
    def all_joints_numbers(self):
        return len(self.all_joints_names)

    @property
    def all_joints_names(self):
        return [
            pybullet.getJointInfo(self.robot_id, i)[1]
            for i in range(pybullet.getNumJoints(self.robot_id))
        ]

    @property
    def data_frame_num(self):
        return (
            len(list(self.inferenced_file["l_hand"]))
            if self.inferenced_file is not None
            else 0
        )


def search_inferenced_files(path_name) -> list:
    h5_files = []
    for path, _, file_list in os.walk(path_name):
        for file in file_list:
            if file.endswith(".h5") and ("humanDemonstrate" not in file):
                h5_files.append(os.path.join(path, file).replace("\\", "/"))
    return h5_files


if __name__ == "__main__":
    inferenced_files = search_inferenced_files("./saved/inferenced")
    log_folder = "./saved/log/simulation"
    create_folder(log_folder)
    for file in inferenced_files:
        base_file_name = os.path.basename(file)[:-3]
        # Start log file
        logger.add(
            sink=log_folder + "simulation_" + base_file_name + ".log",
            rotation="5MB",
            encoding="utf-8",
        )
        # Initialize the simulation
        yumi = YuMi_Simulation(
            debug_message_cfg={"draw_debug": False, "print_debug": True},
            inferenced_file=h5py.File(file, "r"),
            demonstrate_file=h5py.File("./saved/inferenced/humanDemonstrate.h5", "r"),
        )
        logger.info("Simulating file:" + file)
        # Display all frames
        for index in range(yumi.data_frame_num):
            yumi.step_simulation(index)
        # Stop the simulation
        yumi.close_simulation()
        del yumi
