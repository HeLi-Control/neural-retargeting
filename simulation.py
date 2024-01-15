import pybullet
import time
import pybullet_data
import h5py
import os


class YuMi_Simulation:
    def __init__(self):
        # Connect the client
        self.client = pybullet.connect(pybullet.GUI)
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
            cameraDistance=1.8,
            cameraYaw=85,
            cameraPitch=-35,
            cameraTargetPosition=[0, 0, 0],
        )
        # Get joints available
        self.available_joints_indices = [
            i
            for i in range(pybullet.getNumJoints(self.robot_id))
            if pybullet.getJointInfo(self.robot_id, i)[2] != pybullet.JOINT_FIXED
        ]
        # Set default color
        self.rgba_color = pybullet.getVisualShapeData(self.robot_id)[
            self.available_joints_indices[0]
        ][7]

    def close_simulation(self):
        pybullet.disconnect(self.client)

    def step_simulation(self, target_positions):
        # Control joints
        pybullet.stepSimulation()
        pybullet.setJointMotorControlArray(
            bodyUniqueId=yumi.robot_id,
            jointIndices=yumi.available_joints_indices,
            controlMode=pybullet.POSITION_CONTROL,
            targetPositions=target_positions,
        )
        # check collision
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
                    print(
                        "Collision Occurred in {} & {}!!!".format(
                            self.all_joints_names[contact[3]],
                            self.all_joints_names[contact[4]],
                        )
                    )
                    pybullet.changeVisualShape(
                        self.robot_id, contact[3], rgbaColor=[1, 0, 0, 1]
                    )
                    pybullet.changeVisualShape(
                        self.robot_id, contact[4], rgbaColor=[1, 0, 0, 1]
                    )
            else:
                pybullet.changeVisualShape(
                    self.robot_id, joint, rgbaColor=self.rgba_color
                )

    @property
    def available_joints_numbers(self):
        return len(self.available_joints_indices)

    @property
    def available_joints_names(self):
        return [
            pybullet.getJointInfo(self.robot_id, index)[1]
            for index in self.available_joints_indices
        ]

    @property
    def all_joints_names(self):
        return [
            pybullet.getJointInfo(self.robot_id, i)[1]
            for i in range(pybullet.getNumJoints(self.robot_id))
        ]


if __name__ == "__main__":
    # Initialize the simulation
    yumi = YuMi_Simulation()
    # Read inferenced data
    h5_files = []
    for path, _, file_list in os.walk("saved/inferenced"):
        for file in file_list:
            if file.endswith(".h5"):
                h5_files.append(os.path.join(path, file).replace("\\", "/"))
    h5_file = h5py.File(h5_files[0], "r")  # Read only the first file
    # Display all frames
    for index in range(len(list(h5_file["l_hand"]))):
        data_frame = (
            list(h5_file["l_arm"])[index].tolist()
            + list(h5_file["l_hand"])[index].tolist()
            + list(h5_file["r_arm"])[index].tolist()
            + list(h5_file["r_hand"])[index].tolist()
        )
        yumi.step_simulation(data_frame)
        time.sleep(0.005)
    print(yumi.available_joints_names)
    # Stop the simulation
    yumi.close_simulation()
