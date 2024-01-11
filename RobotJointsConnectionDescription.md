# 机器人关节连接关系

从 ROS 中导出了一份连接关系的 `yumi-with-hands.pdf` 位于本目录下。

-  `world` 连杆连接到无运动自由度的 `world_joint` 关节
- `world_joint` 关节连接到固定的 `yumi_base_link` 连杆
- `yumi_base_link` 连杆通过固定关节 `yumi_base_link_to_body` 连接到 `yumi_body`【实际上这两个连杆的坐标系是完全重合的】
- `yumi_body` 分别连接到左右两侧的肩膀
  - 从肩膀到手腕总共有7个自由度，这里连接顺序为（以左肩膀为例，右肩膀同理）：
  - `yumi_body --yumi_joint_1_l-> yumi_link_1_l --yumi_joint_2_l-> yumi_link_2_l --yumi_joint_7_l->  yumi_link_3_l --yumi_joint_3_l-> yumi_link_4_l --yumi_joint_4_l-> yumi_link_5_l --yumi_joint_5_l-> yumi_link_6_l --yumi_joint_6_l-> yumi_link_7_l`
  - 训练使用的肩关节是 `yumi_joint_2_l` 和 `yumi_joint_2_r`
  - 训练使用的肘关节是 `yumi_joint_4_l` 和 `yumi_joint_4_r`
  - 训练使用的末端执行关节是 `yumi_joint_6_l`  和 `yumi_joint_6_r`
- 接下来是手掌部分
  - 从手末端连杆 `yumi_link_7_l` 出发连接了关节 `yumi_link_7_l_joint`，该关节通过连杆 `right_hand_base` 连接到了关节 `Link111_shift`
  - 从关节`Link111_shift` 出发连接了五根手指，除了拇指存在三个关节以外剩下的四根手指都只有两个关节
  - 训练使用的手掌跟关节是 `Link111_shift`
  - 训练使用的手掌肘关节是 `Link1`, `Link2`, `Link3`, `Link4` 和 `Link5`
  - 训练使用的末端执行关节是 `Link11`, `Link22`, `Link33`, `Link44` 和 `Link53`

臂部分共14个关节，12个连杆

手部共13个关节，12个连杆