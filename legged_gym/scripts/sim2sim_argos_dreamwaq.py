"""
MuJoCo Deployment Script for GO2 Robot
使用 legged_robot 训练、argos_config 配置的模型部署脚本
"""

# from legged_gym.envs.Go2_MoB.GO2_Trot.GO2_Trot_config import GO2_Trot_Cfg_Yu
import math
import numpy as np
import mujoco
import mujoco.viewer
from collections import deque
from scipy.spatial.transform import Rotation as R
import torch
import time
import os
import glob 


# 在文件开头导入 键盘控制
import sys, tty, termios, fcntl
USE_KEYBOARD = True

def get_key():
    """非阻塞获取单个按键（Linux/macOS）"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    except:
        ch = None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
Robot_Scene = os.path.join(ROOT_DIR, 'resources/robots/xdog/xdog.xml') #robot路径
POLICY_DIR = os.path.join(ROOT_DIR, 'policy')
POLICY_PATTERN = 'policy_waq_xdog_latest.pt'


def find_latest_policy(policy_dir=POLICY_DIR, pattern=POLICY_PATTERN):
    policy_files = glob.glob(os.path.join(policy_dir, pattern))
    if not policy_files:
        raise FileNotFoundError(f"No policy file found: {os.path.join(policy_dir, pattern)}")
    return max(policy_files, key=os.path.getmtime)


Policy_File = find_latest_policy() #策略路径
"""
# 全局速度指令变量
x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 1.5, 0.0, 0.0
x_vel_max, y_vel_max, yaw_vel_max = 1.5, 0.5, 1.2"""

simulation_is_running=False


def quaternion_to_euler_array(quat):
    """四元数转欧拉角 [x, y, z, w] -> [roll, pitch, yaw]"""
    w, x, y, z= quat
    
    # Roll (x轴旋转)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y轴旋转)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z轴旋转)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    return np.array([roll_x, pitch_y, yaw_z])


def get_obs(data, model):
    """从 MuJoCo 数据结构中提取观测信息"""
    # 关节位置和速度 (12个关节)
    q = data.qpos[7:19].astype(np.double)
    dq = data.qvel[6:].astype(np.double)
    
    # 四元数姿态 (转换为 [x, y, z,con w] 格式)
    quat = data.qpos[3:7].astype(np.double)[[1, 2, 3, 0]]
    
    # 使用四元数旋转得到机身坐标系下的速度
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # 线速度（机身坐标系）
    omega = data.qvel[3:6].astype(np.double)  # MuJoCo free joint angular velocity is already in body frame
    
    # 投影重力向量（机身坐标系）
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    
    # 机身位置
    base_pos = data.qpos[0:3].astype(np.double)
    
    # 足端位置和接触力
    foot_positions = []
    foot_forces = []
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name is not None and 'foot' in body_name.lower():
            foot_positions.append(data.xpos[i][2].copy().astype(np.double))
            foot_forces.append(data.cfrc_ext[i][2].copy().astype(np.double))
    
    return (q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces)


def pd_control(target_q, q, kp, target_dq, dq, kd, default_dof_pos):
    """PD控制器计算力矩"""
    torque_out = (target_q + default_dof_pos - q) * kp + (target_dq - dq) * kd
    return torque_out


def run_mujoco(policy, cfg):
    """
    运行 MuJoCo 仿真
    
    Args:
        policy: 训练好的策略网络
        cfg: 配置对象
    """
    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
    print("=" * 60)
    print("MuJoCo 仿真启动 - Argos Robot")
    print("=" * 60)
    print("\n手柄控制已禁用，使用固定速度指令:")
    #print(f"  vx={x_vel_cmd:.2f}, vy={y_vel_cmd:.2f}, yaw={yaw_vel_cmd:.2f}")
    # print("  6/7: 快速前进/后退 (+/-0.3)")
    # print("  8/9: 快速左移/右移 (+/-0.3)")
    # print("  -/=: 快速左转/右转 (+/-0.5)")
    # print("  1: 快速重置所有命令")
    print("=" * 60)
    
    # 映射表 (基于 deploy.yaml)
    # 基于你的 go2.xml 物理顺序重写的映射表
    # 索引 0,1,2 是 FL; 3,4,5 是 FR; 6,7,8 是 RL; 9,10,11 是 RR
    # joint_ids_map = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
    joint_ids_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    #使用键盘控制定义量
    # 设置终端为非阻塞模式（仅适用于 Linux/macOS）
    if USE_KEYBOARD:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        except Exception as e:
            print("Warning: could not set terminal non-blocking", e)
    else:
        fd = None

    # 速度指令变量
    
    # 步长与限幅
    step_vx, step_vy, step_yaw = 0.25, 0.1, 0.2
    max_vx, min_vx = 2.5, -1.0
    max_vy, min_vy = 0.5, -0.5
    max_yaw, min_yaw = 1.57, -1.57
    #使用键盘
    
    # 加载 MuJoCo 模型
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    
    # 初始化关节位置
    num_actuated_joints = cfg.env.num_actions
    data.qpos[-num_actuated_joints:] = cfg.robot_config.default_dof_pos
    
    # 执行一次物理步
    mujoco.mj_step(model, data)
    
    # 启动 MuJoCo viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        sim_start_wall = time.perf_counter()
        next_step_wall = sim_start_wall
        count_lowlevel = 1

        simulation_is_running=True
        
        # 初始化观测历史缓存 (用于堆叠多帧观测)
        hist_obs = deque()
        for _ in range(cfg.env.num_observation_history):
            hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))
        
        target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
        action = np.zeros((cfg.env.num_actions), dtype=np.double)
        
        # 设置NumPy打印格式
        np.set_printoptions(formatter={'float': '{:0.4f}'.format})
        
        print("\n仿真开始运行...")
        print(f"预热时间: {cfg.sim_config.warmup_time}秒")
        print(f"观测维度: {cfg.env.num_observations} (当前帧: {cfg.env.num_single_obs}, 历史: {cfg.env.num_obs_history})")
        print(f"动作维度: {cfg.env.num_actions}")
        print(f"控制频率: {1/(cfg.sim_config.dt * cfg.sim_config.decimation):.1f} Hz")
        print("=" * 60 + "\n")
        
        try:
            while viewer.is_running() and data.time < cfg.sim_config.sim_duration and simulation_is_running:
                
                # ---------- 键盘控制（非阻塞）----------
                if USE_KEYBOARD:
                    try:
                        key = sys.stdin.read(1)
                        if key == '6':
                            x_vel_cmd = min(x_vel_cmd + step_vx, max_vx)
                            print(f"vx={x_vel_cmd:.2f}")
                        elif key == '7':
                            x_vel_cmd = max(x_vel_cmd - step_vx, min_vx)
                            print(f"vx={x_vel_cmd:.2f}")
                        elif key == '8':
                            y_vel_cmd = min(y_vel_cmd + step_vy, max_vy)
                            print(f"vy={y_vel_cmd:.2f}")
                        elif key == '9':
                            y_vel_cmd = max(y_vel_cmd - step_vy, min_vy)
                            print(f"vy={y_vel_cmd:.2f}")
                        elif key == '-':
                            yaw_vel_cmd = max(yaw_vel_cmd - step_yaw, min_yaw)
                            print(f"yaw={yaw_vel_cmd:.2f}")
                        elif key == '=':
                            yaw_vel_cmd = min(yaw_vel_cmd + step_yaw, max_yaw)
                            print(f"yaw={yaw_vel_cmd:.2f}")
                        elif key == '1':
                            x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
                            print("Reset to (0,0,0)")
                    except:
                        pass
                else:
                    # 固定速度模式：直接从配置或常数读取
                    x_vel_cmd = 1.5
                    y_vel_cmd = 0.0
                    yaw_vel_cmd = 0.0
                # 获取当前观测
                q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces = get_obs(data, model)
                    
                # 目标关节速度（通常为0）
                target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
                
                # 每隔 decimation 步更新一次策略
                if count_lowlevel % cfg.sim_config.decimation == 0:
                    
                    # 构建观测向量 (与 compute_observations 一致)
                    obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
                    
                    # 欧拉角
                    eu_ang = quaternion_to_euler_array(quat)
                    eu_ang[eu_ang > math.pi] -= 2 * math.pi
                    
                    # 观测顺序 (与 legged_robot.py 的 compute_observations 一致):
                    # ang_vel (3) + projected_gravity (3) + commands (3) + dof_pos (12) + dof_vel (12) + actions (12)
                    # 总计: 3+3+3+12+12+12 = 45维
                    idx = 0
                    
                    # 1. 角速度 (3维，机身坐标系)
                    obs[0, idx:idx+3] = omega * cfg.normalization.obs_scales.ang_vel
                    idx += 3
                    
                    # 2. 投影重力 (3维，机身坐标系)
                    obs[0, idx:idx+3] = gvec  # 已经是单位向量，通常不需要缩放
                    idx += 3
                    
                    # 3. 指令 (3维: vx, vy, yaw_rate)
                    obs[0, idx:idx+3] = np.array([x_vel_cmd, y_vel_cmd, yaw_vel_cmd]) * cfg.normalization.obs_scales.commands_scale
                    idx += 3
                    
                    # 4. 关节位置 (12维，相对于默认位置)
                    obs[0, idx:idx+12] = ((q - cfg.robot_config.default_dof_pos) * cfg.normalization.obs_scales.dof_pos)[joint_ids_map]
                    idx += 12
                    
                    # 5. 关节速度 (12维)
                    obs[0, idx:idx+12] = (dq * cfg.normalization.obs_scales.dof_vel)[joint_ids_map]
                    idx += 12
                    
                    # 6. 上一步动作 (12维)
                    obs[0, idx:idx+12] = action
                    idx += 12
                    # 现在 idx = 45，正确！
                    
                    # 裁剪观测
                    obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
                    
                    # 更新历史观测：训练端 HistoryWrapper 使用旧帧在前、新帧在后的顺序
                    hist_obs.append(obs)
                    hist_obs.popleft()

                    # 策略导出模型输入格式: [当前 obs(45), obs_history(45*5)]
                    policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
                    policy_input[0, :cfg.env.num_single_obs] = obs[0, :]
                    for i in range(cfg.env.num_observation_history):
                        start_idx = cfg.env.num_single_obs + i * cfg.env.num_single_obs
                        end_idx = start_idx + cfg.env.num_single_obs
                        policy_input[0, start_idx:end_idx] = hist_obs[i][0, :]
                    
                    # 策略推理
                    with torch.no_grad():
                        action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                        
                    
                    # 裁剪动作
                    action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
                    
                    # 计算目标关节角度
                    # 1. 基础缩放
                    actions_scaled = action * cfg.control.action_scale
                    
                    
                    
                    # 2. 髋关节(Hip)单独缩放 (索引 0, 3, 6, 9)
                    # 对应 legged_robot.py 中的 hip_reduction 处理
                    # hip_indices = [0, 3, 6, 9]
                    # hip_reduction = cfg.control.hip_reduction
                    # actions_scaled[hip_indices] *= hip_reduction
                    
                    target_q[joint_ids_map] = actions_scaled
                
                # PD 控制计算力矩
                # 预热阶段：机器人保持初始姿态
                if data.time < cfg.sim_config.warmup_time:
                    tau = pd_control(
                            np.zeros((cfg.env.num_actions)), q,
                            cfg.robot_config.kps, target_dq, dq,
                            cfg.robot_config.kds, cfg.robot_config.default_dof_pos
                        )
                else:
                    tau = pd_control(
                        target_q, q,
                        cfg.robot_config.kps, target_dq, dq,
                        cfg.robot_config.kds, cfg.robot_config.default_dof_pos
                    )
                
                # 力矩限幅
                tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
                
                # 应用力矩并执行仿真步
                data.ctrl = tau 
                mujoco.mj_step(model, data)
                count_lowlevel += 1
                
                # 同步 viewer
                viewer.sync()

                # 实时同步：每个物理步对应 cfg.sim_config.dt 秒真实时间
                next_step_wall += cfg.sim_config.dt
                sleep_time = next_step_wall - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_step_wall = time.perf_counter()
                # 每 100 个物理步（即 0.5 秒，因为 dt=0.005）打印一次高度和速度指令
                if count_lowlevel % 100 == 0:
                    print(f"time={data.time:.2f}s  base_z={data.qpos[2]:.3f}  "
                        f"vx={x_vel_cmd:.2f}  vy={y_vel_cmd:.2f}  yaw={yaw_vel_cmd:.2f}", flush=True)
        finally:
            if USE_KEYBOARD and fd is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    print("\n退出仿真")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='MuJoCo部署脚本 - Argos')
    parser.add_argument('--load_model', type=str, 
                       default=Policy_File,
                       help=f'策略模型路径 (.pt文件); 默认加载 {POLICY_DIR} 下最新的 {POLICY_PATTERN}')
    parser.add_argument('--terrain', action='store_true', 
                       help='是否使用地形 (默认:平地)')
    args = parser.parse_args()
    
    
    class Sim2simCfg():
        """MuJoCo 仿真配置"""
        
        class env():
            # 观测相关配置
            num_single_obs = 45  # 单帧观测维度: 3+3+3+12+12+12=45
            num_observation_history = 5  # 与训练端 cfg.env.num_observation_history 一致
            num_obs_history = num_single_obs * num_observation_history
            num_observations = num_single_obs + num_obs_history  # 当前观测 + 历史观测 = 270
            num_actions = 12  # 动作维度
        
        class sim_config:
            # MuJoCo 模型路径
            mujoco_model_path = Robot_Scene
            
            # 仿真参数
            sim_duration = 120.0  # 仿真时长（秒）
            dt = 0.005  # 仿真时间步长 (200Hz)
            decimation = 4  # 控制降采样 (策略频率 = 200/4 = 50Hz)
            warmup_time = 1.0  # 预热时间（秒）
        
        class robot_config:
            # PD 控制器增益 (与训练配置一致)
            # go2_config.py: stiffness=20, damping=0.5
            kps = np.array([20, 20, 20] * 4, dtype=np.double)  # 
            kds = np.array([0.5, 0.5, 0.5] * 4, dtype=np.double)  # 
            
            # 力矩限制
            tau_limit = np.array([23.7, 23.7, 45.43] * 4, dtype=np.double)
            
            # 默认关节角度 (与训练配置一致)
            default_dof_pos = np.array([
                -0.1, 0.8, -1.5,    # FL (前左)
                0.0, 0.8, -1.5,   # FR (前右)
                -0.1, 0.8, -1.5,    # HL/RL (后左)
                0.0, 0.8, -1.5,   # RR (后右)
            ], dtype=np.double)

        class control():
            # PD Drive parameters:
            # control_type = 'P'
            # stiffness = {'joint': 30.0}  # [N*m/rad]
            # damping = {'joint': 1}     # [N*m*s/rad]
            # # action scale: target angle = actionScale * action + defaultAngle
            action_scale = 0.25
        
        class normalization():
            # 观测缩放因子
            class obs_scales:
                commands_scale = np.array([2.0, 2.0, 0.25])  # [vx, vy, yaw_rate]
                lin_vel = 2.0
                ang_vel = 0.25
                dof_pos = 1.0
                dof_vel = 0.05
                quat = 1.0
            
            # 裁剪范围
            clip_observations = 100.
            clip_actions = 100.
    
    
    # 加载训练好的策略
    print(f"\n正在加载策略: {args.load_model}")
    try:
        policy = torch.jit.load(args.load_model)
        policy.eval()
        print("策略加载成功!")
    except Exception as e:
        print(f"策略加载失败: {e}")
        exit(1)
    
    # 运行 MuJoCo 仿真
    run_mujoco(policy, Sim2simCfg())

if __name__ == '__main__':
    main()
