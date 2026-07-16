# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value
import scipy.io
import pandas as pd

class Logger:
    def __init__(self, dt, dir_path):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None
        self.filename= dir_path + '/run_log.mat'

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def save_run(self):
        print('projected_gravity.size:',len(self.state_log['projected_gravity']))
        print('base_ang_vel.size:',len(self.state_log['base_ang_vel']))
        print('dof_pos.size:',len(self.state_log['dof_pos']))
        
        
        #保存观测值用于raisim
        num_ = len(self.state_log['obs']) 
        variable = pd.DataFrame(self.state_log['obs'][0]) 
        variable1 = pd.DataFrame(self.state_log['actions'][0]) 
        variable2 = pd.DataFrame(self.state_log['obs_history'][0]) 
        for i in range(1,num_): 
            variable = pd.concat([ variable,pd.DataFrame(self.state_log['obs'][i])]) 
            variable1 = pd.concat([ variable1,pd.DataFrame(self.state_log['actions'][i])]) 
            variable2 = pd.concat([ variable2,pd.DataFrame(self.state_log['obs_history'][i])]) 
        variable.to_csv('observersion_input.csv', header = None, index = None)  
        variable1.to_csv('action_output.csv', header = None, index = None)  
        variable2.to_csv('observersion_history_input.csv', header = None, index = None)  
           
        # scipy.io.savemat(self.filename,
        #                  mdict=self.state_log)
        
        

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def _plot(self):
        # nb_rows = 2
        # nb_cols = 2
        # fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        

        # if log["base_vel_x"]: plt.plot(time, log["base_vel_x"], label='base_vel_x')
        # if log["base_vel_y"]: plt.plot(time, log["base_vel_y"], label='base_vel_y')
        # if log["base_vel_z"]: plt.plot(time, log["base_vel_z"], label='base_vel_z')
        # if log["base_vel_x"]: plt.plot(time, log["base_vel_x"], label='base_vel_x')
        # if log["base_vel_y"]: plt.plot(time, log["base_vel_y"], label='base_vel_y')
        # if log["base_vel_z"]: plt.plot(time, log["base_vel_z"], label='base_vel_z')
        # if log["esti_lin_vel_x"]: plt.plot(time, log["esti_lin_vel_x"], label='esti_lin_vel_x')
        # if log["esti_lin_vel_y"]: plt.plot(time, log["esti_lin_vel_y"], label='esti_lin_vel_y')
        # if log["esti_lin_vel_z"]: plt.plot(time, log["esti_lin_vel_z"], label='esti_lin_vel_z')

        #to estimate the pos
        if log["base_x"]: plt.plot(time, log["base_x"], label='base_x')
        if log["base_y"]: plt.plot(time, log["base_y"], label='base_y')
        if log["base_z"]: plt.plot(time, log["base_z"], label='base_z')

        # if log["command_x"]: plt.plot(time, log["command_x"], label='command_x')
        # if log["command_y"]: plt.plot(time, log["command_y"], label='command_y')
        # if log["command_yaw"]: plt.plot(time, log["command_yaw"], label='command_yaw')
        plt.legend()
        plt.show()
        
        
        if log["torques_0"]: plt.plot(time, log["torques_0"], label='torques_0')
        if log["torques_1"]: plt.plot(time, log["torques_1"], label='torques_1')
        if log["torques_2"]: plt.plot(time, log["torques_2"], label='torques_2')
        if log["torques_3"]: plt.plot(time, log["torques_3"], label='torques_3')
        if log["torques_4"]: plt.plot(time, log["torques_4"], label='torques_4')
        if log["torques_5"]: plt.plot(time, log["torques_5"], label='torques_5')
        if log["torques_6"]: plt.plot(time, log["torques_6"], label='torques_6')
        if log["torques_7"]: plt.plot(time, log["torques_7"], label='torques_7')
        if log["torques_8"]: plt.plot(time, log["torques_8"], label='torques_8')
        if log["torques_9"]: plt.plot(time, log["torques_9"], label='torques_9')
        if log["torques_10"]: plt.plot(time, log["torques_10"], label='torques_10')
        if log["torques_11"]: plt.plot(time, log["torques_11"], label='torques_11')
        plt.legend()
        plt.show()
        
        if log["dof_pos_0"]: plt.plot(time, log["dof_pos_0"], label='dof_pos_0')
        if log["dof_pos_1"]: plt.plot(time, log["dof_pos_1"], label='dof_pos_1')
        if log["dof_pos_2"]: plt.plot(time, log["dof_pos_2"], label='dof_pos_2')
        if log["dof_pos_3"]: plt.plot(time, log["dof_pos_3"], label='dof_pos_3')
        if log["dof_pos_4"]: plt.plot(time, log["dof_pos_4"], label='dof_pos_4')
        if log["dof_pos_5"]: plt.plot(time, log["dof_pos_5"], label='dof_pos_5')
        if log["dof_pos_6"]: plt.plot(time, log["dof_pos_6"], label='dof_pos_6')
        if log["dof_pos_7"]: plt.plot(time, log["dof_pos_7"], label='dof_pos_7')
        if log["dof_pos_8"]: plt.plot(time, log["dof_pos_8"], label='dof_pos_8')
        if log["dof_pos_9"]: plt.plot(time, log["dof_pos_9"], label='dof_pos_9')
        if log["dof_pos_10"]: plt.plot(time, log["dof_pos_10"], label='dof_pos_10')
        if log["dof_pos_11"]: plt.plot(time, log["dof_pos_11"], label='dof_pos_11')
        plt.legend()
        plt.show()

        if log["dof_V_0"]: plt.plot(time, log["dof_V_0"], label='dof_V_0')
        if log["dof_V_1"]: plt.plot(time, log["dof_V_1"], label='dof_V_1')
        if log["dof_V_2"]: plt.plot(time, log["dof_V_2"], label='dof_V_2')
        if log["dof_V_3"]: plt.plot(time, log["dof_V_3"], label='dof_V_3')
        if log["dof_V_4"]: plt.plot(time, log["dof_V_4"], label='dof_V_4')
        if log["dof_V_5"]: plt.plot(time, log["dof_V_5"], label='dof_V_5')
        if log["dof_V_6"]: plt.plot(time, log["dof_V_6"], label='dof_V_6')
        if log["dof_V_7"]: plt.plot(time, log["dof_V_7"], label='dof_V_7')
        if log["dof_V_8"]: plt.plot(time, log["dof_V_8"], label='dof_V_8')
        if log["dof_V_9"]: plt.plot(time, log["dof_V_9"], label='dof_V_9')
        if log["dof_V_10"]: plt.plot(time, log["dof_V_10"], label='dof_V_10')
        if log["dof_V_11"]: plt.plot(time, log["dof_V_11"], label='dof_V_11')
        plt.legend()
        plt.show()


        if log["actions_0"]: plt.plot(time, log["actions_0"], label='actions_0')
        if log["actions_1"]: plt.plot(time, log["actions_1"], label='actions_1')
        if log["actions_2"]: plt.plot(time, log["actions_2"], label='actions_2')
        if log["actions_3"]: plt.plot(time, log["actions_3"], label='actions_3')
        if log["actions_4"]: plt.plot(time, log["actions_4"], label='actions_4')
        if log["actions_5"]: plt.plot(time, log["actions_5"], label='actions_5')
        if log["actions_6"]: plt.plot(time, log["actions_6"], label='actions_6')
        if log["actions_7"]: plt.plot(time, log["actions_7"], label='actions_7')
        if log["actions_8"]: plt.plot(time, log["actions_8"], label='actions_8')
        if log["actions_9"]: plt.plot(time, log["actions_9"], label='actions_9')
        if log["actions_10"]: plt.plot(time, log["actions_10"], label='actions_10')
        if log["actions_11"]: plt.plot(time, log["actions_11"], label='actions_11')
        plt.legend()
        plt.show()
                
        # if log["actions"]: plt.plot(time, log["actions"][0], label='actions0')
        # if log["actions"]: plt.plot(time, log["actions"][1], label='actions1')
        # if log["actions"]: plt.plot(time, log["actions"][2], label='actions2')
        # if log["actions"]: plt.plot(time, log["actions"][3], label='actions3')
        # if log["actions"]: plt.plot(time, log["actions"][4], label='actions4')
        # if log["actions"]: plt.plot(time, log["actions"][5], label='actions5')
        # if log["actions"]: plt.plot(time, log["actions"][6], label='actions6')
        # if log["actions"]: plt.plot(time, log["actions"][7], label='actions7')
        # if log["actions"]: plt.plot(time, log["actions"][8], label='actions8')
        # if log["actions"]: plt.plot(time, log["actions"][9], label='actions9')
        # if log["actions"]: plt.plot(time, log["actions"][10], label='actions10')
        # if log["actions"]: plt.plot(time, log["actions"][11], label='actions11')
        # plt.legend()
        # plt.show()
        
        # plot joint targets and measured positions
        # a = axs[1, 0]
        # if log["dof_pos"]: a.plot(time, log["dof_pos"], label='measured')
        # if log["dof_pos_target"]: a.plot(time, log["dof_pos_target"], label='target')
        # a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position')
        # a.legend()
        # # plot joint velocity
        # a = axs[1, 1]
        # if log["actionsel"]: a.plot(time, log["actionsel"], label='measured')
        # if log["actionsel_target"]: a.plot(time, log["actionsel_target"], label='target')
        # a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        # a.legend()
        # plot base vel x
        
        # a = axs[0, 0]
        # if log["base_vel_x"]: a.plot(time, log["base_vel_x"], label='measured')
        # if log["esti_lin_vel_x"]: a.plot(time, log["esti_lin_vel_x"], label='estimated')
        # # if log["vel_esti_f_x"]: a.plot(time, log["vel_esti_f_x"], label='filter')
        # # if log["i"]: a.plot(time, log["i"], label='i')
        # a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
        # a.legend()
        # # plot base vel y
        # a = axs[0, 1]
        # if log["base_vel_y"]: a.plot(time, log["base_vel_y"], label='measured')
        # # if log["vel_esti_f_y"]: a.plot(time, log["vel_esti_f_y"], label='filter')
        
        # if log["esti_lin_vel_y"]: a.plot(time, log["esti_lin_vel_y"], label='estimated')
        
        # a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
        # a.legend()
        

        # # plot base vel yaw
        # a = axs[0, 2]
        # if log["base_vel_yaw"]: a.plot(time, log["base_vel_yaw"], label='measured')
        # if log["command_yaw"]: a.plot(time, log["command_yaw"], label='commanded')
        # a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base velocity yaw')
        # a.legend()
        # plot base vel z
        
        # a = axs[1, 1]
        # if log["base_vel_z"]: a.plot(time, log["base_vel_z"], label='measured')
        # if log["esti_lin_vel_z"]: a.plot(time, log["esti_lin_vel_z"], label='estimated')
        # a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity z')
        # a.legend()
        
        # # plot contact forces
        # a = axs[2, 0]
        # if log["contact_forces_z"]:
        #     forces = np.array(log["contact_forces_z"])
        #     for i in range(forces.shape[1]):
        #         a.plot(time, forces[:, i], label=f'force {i}')
        # a.set(xlabel='time [s]', ylabel='Forces z [N]', title='Vertical Contact forces')
        # a.legend()
        # # plot torque/vel curves
        # a = axs[2, 1]
        # if log["actionsel"]!=[] and log["dof_torque"]!=[]: a.plot(log["actionsel"], log["dof_torque"], 'x', label='measured')
        # a.set(xlabel='Joint vel [rad/s]', ylabel='Joint Torque [Nm]', title='Torque/velocity curves')
        # a.legend()
        # # plot torques
        # a = axs[2, 2]
        # if log["dof_torque"]!=[]: a.plot(time, log["dof_torque"], label='measured')
        # a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        # a.legend()
        plt.show()

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()