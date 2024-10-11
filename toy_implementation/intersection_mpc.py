#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

import casadi as cs
import copy

#### pygame setup #####

screen_width = 4000

###### Agents #####

N = 3 #### number of agents

M = 2 #### number of futures

### unicycle model 

state_dim = 3
control_dim = 2

#### Sim time details ####

T = 15

dt = 0.5

clockspeed = int(1/dt)

S = int(T/dt)

control_horizon = 10

################ intersection set up ############

no_lanes = 4

lanes = np.arange(0,no_lanes)

lane_width = 500
lane_length = screen_width/2 - lane_width/2

upper = screen_width/2 + lane_width/2.6

lower = screen_width/2 - lane_width/2.5

car_width = lane_width/5
car_length = car_width*1.5


# In[2]:


############ reference trajectories ############

distance_increment = 200

s = np.arange(0,screen_width,distance_increment)


p0_ref = [np.array([lower, screen_width - s[j]-car_length,-3.14/2]) for j in range(len(s))] ## straight downward

p1_ref = [np.array([upper, s[j] ,3.14/2]) for j in range(len(s)) ]  ## straight upward

p2_ref = [np.array([s[j], lower,0.0]) for j in range(len(s))] ## straight leftward

p3_ref = [np.array([screen_width - s[j] - car_length,upper,-3.1415]) for j in range(len(s))] ## straight rightward

### start in p3, turn left to p1 ###

s2 = np.arange(0,screen_width/2 - lane_width/2,distance_increment)

p4_ref = [np.array([screen_width - s[j] - car_length,upper,-3.14]) for j in range(len(s2))]

v = len(s2+10)

angle_inc = np.arange(0,3.1415/2, 3.1415/(2*v))

curve = []

### start in p2, turn left p0 ###

p5_ref = []

for l in range(500):
    p0_ref.append(p0_ref[-1])
    p1_ref.append(p1_ref[-1])
    p2_ref.append(p2_ref[-1])
    p3_ref.append(p3_ref[-1])

##### unicycle dynamics ####

def find_closest_point(x,ref_path):
    x = np.reshape(x,(state_dim,1))
    dists = np.linalg.norm(ref_path[0:2,:] - x[0:2],axis=1)
    # closest_index = 

    return np.argmin(dists) + 2

class Vehicle():

    def __init__(self, index, lane):
        super().__init__()
        self.index = index
        self.lane = lane
        self.state_traj = np.zeros((state_dim, S))
        self.control_traj = np.zeros((control_dim,S))

        if self.lane == 0:
            self.ref_path = np.array(p0_ref).T
            self.lane_dir = 1
            
        elif self.lane == 1:
            self.ref_path = np.array(p1_ref).T
            self.lane_dir = 1
            
        elif self.lane == 2:
            self.ref_path = np.array(p2_ref).T            
            self.lane_dir = 0
            
        else:
            self.ref_path = np.array(p3_ref).T            
            self.lane_dir = 0
            
        self.state_traj[:,0] = self.ref_path[:,0]
        self.goal = self.ref_path[:,-1]
        
        self.predict_state_traj = np.zeros((M,state_dim,control_horizon))

        for k in range(M):
            self.predict_state_traj[k,:,:] = self.ref_path[:,1:control_horizon+1]

        self.predict_state_temp = np.array(self.ref_path[0:control_horizon])
        self.predict_control_temp = np.zeros((control_dim,control_horizon))

        self.other_vehs = []

#### add vehicles in scene #### syntax: Vehicle(index, path)

vehicle_indices = np.arange(0,N,1)
vehicles = []
vehicles.append(Vehicle(0,1))
vehicles.append(Vehicle(1,2))
vehicles.append(Vehicle(2,3))


#### add vehicle on same lane as ego 
# vehicles.append(Vehicle(1,2))

for i in range(N):

    ovs = []
    for j in range(N):
        if vehicles[j].index != vehicles[i].index:
            ovs.append(vehicles[j].index)
    vehicles[i].other_vehs = ovs



# In[3]:


######### dynamics ############
def xdot_dynamics(x,u):
    xdot1 = u[0]*cs.cos(x[2])
    xdot2 = u[0]*cs.sin(x[2])
    xdot3 = u[1]
    return cs.vertcat(xdot1,xdot2,xdot3)

######### start of MPC #########

import time

opt_time = 0

init_conf = 0.6
f1 = init_conf

for t in range(S-1):

    for i in range(N):        
        if i == 0:    ### count optimization time for only ego
            tic = time.time()
            
        veh = vehicles[i]

        other_vehs = veh.other_vehs
        
        opti = cs.Opti()
        
        X = opti.variable(state_dim,control_horizon)
        x = X[0,:]
        y = X[1,:]
        theta = X[2,:]
        U = opti.variable(control_dim,control_horizon)

        Xdot_square = 0
        
        #### slack variable for constraint violation ####3
        constraint_slack = opti.variable(M,control_horizon)     
        violation_weight = 1e8
        violation_cost = 0


        ### initial conditions 
        init_point = veh.state_traj[:,t]
        opti.subject_to(X[:,0] == init_point)
        
        U_bar = 200
        ##### dynamics constraint ########

        for k in range(1,control_horizon):
            dxdt = xdot_dynamics(X[:,k-1],U[:,k-1]) 
            
            comfort_weight = np.array([1e0,1e0,1e-2])
            
            weighted_dxdt = dxdt * comfort_weight
            
            Xdot_square += weighted_dxdt.T @ weighted_dxdt

            x_next = X[:,k-1] + dt * dxdt

            opti.subject_to(X[:,k] == x_next)
            opti.subject_to((U[0,k-1] - U[0,k])**2 <= U_bar)

            

        #### collision avoidance 
        f1 = 1/(1 + np.exp(-f1 *0.1 * (0.05*t+1)))
        # print(f1)
        confidence = [f1,1-f1]
        L = car_length*1.5 
            

        for k in range(control_horizon-1):
            if i == 0:
                for j in range(N-1): 
                    for m in range(M):
                        ov = vehicles[other_vehs[j]]
                        ov_path = copy.deepcopy(ov.predict_state_traj[m,:,:])

                        if ov_path[ov.lane_dir,1] <= screen_width/2: 
                            
                            ##### L1 distance constraint ######
                            
                            # dx = cs.fabs(X[0,k] - ov_path[0,k])
                            # dy = cs.fabs(X[1,k] - ov_path[1,k])
                            # opti.subject_to(dx + dy >= 2*L + constraint_slack[m,k])

                            ##### L2 distance #####
                            
                            dx = (X[0,k] - ov_path[0,k])**2
                            dy = (X[1,k] - ov_path[1,k])**2
                            opti.subject_to(cs.sqrt(dx + dy) >= 2*L + constraint_slack[m,k])                            
                            
                            opti.subject_to(constraint_slack[m,k]>= -1.9*L)
                            
                            barrier_cost = cs.log(1 + cs.exp( -confidence[m] * 2* (cs.sqrt(dx + dy) - 2.5*L + 1e-4 ))) 
                            
                            violation_cost += violation_weight * constraint_slack[m,k]**2 + barrier_cost * 1e2
    
            else:
                ov = vehicles[0]
                ov_path = copy.deepcopy(ov.predict_state_temp)
                
                if ov_path[ov.lane_dir,1] <= screen_width/2:
                    ##### L1 distance constraint ######
                    
                    # dx = cs.fabs(X[0,k] - ov_path[0,k+1])
                    # dy = cs.fabs(X[1,k] - ov_path[1,k+1])
                    # opti.subject_to(dx + dy >= 3*L + constraint_slack[m,k])

                    ##### L2 distance constraint ######  
                    
                    dx = (X[0,k] - ov_path[0,k+1])**2
                    dy = (X[1,k] - ov_path[1,k+1])**2
                    opti.subject_to(cs.sqrt(dx + dy) >= 3*L + constraint_slack[m,k])

                    
                    opti.subject_to(constraint_slack[m,k] >= -3*L)
        
                    barrier_cost = cs.log(1 + cs.exp( -1e-1* (dx + dy - 3.4*L + 1e-4 ))) 
                    
                    violation_cost += violation_weight * constraint_slack[m,k]**2 + barrier_cost * 1e4


        # ###### stop-sign constraint ####
        # closeness = 3 * L
        # stop_point = screen_width/2 - lane_width
        # for k in range(control_horizon): 
        #     stop = cs.if_else(cs.fabs(X[veh.lane_dir,k]) - )
        
        #### actuator constraints ####
        v_max = 300.0  # Maximum linear velocity
        v_min = -1e-8
        omega_max = 0.005    # Maximum angular velocity
        
        opti.subject_to(opti.bounded(v_min, U[0, :], v_max))
        opti.subject_to(opti.bounded(-omega_max, U[1, :], omega_max))
        # opti.subject_to(opti.bounded(screen_width/2, X[0, :], screen_width/2+lane_width/2))

        #### running cost for tracking #######

        ref_start = find_closest_point(init_point,veh.ref_path)

        ref_traj = np.array(veh.ref_path[:,t:t+control_horizon])

        Xdiff = cs.reshape(X - ref_traj,(state_dim) * control_horizon,1)
        
        Xdiff[2,:] = Xdiff[2,:] * 1e1
        
        running_cost = 1/control_horizon * Xdiff.T @ Xdiff 
        
        #### running cost for comfort cost ##########
        
        comfort_cost = Xdot_square /control_horizon 
                   
        ####  terminal cost ########
        
        goal_diff = X[0:2,-1] - np.array(veh.goal[0:2])
    
        terminal_cost = goal_diff.T @ goal_diff * t 

        ##### control cost ###########
        relative_u_weight = [0e1,1e2] ##### [weight on velocity, weight on angular velocity]
                    
        u_weight = np.array(relative_u_weight).reshape((control_dim,1)).repeat(control_horizon,axis = 1)
        
        # print(u_weight.shape)
        
        U_weighted = U * u_weight
         
        control_stacked = cs.reshape(U_weighted,control_dim * control_horizon,1)
        control_cost = control_stacked.T @ control_stacked

        
        ##### cost function set up #########
        
        W_state = 1e1
        W_control = 0e-2
        W_terminal = 1e3
        W_comfort = 1e3

        cost = W_state * running_cost + W_control * control_cost + W_terminal * terminal_cost + violation_cost + W_comfort * comfort_cost
        
        # opti.set_initial(U[1,:],3.1415/2)
        
        ###### solver set up #######
        
        opti.minimize(cost)
        
        opti.solver('ipopt', {
            'ipopt.print_level': 0,  # Suppresses the print level of IPOPT
            'print_time': False,     # Disables the printing of solver timing information
            'ipopt.sb': 'yes',
            'ipopt.max_iter':500,                # Suppresses the IPOPT banner
        })

        sol = opti.solve()
        X_sol = np.array(sol.value(X))
        U_sol = np.array(sol.value(U))

        if i == 0:
            toc = time.time()
            opt_time += toc - tic 


        
        # print('comfort cost = %.2f' % (sol.value(comfort_cost)))
        # print('terminal cost = %.2f' % (sol.value(terminal_cost)))        
        
        #### if solver has issues ######
        
        # try:
        #     sol = opti.solve() 
        #     X_sol = np.array(sol.value(X))
        #     U_sol = np.array(sol.value(U))
        
        
        # except RuntimeError as e:
        #     # print(str(e))
        #     if 'Maximum_Iterations_Exceeded' in str(e):
        #         print("MIE, using last solution for vehicle " + str(veh.index) + 'at time t = ' + str(t) )
        #         # Extract the last value of the solution from the optimizer variables
        #         X_sol = opti.debug.value(X)
        #         U_sol = opti.debug.value(U)                              

        #     else:
        #         print('solver fail for vehicle ' + str(veh.index) + 'at time t = ' + str(t))
        #         X_sol = veh.predict_state_temp
        #         U_sol = veh.predict_control_temp                


        veh.state_traj[:,t+1] = np.array(X_sol[:,1])
        veh.control_traj[:,t] = np.array(U_sol[:,0])
    
        veh.predict_state_temp = np.array(X_sol)
        veh.predict_control_temp = np.array(U_sol)

    for i in range(N):
        veh = vehicles[i]
        for m in range(M):
            # if veh.index == 1:
                mean = np.ones(1)
                covariance = np.ones(1)
                veh.predict_state_traj[m,:,:] = copy.deepcopy(veh.predict_state_temp)
                if m == 0:
                    # veh.predict_state_traj[m,:,:] = veh.predict_state_temp[0,:] + np.random.uniform(0,10,control_horizon)
                    veh.predict_state_traj[m,0,:] +=  np.random.randn(control_horizon)*0 - 1
                else:
                    # veh.predict_state_traj[m,:,:] = veh.predict_state_temp[0,:] - np.random.uniform(0,20,control_horizon)
                    veh.predict_state_traj[m,0,:] +=  -np.random.randn(control_horizon)*10 + 30

    

avg_opt_time = (opt_time)/(S-1)

print('optimization time  =  %f' % avg_opt_time)



# In[4]:


##### plot trajectory for ego #######

# plt.figure()
# time_ser = np.arange(0,S,1)
# # plt.plot(time_ser*dt,vehicles[0].control_traj[1,:],'ro-')
# # plt.plot(time_ser*dt,vehicles[0].control_traj[0,:],'bo-')
# plt.plot(time_ser*dt,vehicles[0].state_traj[1,:],'bo-')
# plt.show()


# In[5]:


######### matplotlib visualization #######

import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from PIL import Image
from IPython.display import Image as IPImage, display
import math

###### visualization params
screen_height = screen_width


def draw_intersection():

    ### draw roads ####
    straight_upward = patches.Rectangle((0,screen_width/2 - lane_width/2),screen_width,lane_width,linewidth = 0,edgecolor='r',facecolor= 'gray')
    straight_leftward = patches.Rectangle((screen_width/2 - lane_width/2,0),lane_width,screen_height,linewidth = 0,edgecolor='g',facecolor= 'gray')


    #### draw lane separation yellow lines ###
    plt.plot([0,screen_width/2 - lane_width/2],[screen_width/2,screen_width/2],linewidth = 1,color = 'yellow')
    plt.plot([screen_width/2 + lane_width/2,screen_width],[screen_width/2,screen_width/2],linewidth = 1,color = 'yellow')    

    plt.plot([screen_width/2,screen_width/2],[0,screen_width/2 - lane_width/2],linewidth = 1,color = 'yellow')
    plt.plot([screen_width/2,screen_width/2],[screen_width/2 + lane_width/2,screen_width],linewidth = 1,color = 'yellow')        
    
    ax.add_patch(straight_upward)
    ax.add_patch(straight_leftward)

def draw_vehicle(veh):
    if veh.index == 0:
        vehicle_color = "g" 
    else:
        vehicle_color = "r"

    vehicle_rect = patches.Rectangle((veh.state_traj[0,0],veh.state_traj[1,0]),car_length,car_width,angle = np.degrees(veh.state_traj[2,0]), linewidth = 1,edgecolor='k',facecolor= vehicle_color)

    return vehicle_rect



def frame_update(frame): 
    for i,rect in enumerate(vehicle_rectangles):
        rect.set_xy((vehicles[i].state_traj[0,frame],vehicles[i].state_traj[1,frame]))
        rect.angle = np.degrees(vehicles[i].state_traj[2,frame])
        
    return vehicle_rectangles


fig,ax = plt.subplots(figsize = (5,5))
draw_intersection()

vehicle_rectangles = []
for i in range(N):
    c = draw_vehicle(vehicles[i])
    ax.add_patch(c)
    vehicle_rectangles.append(c)
    
ax.set_xlim(0,screen_width)
ax.set_ylim(0,screen_width)        

ani = FuncAnimation(fig, frame_update, frames=S, blit=False)

ani.save('./vids/test.gif', writer='pillow', fps=3)

#### use the line below to create an mp4 video

# ani.save('./vids/test.mp4', writer='ffmpeg', fps=20)





# In[6]:


##### use this if using IPython

display(IPImage('./vids/test.gif'))


#### uncomment if running from terminal #####

# image = Image.open('./vids/rectangle_animation.gif')
# image.show()



##### uncomment if you want to save every frame ####

# def save_frame(*args):
#     plt.savefig(f'./vids/frame_{ani.frame_seq.index:03d}.png')

# ani.save_count = S-1

# # Connect the callback to save the frames after they are drawn
# ani._start()

# for i, frame in enumerate(ani.frame_seq):
#     save_frame(frame)



# In[ ]:




