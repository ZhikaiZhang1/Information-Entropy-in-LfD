from pypbd import TPGMM_Time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from math import sqrt
from dataclasses import dataclass
import pandas as pd

# This file is largely the same as path.py and uses several of its functions, but this is more for testing purposes and experimenting
# with visualizations and algorithms.

@dataclass
class Params:
    goal_x = 0.3
    goal_y = 0.4
    demos_to_use = 4
    increment = 0.03
    end_goal_radius = 0.006
    alpha = 0
    beta = 1
    gamma = 0.01
    delta = 1
    x_target = -0.35
    y_target = 0.475
    directory_name = 'processed_path_demos'

def retrieve_demos(number_of_demos): #obtain demos
    if number_of_demos <= 0: #ensure number of demos is greater than 0
        print("Must be greater than 0")
        return

    data = np.expand_dims(np.load(settings.directory_name+'/demo' + '1' + '.npy'), axis=2) #examine demo 1
    for i in range(2, number_of_demos+1): #examine other demo
        demo = np.expand_dims(np.load(settings.directory_name+'/demo' + str(i) + '.npy'), axis=2) #expand dimensions
        data = np.concatenate((data, demo), axis = 2) #add new demo to existing demo array
        i += 1 #iterate
    print("SHAPE",data.shape)
    print(data[1,:,:])
    return data #return full demo array

def plot_demos(number_of_demos):  #visually plot demo entries
    if number_of_demos <= 0: #ensure number of demos is greater than 0
        print("Must be greater than 0")
        return

    for i in range(1, number_of_demos+1): 
        demo = np.expand_dims(np.load(settings.directory_name+'/demo' + str(i) + '.npy'), axis=2) #load demo
        plt.plot(demo[:,1,0], demo[:,2,0], color = 'blue', linewidth = 0.7) #plot blue line illustrating path
        plt.scatter(demo[:,1,0], demo[:,2,0],zorder=100,s=5) # plot all the points along the trajectory
        print("demo " + str(i) + ": ", demo[0,1,0], demo[0,2,0]) #print out starting point
        i += 1

def get_frame_dtype(dim): #get frame data type, from TPGMM_time.py
    return np.dtype([("A", "f8", (dim, dim)), ("b", "f8", (dim,))]) #A is transformation matrix, b is offset vector

def get_frames(data, dims): #generate frames of observation
    frames = np.array([([[1,0],[0,1]],[data[0,1,0],data[0,2,0]]), #start point
                     ([[1,0],[0,1]],[data[99,1,0],data[99,2,0]]), #end point
                     ([[1,0],[0,1]],[-0.12,0.19]), ([[1,0],[0,1]],[0.12,0.34])] #obstacles
                     , dtype=get_frame_dtype(dims))
    frames = np.expand_dims(frames, axis = 1)

    for i in range(1, np.size(data, axis = 2)):
        frame_demo = np.array([([[1,0],[0,1]],[data[0,1,i],data[0,2,i]]), #start point
                     ([[1,0],[0,1]],[data[99,1,i],data[99,2,i]]), #end point
                     ([[1,0],[0,1]],[-0.12,0.19]), ([[1,0],[0,1]],[0.12,0.34])] #obstacles
                     , dtype=get_frame_dtype(dims))
        
        frame_demo = np.expand_dims(frame_demo, axis = 1)
        frames = np.concatenate((frames, frame_demo), axis = 1)
        i += 1

    return(frames)

def generate_trajectory(x_init, y_init, dims): #generates a new trajectory starting from x_init and y_init
    #generate the new frame (only modifying start point observer)
    new_frame = np.array([([[1,0],[0,1]],[x_init,y_init]), #start point
                     ([[1,0],[0,1]],[data[99,1,0],data[99,2,0]]), #end point
                     ([[1,0],[0,1]],[-0.12,0.19]), ([[1,0],[0,1]],[0.12,0.34])] #obstacles
                     , dtype=get_frame_dtype(dims))
    
    return tpgmm.reproduce(new_frame,np.linspace(0,10.0,num=100)) #uses TPGMM_time to reproduce trajectory

def add_rectangles(): #add visual obstacles on plot
    rect_w, rect_y = 0.12, 0.30
    rect_1 = patches.Rectangle((-0.13,0.04),rect_w,rect_y, fill = True, color = '0.1')
    rect_2 = patches.Rectangle((0.11,0.19),rect_w,rect_y, fill = True, color = '0.1')
    rect_3 = patches.Rectangle((-0.36,0.04),0.145,0.45, fill = True, color = '0.92')
    rect_4 = patches.Rectangle((-0.36,0.04),0.72,0.45, fill = False, color = '0.1', ls = '--', linewidth = 0.5)

    fig, ax = plt.subplots()
    ax.add_patch(rect_1), ax.add_patch(rect_2), ax.add_patch(rect_3), ax.add_patch(rect_4)

def trajectory_grid(increment, end_goal_radius): #generate grid of trajectories
    print("generating trajectories ...")
    x_points = np.linspace(-0.36+increment/2, -0.215-increment/2, round(((0.36-increment/2)-(0.215+increment/2))/increment)) #x-grid generated
    y_points = np.linspace(0.04+increment/2, 0.49-increment/2, round(((0.49-increment/2)-(0.04+increment/2))/increment)) #y-grid generated
    print("total:",len(x_points)*len(y_points))
    point_dict = {} #Structure: (x,y):[valid?,start_dis,end_dis,collision_states,drift,uncertainty]
    for x in x_points: #nested for loop, to cover all the grid
        for y in y_points:
            new_path = generate_trajectory(x_init = x, y_init = y, dims = 2) #reference generate_trajectory function
            col = "green" #assume trajectory is green = valid

            start_dis,end_dis = start_end_dis(new_path,x,y)
            if start_dis > increment/2 or end_dis > end_goal_radius: #if generated start point is outside radius, it is invalid
                col = "red"
            
            collision_states = 0
            drift = [0,0] #x_drift, y_drift
            for i in range(0,100):
                if (not(-0.36 <= new_path['x'][0,0,i] <= 0.36)) or (not(0.04 <= new_path['x'][0,1,i] <= 0.49)): #not in bounds
                    col = "red"
                elif (-0.13 <= new_path['x'][0,0,i] <= -0.01) and (0.04 <= new_path['x'][0,1,i] <= 0.34): #not in first rectangle
                    col = "red"
                    collision_states += 1
                elif (0.11 <= new_path['x'][0,0,i] <= 0.23) and (0.19 <= new_path['x'][0,1,i] <= 0.49): #not in second rectangle
                    col = "red"
                    collision_states += 1

                #Calculate drift score
                if i>=1:
                    dx = (settings.goal_x-new_path['x'][0,0,i]) - (settings.goal_x-new_path['x'][0,0,i-1])
                    dy = (settings.goal_y-new_path['x'][0,1,i]) - (settings.goal_y-new_path['x'][0,1,i-1])
                    if dx > 0: # this condition means the trajectory is straying away from end point in x
                        drift[0] += dx
                    if dy > 0: # this condition means the trajectory is straying away from end point in y
                        drift[1] += dy
            
            uncertainty = settings.alpha*start_dis+settings.beta*end_dis+settings.gamma*collision_states+settings.delta*drift[0]
            point_dict[(x,y)]=[col,start_dis,end_dis,collision_states,drift,uncertainty]
            plt.plot(new_path['x'][0,0,:], new_path['x'][0,1,:], color = col, linewidth = 0.3) #plot generated trajectory
            plt.plot(x,y, marker="+", color=col, markersize=5, linewidth = 0.3) #visual marker
            plt.gca().add_artist(plt.Circle((x, y), increment/2, color = col, fill = False, linewidth = 0.3)) #visual marker of acceptable radius

    sorted_dict = {k: v for k, v in sorted(point_dict.items(), key=lambda item: item[1][5])} # Sort dictionary by increasing uncertainty
    max_uncertainty_pt = list(sorted_dict.keys())[-1]

    print("Point with highest uncertainty:",max_uncertainty_pt,"Uncertainty:",sorted_dict[list(sorted_dict.keys())[-1]])

    plt.plot(max_uncertainty_pt[0],max_uncertainty_pt[1],marker=".",color="blue",markersize=5,linewidth=0.3) #highlight max uncertainty point
    #plt.plot(list(sorted_dict.keys())[-2][0],list(sorted_dict.keys())[-2][1],marker=".",color="orange",markersize=5,linewidth=0.3) #highlight max uncertainty point

    plt.gca().add_artist(plt.Circle((settings.goal_x, settings.goal_y), end_goal_radius, color = "0.1", fill = False, linewidth = 0.3)) #visual marker of acceptable radius

    return (x_points,y_points,point_dict,max_uncertainty_pt)

def start_end_dis(new_path,x,y):
    start_dis = sqrt((x-new_path['x'][0,0,0])**2+(y-new_path['x'][0,1,0])**2) #pythagoras to determine distance away from intended start
    end_dis = sqrt((settings.goal_x-new_path['x'][0,0,-1])**2+(settings.goal_y-new_path['x'][0,1,-1])**2) #pythag to determine distance away from goal
    return (start_dis,end_dis)

def generate_dataframe(x_points,y_points,point_dict,max_u_pt):
    #initialize an empty dataframe
    init_df = np.zeros((len(y_points),len(x_points)))

    df = pd.DataFrame(data=init_df,index=y_points,columns=x_points)
    for y in y_points:
        for x in x_points:
            df.at[y,x]=point_dict[(x,y)][5]#/point_dict[max_u_pt][5]
    return df
def p_controller(x_target,y_target):
    # this is the point you want the controller to converge on
    x = x_target
    y = y_target
    
    # weights for the x and y portion of the controller, in a different implementation I made these the same 
    k_px = 1
    k_py = 1

    # stuff for d controller that never really worked out because x and y aren't independent... weird.
    k_d = 0
    x_diff = 10000 # initialize x_diff and y_diff to be large values
    y_diff = 10000 
    x_array = []
    y_array = []
    x_error_array = []
    y_error_array = []
    x_dedt = 0
    y_dedt = 0
    
    while abs(x_diff) > 0.0001 and abs(y_diff) > 0.0001:
    #for i in range(10):
        new_path = generate_trajectory(x_init = x, y_init = y, dims = 2)
        # Uncomment this if you want to see every path that is generated as opposed to just the final one
        #plt.plot(new_path['x'][0,0,:], new_path['x'][0,1,:], color = "red", linewidth = 0.3,label="hi") #plot generated trajectory

        x_array.append(new_path['x'][0,0,0])
        y_array.append(new_path['x'][0,1,0])

        x_diff = x_target-new_path['x'][0,0,0]
        y_diff = y_target-new_path['x'][0,1,0]

        x_error_array.append(x_diff)
        y_error_array.append(y_diff)
        
        if len(x_error_array) > 1:
            x_dedt = x_diff - x_error_array[-2]
            y_dedt = y_diff - y_error_array[-2]

        x += x_diff*k_px+x_dedt*k_d
        y += y_diff*k_py+y_dedt*k_d

        print("x,y",new_path['x'][0,0,0],new_path['x'][0,1,0])
        print("diff",x_diff,y_diff)
        print("control signal",x_diff*k_px,y_diff*k_py)
    '''
    # different p controller implementation - more of a vector based approach

    k_p = 0.5
    #while abs(x_diff) > 0.0001 and abs(y_diff) > 0.0001:
    for i in range(10):
        new_path = generate_trajectory(x_init=x,y_init=y,dims=2)
        # Uncomment this if you want to see every path that is generated as opposed to just the final one
        plt.plot(new_path['x'][0,0,:],new_path['x'][0,1,:],color="red",linewidth=0.3)

        x_array.append(new_path['x'][0,0,0])
        y_array.append(new_path['x'][0,1,0])

        x_diff=x_target-new_path['x'][0,0,0]
        y_diff=y_target-new_path['x'][0,1,0]
        diff_vector = [x_diff,y_diff]
        theta = math.atan(y_diff/x_diff)
        mag = math.sqrt(x_diff**2+y_diff**2)
        x+=mag*k_p*math.cos(theta)
        y+=mag*k_p*math.sin(theta)
        print("x,y",new_path['x'][0,0,0],new_path['x'][0,1,0])
        print("diff",x_diff,y_diff)
        print("control signal",x_diff*k_px,y_diff*k_py)
    '''
    plt.plot(new_path['x'][0,0,:],new_path['x'][0,1,:],color="red",linewidth=0.3)    
    plt.plot(settings.x_target,settings.y_target, marker="+", color="red", markersize=5, linewidth = 0.3) #visual marker
    plt.gca().add_artist(plt.Circle((settings.x_target, settings.y_target), settings.increment/2, color = "red", fill = False, linewidth = 0.3)) #visual marker of acceptable radius
    return (x_array,y_array,new_path)
# --- MAIN --- #
add_rectangles() 

settings = Params()

tpgmm = TPGMM_Time(10, max_steps = 200) #initialize TPGMM_Time
data = retrieve_demos(settings.demos_to_use) #obtain demos
frames = get_frames(data, dims = 2) #generate frames
tpgmm.fit(data,frames) #fit data to frames

plot1 = plt.figure(1)

#x_points, y_points, point_dict, max_u_pt = trajectory_grid(settings.increment, settings.end_goal_radius) #generate grid of trajectories

x_array,y_array,p=p_controller(settings.x_target,settings.y_target)

plot_demos(settings.demos_to_use) #visually plot demos extracted

#plot the graphs to show controller behaviour
t = [i for i in range(len(x_array))]
x_line = [settings.x_target for i in x_array]
y_line = [settings.y_target for i in y_array]

variance_x = 0
variance_y = 0

# finds a trajectory from the start point of a user demo, third index = demo #
# a,b not needed
a,b,traj_var=p_controller(data[0,1,0],data[0,2,0])

plt.scatter(traj_var['x'][0,0], traj_var['x'][0,1],zorder=100,s=5)

# Calculate the variance score of the trajectory
for i in range(len(traj_var['x'][0,0])):
    traj_demo_x = [traj_var['x'][0,0,i],data[i,1,0]]
    traj_demo_y = [traj_var['x'][0,1,i],data[i,2,0]]
    variance_x += abs(traj_demo_x[1]-traj_demo_x[0])
    variance_y += abs(traj_demo_y[1]-traj_demo_y[0])
    plt.plot(traj_demo_x,traj_demo_y,linewidth=0.5)

print("Variance in x:", variance_x)
print("Variance in y:", variance_y)

plt.axis('scaled')

plot2 = plt.figure(2)
plt.plot(t,x_array)
plt.plot(t,x_line)
plot3 = plt.figure(3)
plt.plot(t,y_array)
plt.plot(t,y_line)
plt.show() #ensure 1:1 scaling of the plot, and illustrate it