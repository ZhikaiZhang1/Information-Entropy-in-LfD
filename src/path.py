from pypbd import TPGMM_Time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from math import sqrt, log
from dataclasses import dataclass
import seaborn as sns
import pandas as pd
import time
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot, transforms
from math import sqrt, log
from dataclasses import dataclass
import seaborn as sns
import pandas as pd
from random import randrange
import time
import os
import re
from matplotlib.patches import Ellipse

@dataclass
class Params:
    vis_mode = 2 #2: information entropy, 1: 4cm_rule
    inframe_display = 0
    show_data_pnts = True
    demos_to_use = 5
    increment = 0.03
    end_goal_radius = 0.015
    alpha = 0  # now that p-controller has been implemented, start_dis is essentially a useless metric
    beta = 0.3 #endpoint
    gamma = 0.4 #collision
    delta = 0.0 #drift
    epsilon = 0.3 #out of bound
    pd_accuracy = 0.01
    k_p = 1
    #K_I = [0.001, 0.004]
    #KI_accumulation = [0, 0]
    K_D = [0.1, 0.66]
    KD_prev = [0, 0]
    error_traj = []
    directory_name = '../processed_path_demos'
    starting_index = 0
    reg = 1e-5
    max_gmm = 30
    shift = [-0.225, -0.05]
    goal_x = 0.3 - 0.225
    goal_y = 0.4 - 0.05
    obstacle_workspace_ratio = 1#((0.015+0.12)*2 * 0.3*2)/(0.72*0.45)
    Data = None
    red = 0
    demo_name = "demo"


def retrieve_demos(number_of_demos):  # obtain demos
    if number_of_demos <= 0:  # ensure number of demos is greater than 0
        print("Must be greater than 0")
        return

    data = np.expand_dims(np.load(settings.directory_name + '/' + settings.demo_name + str(settings.starting_index) + '.npy'),
                          axis=2)  # examine demo 1
    for i in range(settings.starting_index + 1, settings.starting_index + number_of_demos):  # examine other demo
        demo = np.expand_dims(np.load(settings.directory_name + '/' + settings.demo_name + str(i) + '.npy'), axis=2)  # expand dimensions
        data = np.concatenate((data, demo), axis=2)  # add new demo to existing demo array
        i += 1  # iterate
    print("SHAPE", data.shape)
    print("axis 2 size is: ", np.size(data, axis=2))
    print("data is: ", data)
    return data  # return full demo array

def plot_ellipsoids(Mu, Sigma, ax):
    #fig, ax = plt.subplots()
    average_starting = np.array([np.mean(settings.Data[0,1,:]), np.mean(settings.Data[0,2,:])])
    for frame in range(Sigma.shape[2]):
        #fig, ax = plt.subplots()
        for component in range(Sigma.shape[3]):
            w,v = np.linalg.eig(Sigma[1:3,1:3,frame, component])
            u = np.array([v[-2,-2],v[-2,-1]]) / np.linalg.norm(np.array([v[-2,-2],v[-2,-1]]))
            #v[1]/
            centre = Mu[1:3,frame, component]
            #print("eigen vector is: ", v, "centre is: ", centre, "eigen values are: ", w, "num of frame is: ", Sigma.shape[2])
            #print("shape of covariance is: ", Sigma[:,:,frame, component])
            #angle_eig = angle_between(np.array([v[1,1],v[1,2]]), np.array([1,0]))
            angle_eig = np.arctan(u[1] / u[0])
            color = "blue"

            if frame == 0:
                color = "blue"
                centre = centre + average_starting# = centre - np.array([0,centre[1]]) + average_starting
                '''
                ellipse = Ellipse(xy=(centre[-2], centre[-1]), width=np.sqrt(abs(w[-2])), height=np.sqrt(abs(w[-1])),
                                  angle=180 + np.rad2deg(angle_eig),
                                  edgecolor=color, fc='None', lw=2)
                ax.add_patch(ellipse)
                '''
            elif frame == 1:
                color = "black"
                #centre = centre - np.array([-0.075, -0.35])# + average_starting
                '''
                ellipse = Ellipse(xy=(centre[-2], centre[-1]), width=np.sqrt(abs(w[-2])), height=np.sqrt(abs(w[-1])),
                                  angle=180 + np.rad2deg(angle_eig),
                                  edgecolor=color, fc='None', lw=2)
                ax.add_patch(ellipse)
                '''
            elif frame == 2:
                color = "yellow"
                centre = centre  - np.array([0.345, -0.14])#+ average_starting
                '''
                ellipse = Ellipse(xy=(centre[-2], centre[-1]), width=np.sqrt(abs(w[-2])), height=np.sqrt(abs(w[-1])),
                                  angle=180 + np.rad2deg(angle_eig),
                                  edgecolor=color, fc='None', lw=2)
                ax.add_patch(ellipse)
                '''
            elif frame == 3:
                color = "green"
                centre = centre - np.array([0.105, -0.29]) #+ average_starting
                '''
                ellipse = Ellipse(xy=(centre[-2], centre[-1]), width=np.sqrt(abs(w[-2])), height=np.sqrt(abs(w[-1])),
                                  angle=180 + np.rad2deg(angle_eig),
                                  edgecolor=color, fc='None', lw=2)
                ax.add_patch(ellipse)
                '''

            ellipse = Ellipse(xy=(centre[-2], centre[-1]), width=np.sqrt(abs(w[-2])), height=np.sqrt(abs(w[-1])),
                              angle=180 + np.rad2deg(angle_eig),
                              edgecolor=color, fc='None', lw=2)
            ax.add_patch(ellipse)


            #plt.axis('scaled')


def plot_demos(number_of_demos, Mu, Sigma):  # visually plot demo entries
    if number_of_demos <= 0:  # ensure number of demos is greater than 0
        print("Must be greater than 0")
        return
    average_starting = np.array([np.mean(settings.Data[0, 1, :]), np.mean(settings.Data[0, 2, :])])
    for i in range(settings.starting_index, settings.starting_index + number_of_demos):
        demo = np.expand_dims(np.load(settings.directory_name + '/' + settings.demo_name + str(i) + '.npy'), axis=2)  # load demo
        if (settings.inframe_display == 1):
            frame_shift = [-demo[0, 1, 0]+average_starting[0], -demo[0, 2, 0]+average_starting[1]]
        else:
            frame_shift = [0,0]
        plt.plot(demo[:, 1, 0] + frame_shift[0], demo[:, 2, 0]+frame_shift[1], color='blue', linewidth=0.7)  # plot blue line illustrating path
        if (settings.show_data_pnts):
            plt.plot(demo[:, 1, 0] + frame_shift[0], demo[:, 2, 0] + frame_shift[1], 'x', color='orange', linewidth=0.7)
        # plt.scatter(demo[:,1,0], demo[:,2,0],zorder=100,s=5) #visualize the points in each trajectory
        coordinate = get_first_point(x_points, y_points, [demo[0, 1, 0], demo[0, 2, 0]])
        print("demo " + str(i) + ": ", coordinate[1]*4+coordinate[0])  # print out starting point
        i += 1
    #plot_ellipsoids(Mu, Sigma)


def get_frame_dtype(dim):  # get frame data type, from TPGMM_time.py
    return np.dtype([("A", "f8", (dim, dim)), ("b", "f8", (dim,))])  # A is transformation matrix, b is offset vector


def get_frames(data, dims):  # generate frames of observation
    frames = np.array([([[1, 0], [0, 1]], [data[0, 1, 0], data[0, 2, 0]]),  # start point
                       ([[1, 0], [0, 1]], [data[99, 1, 0], data[99, 2, 0]]),  # end point
                       #([[1, 0], [0, 1]], [-0.12 + settings.shift[0], 0.19 + settings.shift[1]]),
                       #([[1, 0], [0, 1]], [data[99, 1, 0], data[99, 2, 0]])
                       ]  # obstacles
                      , dtype=get_frame_dtype(dims))
    frames = np.expand_dims(frames, axis=1)
    #print("frame 0 rotation matrix is now: ", frames[0, :]["A"], "frame 0 translation matrix is now: ", frames[0, :]["b"])
    #print("frame 1 rotation matrix is now: ", frames[1, :]["A"], "frame 1 translation matrix is now: ",
    #      frames[1, :]["b"])
    frames[1, :]["b"] = np.zeros(2) # edit frames


    for i in range(1, np.size(data, axis=2)):
        frame_demo = np.array([([[1, 0], [0, 1]], [data[0, 1, i], data[0, 2, i]]),  # start point
                               ([[1, 0], [0, 1]], [data[99, 1, i], data[99, 2, i]]),  # end point
                               #([[1, 0], [0, 1]], [-0.12 + settings.shift[0], 0.19 + settings.shift[1]]),
                               #([[1, 0], [0, 1]], [data[99, 1, 0], data[99, 2, 0]])
                               ]  # obstacles
                              , dtype=get_frame_dtype(dims))

        frame_demo = np.expand_dims(frame_demo, axis=1)
        frame_demo[1, :]["b"] = np.zeros(2)  # edit frames
        frames = np.concatenate((frames, frame_demo), axis=1)
        i += 1
    print("frames is now: ", frames)
    return (frames)


def generate_trajectory(x_init, y_init, dims):  # generates a new trajectory starting from x_init and y_init
    # generate the new frame (only modifying start point observer)
    new_frame = np.array([([[1, 0], [0, 1]], [x_init, y_init]),  # start point
                          ([[1, 0], [0, 1]], [data[99, 1, 0], data[99, 2, 0]]),  # end point
                          #([[1, 0], [0, 1]], [-0.12 + settings.shift[0], 0.19 + settings.shift[1]]),
                          #([[1, 0], [0, 1]], [data[99, 1, 0], data[99, 2, 0]])
                          ]  # obstacles
                         , dtype=get_frame_dtype(dims))

    new_frame["b"][1] = np.zeros(2)  # edit frames
    #print("frames is now: ", new_frame)
    #print("frames B is now: ", new_frame["b"])
    return tpgmm.reproduce(new_frame, np.linspace(0, 10.0, num=100))  # uses TPGMM_time to reproduce trajectory


def add_rectangles():  # add visual obstacles on plot
    rect_w, rect_y = 0.12, 0.30
    rect_1 = patches.Rectangle((-0.13+ settings.shift[0], 0.04+ settings.shift[1]), rect_w, rect_y, fill=True, color='0.1')
    rect_2 = patches.Rectangle((0.11+ settings.shift[0], 0.19+ settings.shift[1]), rect_w, rect_y, fill=True, color='0.1')
    rect_3 = patches.Rectangle((-0.36+ settings.shift[0], 0.04+ settings.shift[1]), 0.145, 0.45, fill=True, color='0.92')
    rect_4 = patches.Rectangle((-0.36+ settings.shift[0], 0.04+ settings.shift[1]), 0.72, 0.45, fill=False, color='0.1', ls='--', linewidth=0.5)
    fig, ax = plt.subplots()
    #adding boundaries
    bounds = [[[-0.1375+ settings.shift[0], 0.04+ settings.shift[1]],[-0.1375+ settings.shift[0], 0.34+ settings.shift[1]]],[[-0.13+ settings.shift[0], 0.3475+ settings.shift[1]],[-0.01+ settings.shift[0], 0.3475+ settings.shift[1]]], [[-0.0025+ settings.shift[0], 0.04+ settings.shift[1]],[-0.0025+ settings.shift[0], 0.34+ settings.shift[1]]],[[0.1025+ settings.shift[0], 0.49+ settings.shift[1]],[0.1025+ settings.shift[0], 0.19+ settings.shift[1]]], [[0.11+ settings.shift[0], 0.1825+ settings.shift[1]],[0.23+ settings.shift[0], 0.1825+ settings.shift[1]]],[[0.2375+ settings.shift[0], 0.19+ settings.shift[1]],[0.2375+ settings.shift[0], 0.49+ settings.shift[1]]]]
    for i in bounds:
        ax.plot([i[0][0], i[1][0]], [i[0][1], i[1][1]], '--',color='0.1', linewidth = 1)
    #rect_5 = patches.Rectangle((-0.1375, 0.04), rect_w+0.015, rect_y+0.0075, fill=False, color='0.1', ls='--', linewidth=0.5)
    #rect_6 = patches.Rectangle((0.1025, 0.1825), rect_w+0.015, rect_y+0.0075, fill=False, color='0.1', ls='--', linewidth=0.5)
    #adding arcs
    eclipse_centre = [[-0.13+ settings.shift[0], 0.34+ settings.shift[1]],[-0.01+ settings.shift[0], 0.34+ settings.shift[1]],[0.11+ settings.shift[0], 0.19+ settings.shift[1]],[0.23+ settings.shift[0], 0.19+ settings.shift[1]]]
    arc1 = patches.Arc((eclipse_centre[0][0], eclipse_centre[0][1]), 0.015, 0.015, angle=90, theta1 = 0, theta2 = 90, fill=False, color='0.1', ls='--')
    arc2 = patches.Arc((eclipse_centre[1][0], eclipse_centre[1][1]), 0.015, 0.015, angle=0, theta1 = 0, theta2 = 90,  fill=False, color='0.1', ls='--')
    arc3 = patches.Arc((eclipse_centre[2][0], eclipse_centre[2][1]), 0.015, 0.015, angle=180, theta1 = 0, theta2 = 90, fill=False, color='0.1', ls='--')
    arc4 = patches.Arc((eclipse_centre[3][0], eclipse_centre[3][1]), 0.015, 0.015, angle=-90, theta1 = 0, theta2 = 90, fill=False, color='0.1', ls='--')

    ax.add_patch(rect_1), ax.add_patch(rect_2), ax.add_patch(rect_3), ax.add_patch(rect_4)#, ax.add_patch(rect_5), ax.add_patch(rect_6)
    ax.add_patch(arc1), ax.add_patch(arc2), ax.add_patch(arc3), ax.add_patch(arc4)
    return ax

def point_distance(pnt1, pnt2):
    return np.linalg.norm(pnt1 - pnt2)

def get_instant_jerk(s, s0, v0, a0, t):
    #s = s0 + v0*t + 1/2*a0*t^2 + 1/6*j*t^3
    v = (s - s0) / t
    a = (v-v0)/t
    jerk = 6*((s-s0)+v0*t+1/2*a0*t*t)/(t*t*t)

    return jerk, v, a
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    vect_cross = np.cross(v1_u, v2_u)
    if vect_cross<0:
        return angle
    else:
        return -angle
def is_in_sector(point, base_vect, centre, radius):
    pnt_vector = point - centre
    magnitude = np.linalg.norm(pnt_vector)
    angle = angle_between(pnt_vector, base_vect)
    if magnitude < radius and  angle< 90 and angle > 0:
        return True
    else:
        return False

def trajectory_grid(increment, end_goal_radius):  # generate grid of trajectories

    print("generating trajectories ...")
    x_points = np.linspace(-0.36+ settings.shift[0] + increment / 2, -0.215+ settings.shift[0] - increment / 2,
                           round(((0.36+ settings.shift[0] - increment / 2) - (0.215+ settings.shift[0] + increment / 2)) / increment))  # x-grid generated
    y_points = np.linspace(0.04+ settings.shift[1] + increment / 2, 0.49+ settings.shift[1] - increment / 2,
                           round(((0.49+ settings.shift[1] - increment / 2) - (0.04+ settings.shift[1] + increment / 2)) / increment))  # y-grid generated
    print("total:", len(x_points) * len(y_points))
    point_dict = {}  # Structure: (x,y):[valid?,start_dis,end_dis,collision_states,drift,out_of_bound_states,uncertainty]
    jerk = [0,0] #changed need to be merged
    pnt_dis = 0
    v0 = [0,0]
    a0 = [0,0]
    temp_path = None
    for x in x_points:  # nested for loop, to cover all the grid
        for y in y_points:

            if settings.demos_to_use > 0:
                new_path, MuGMR, SigmaGMR = p_controller(x, y)
            else:
                new_path, MuGMR, SigmaGMR = generate_trajectory(x_init = x, y_init = y, dims = 2)


            #new_path, MuGMR, SigmaGMR = generate_trajectory(x_init = x, y_init = y, dims = 2)
            #generate_trajectory(x_init = x, y_init = y, dims = 2)

            col = "green"  # assume trajectory is green = valid

            start_dis, end_dis = start_end_dis(new_path, x, y)
            if start_dis > increment / 2 or end_dis > end_goal_radius:  # if generated start point is outside radius, it is invalid
                col = "red"

            collision_states = 0
            out_of_bound_states = 0
            drift = [0, 0]  # x_drift, y_drift
            arc_centre = [[-0.13+ settings.shift[0], 0.34+ settings.shift[1]], [-0.01+ settings.shift[0], 0.34+ settings.shift[1]], [0.11+ settings.shift[0], 0.19+ settings.shift[1]], [0.23+ settings.shift[0], 0.19+ settings.shift[1]]]

            for i in range(0, 100): # changed, need to be merged
                x_val = new_path['x'][0, 0, i]
                y_val = new_path['x'][0, 1, i]
                if i >= 1:
                    jerk_instantx, v0[0], a0[0] = get_instant_jerk(x_val, new_path['x'][0, 0, i - 1], v0[0], a0[0], 10 / 100)
                    jerk_instanty, v0[1], a0[1] = get_instant_jerk(y_val, new_path['x'][0, 1, i - 1], v0[1], a0[1], 10 / 100)
                    jerk[0] += jerk_instantx
                    jerk[1] += jerk_instanty
                    pnt_dis += point_distance(np.array([x_val,y_val]), np.array([new_path['x'][0, 0, i - 1], new_path['x'][0, 1, i - 1]]))

                if (not (-0.36+ settings.shift[0] <= x_val <= 0.36+ settings.shift[0])) or (
                not (0.04+ settings.shift[1] <= y_val <= 0.49+ settings.shift[1])):  # not in bounds
                    col = "red"
                    out_of_bound_states += 1
                elif ((-0.1375+ settings.shift[0] <= x_val <= -0.0025+ settings.shift[0]) and (
                        0.04+ settings.shift[1] <= y_val <= 0.34+ settings.shift[1])) or  ((-0.13+ settings.shift[0] <= x_val <= -0.01+ settings.shift[0]) and (
                        0.04+ settings.shift[1] <= y_val <= 0.3475+ settings.shift[1])):  # not in first rectangle and parts of the padding
                    col = "red"
                    collision_states += 1
                elif ((0.1025+ settings.shift[0] <= x_val <= 0.2375+ settings.shift[0]) and (
                        0.19+ settings.shift[1] <= y_val <= 0.49+ settings.shift[1])) or ((0.11+ settings.shift[0] <= x_val <= 0.23+ settings.shift[0]) and (
                        0.1825+ settings.shift[1] <= y_val <= 0.49+ settings.shift[1])):  # not in second rectangle and parts of the padding
                    col = "red"
                    collision_states += 1
                elif is_in_sector(np.array([x_val, y_val]), np.array([0,1]), np.array(arc_centre[0]), 0.0075) or is_in_sector(np.array([x_val, y_val]), np.array([1,0]), np.array(arc_centre[1]), 0.0075) or is_in_sector(np.array([x_val, y_val]), np.array([-1,0]), np.array(arc_centre[2]), 0.0075) or is_in_sector(np.array([x_val, y_val]), np.array([0,-1]), np.array(arc_centre[3]), 0.0075):
                    col = "red"
                    collision_states += 1

                # Calculate drift score
                if i >= 1:
                    dx = (settings.goal_x - new_path['x'][0, 0, i]) - (settings.goal_x - new_path['x'][0, 0, i - 1])
                    dy = (settings.goal_y - new_path['x'][0, 1, i]) - (settings.goal_y - new_path['x'][0, 1, i - 1])
                    if dx > 0:  # this condition means the trajectory is straying away from end point in x
                        drift[0] += dx
                    if dy > 0:  # this condition means the trajectory is straying away from end point in y
                        drift[1] += dy
            if col == "red":
                settings.red += 1
            point_dict[(x, y)] = [col, start_dis, end_dis, collision_states, drift, out_of_bound_states]
            plt.plot(new_path['x'][0, 0, :], new_path['x'][0, 1, :], color=col,
                     linewidth=0.3)  # plot generated trajectory
            plt.plot(x, y, marker="+", color=col, markersize=5, linewidth=0.3)  # visual marker
            plt.gca().add_artist(plt.Circle((x, y), increment / 2, color=col, fill=False,
                                            linewidth=0.3))  # visual marker of acceptable radius

    point_dict = normalize(point_dict)
    #print("uncertanty all point dict is: ", point_dict[(-0.5316666666666666, 0.23115384615384615)])

    sorted_dict = {k: v for k, v in sorted(point_dict.items(),
                                           key=lambda item: item[1][-1])}  # Sort dictionary by increasing uncertainty
    max_uncertainty_pt = list(sorted_dict.keys())[-1]
    max_uncertainty_pt2 = list(sorted_dict.keys())[-2]
    max_uncertainty_pt3 = list(sorted_dict.keys())[-3]

    print("Point with highest uncertainty:", max_uncertainty_pt, "second highest: ",max_uncertainty_pt2,"third highest: ",max_uncertainty_pt3,  "Uncertainty:",
          sorted_dict[list(sorted_dict.keys())[-1]], "medium Uncertainty:",
          sorted_dict[list(sorted_dict.keys())[-50]])
    ###to change in actual thing!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    uncertainty = []
    for i in range(len(list(point_dict.keys()))):
        uncertainty.append(point_dict[(list(point_dict.keys())[i])][-1])
    ###to change in actual thing!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #print("point dict is: ", point_dict[(list(point_dict.keys())[-1])][-1], "index is: ", list(point_dict.keys()))
    if settings.vis_mode == 1:
        # then it is using 4cm rule, need to visualise the intended point differently
        cm_rule(x_points, y_points, point_dict)
    elif settings.vis_mode == 2:
        # then it is using information entropy

        print("Point with highest uncertainty:", max_uncertainty_pt, "Uncertainty:",
              sorted_dict[list(sorted_dict.keys())[-1]])
        plt.plot(max_uncertainty_pt[0], max_uncertainty_pt[1], marker=".", color="blue", markersize=5,
                 linewidth=0.3)  # highlight max uncertainty point

    plt.gca().add_artist(plt.Circle((settings.goal_x, settings.goal_y), end_goal_radius, color="0.1", fill=False,
                                    linewidth=0.3))  # visual marker of acceptable radius

    return (x_points, y_points, point_dict, max_uncertainty_pt, jerk, pnt_dis, MuGMR, SigmaGMR)

#######################4 cm rule stuff########################
def cm_rule(x_points, y_points, point_dict):
    cm_covered = []

    determined_points = []
    red_count = 0
    #first_xy_index = get_first_point(x_points, y_points, [settings.data[1, 0], settings.data[2, 0]])
    #red_points = is_surrounded(x_points, y_points, point_dict, first_xy_index)
    if 1:#len(red_points) == 0:
        for i in range(len(x_points)):
            for j in range(len(y_points)):
                if (point_dict[(x_points[i], y_points[j])][0] == 'red'):
                    red_count += 1
                    if i <= 1:  # for horizontal direction
                        if (point_dict[(x_points[i + 1], y_points[j])][0] == 'green') or (
                                i == 1 and point_dict[(x_points[i - 1], y_points[j])][0] == 'green'):
                            cm_covered = add_pairs(cm_covered, [i, j])
                    elif i >= 2:
                        if (point_dict[(x_points[i - 1], y_points[j])][0] == 'green') or (
                                i == 2 and point_dict[(x_points[i + 1], y_points[j])][0] == 'green'):
                            cm_covered = add_pairs(cm_covered, [i, j])
                    if j <= 1:  # for vertical direction
                        if (point_dict[(x_points[i], y_points[j + 1])][0] == 'green') or (
                                j == 1 and point_dict[(x_points[i], y_points[j - 1])][0] == 'green'):
                            cm_covered = add_pairs(cm_covered, [i, j])
                    elif j >= 12:
                        if (point_dict[(x_points[i], y_points[j - 1])][0] == 'green') or (
                                j == 12 and point_dict[(x_points[i], y_points[j + 1])][0] == 'green'):
                            cm_covered = add_pairs(cm_covered, [i, j])
                    else:
                        if point_dict[(x_points[i], y_points[j - 1])][0] == 'green' or \
                                point_dict[(x_points[i], y_points[j + 1])][0] == 'green':
                            cm_covered = add_pairs(cm_covered, [i, j])
                    if i >= 1 and i <= 2 and j <= 12 and j >= 1:  # for diagonal direction
                        if point_dict[(x_points[i - 1], y_points[j - 1])][0] == 'green' or \
                                point_dict[(x_points[i + 1], y_points[j + 1])][0] == 'green' or \
                                point_dict[(x_points[i + 1], y_points[j - 1])][0] == 'green' or \
                                point_dict[(x_points[i - 1], y_points[j + 1])][0] == 'green':
                            cm_covered = add_pairs(cm_covered, [i, j])

    else:
        cm_covered = red_points
    # now we have a list of the surrounding points, we randomly select one and let it be the next point to set
    if len(cm_covered) > 0:
        if len(cm_covered) == 1:
            determined_points.append(cm_covered[0][0])
            determined_points.append(cm_covered[0][1])
        else:
            index = randrange(0, len(cm_covered) - 1, 1)
            determined_points.append(cm_covered[index][0])
            determined_points.append(cm_covered[index][1])

    if len(determined_points) > 0:
        plt.plot(x_points[determined_points[0]], y_points[determined_points[1]], marker=".", color="blue", markersize=5,
                 linewidth=0.3)  # highlight next point


def get_first_point(x_points, y_points, data_point):
    distance = 100
    ret_coord = [0, 0]
    for i in range(len(x_points)):
        for j in range(len(y_points)):
            distance_temp = np.square(data_point[0] - x_points[i]) + np.square(data_point[1] - y_points[j])
            if distance_temp < distance:
                distance = distance_temp
                ret_coord[0] = i
                ret_coord[1] = j
    return ret_coord


def is_surrounded(x_points, y_points, point_dict, index_point):
    red_points = []
    diag = [0, 0, 0, 0]  # top left, top right, bot left, bot right
    if index_point[0] + 1 <= 3:
        diag[1] += 1
        diag[3] += 1
        if point_dict[(x_points[index_point[0] + 1], y_points[index_point[1]])][0] == 'red':
            red_points = add_pairs(red_points, [index_point[0] + 1, index_point[1]])
    if index_point[0] - 1 >= 0:
        diag[0] += 1
        diag[2] += 1
        if point_dict[(x_points[index_point[0] - 1], y_points[index_point[1]])][0] == 'red':
            red_points = add_pairs(red_points, [index_point[0] - 1, index_point[1]])
    if index_point[1] + 1 <= 13:
        diag[0] += 1
        diag[1] += 1
        if point_dict[(x_points[index_point[0]], y_points[index_point[1] + 1])][0] == 'red':
            red_points = add_pairs(red_points, [index_point[0], index_point[1] + 1])
    if index_point[1] - 1 >= 0:
        diag[2] += 1
        diag[3] += 1
        if point_dict[(x_points[index_point[0]], y_points[index_point[1] - 1])][0] == 'red':
            red_points = add_pairs(red_points, [index_point[0], index_point[1] - 1])
    for i in range(len(diag)):
        if diag[i] == 2:
            if i == 0 and point_dict[(x_points[index_point[0] - 1], y_points[index_point[1] + 1])][0] == 'red':
                red_points = add_pairs(red_points, [index_point[0] - 1, index_point[1] + 1])
            elif i == 1 and point_dict[(x_points[index_point[0] + 1], y_points[index_point[1] + 1])][0] == 'red':
                red_points = add_pairs(red_points, [index_point[0] + 1, index_point[1] + 1])
            elif i == 2 and point_dict[(x_points[index_point[0] - 1], y_points[index_point[1] - 1])][0] == 'red':
                red_points = add_pairs(red_points, [index_point[0] - 1, index_point[1] - 1])
            elif i == 3 and point_dict[(x_points[index_point[0] - 1], y_points[index_point[1] + 1])][0] == 'red':
                red_points = add_pairs(red_points, [index_point[0] - 1, index_point[1] + 1])

    return red_points


def add_pairs(listxy, xy):
    if not (xy in listxy):
        listxy.append(xy)
    return listxy

#############################4cm rule stuff#####################################
def start_end_dis(new_path,x,y):
    start_dis = sqrt((x-new_path['x'][0,0,0])**2+(y-new_path['x'][0,1,0])**2) #pythagoras to determine distance away from intended start
    end_dis = sqrt((settings.goal_x-new_path['x'][0,0,-1])**2+(settings.goal_y-new_path['x'][0,1,-1])**2) #pythag to determine distance away from goal
    return (start_dis,end_dis)



def p_controller(x_target,y_target):
    x = x_target
    y = y_target
    k_p = settings.k_p
    x_error = 10000 # initialize x_diff and y_diff to be large values
    y_error = 10000

    while abs(x_error) > settings.pd_accuracy or abs(y_error) > settings.pd_accuracy:
        new_path,MuGMR, SigmaGMR = generate_trajectory(x_init = x, y_init = y, dims = 2)

        x_error = x_target-new_path['x'][0,0,0]
        y_error = y_target-new_path['x'][0,1,0]

        x += x_error*k_p
        y += y_error*k_p
    return new_path,MuGMR, SigmaGMR

def normalize(point_dict):
    max_end_dis = max(point_dict.items(), key=lambda item: item[1][2])[1][2]
    max_drift_x = max(point_dict.items(), key=lambda item: item[1][4][0])[1][4][0]
    min_end_dis = min(point_dict.items(), key=lambda item: item[1][2])[1][2]
    min_drift_x = min(point_dict.items(), key=lambda item: item[1][4][0])[1][4][0]

    min_collision = min(point_dict.items(), key=lambda item: item[1][3])[1][3]
    max_collision = max(point_dict.items(), key=lambda item: item[1][3])[1][3]
    min_outbound = min(point_dict.items(), key=lambda item: item[1][5])[1][5]
    max_outbound = max(point_dict.items(), key=lambda item: item[1][5])[1][5]

    tot_uncertainty = 0
    # normalize each parameter
    temp_col = [0, 0]
    for k in point_dict.keys():
        #(-0.57+0.455)^2+(0.425-0.005)^2+0.015
        #point_dict[k][2] = (point_dict[k][2] - min_end_dis) / (max_end_dis - min_end_dis)

        if abs(point_dict[k][2]) <= 0.015:
            point_dict[k][2] = 0
        else:
            point_dict[k][2] /= 0.45 ##sqrt((-0.57+0.455)^2+(0.425-0.005)^2)+0.015
            #point_dict[k][2] = (point_dict[k][2] - min_end_dis) / (max_end_dis - min_end_dis)
        '''
        if max_collision > 0:
            point_dict[k][3] = (point_dict[k][3] - min_collision) / (max_collision-min_collision)
        '''
        if temp_col[0] < point_dict[k][3]:
            temp_col[0] = point_dict[k][3]
        if temp_col[1] < point_dict[k][5]:
            temp_col[1] = point_dict[k][5]
        point_dict[k][3]/=30
        #point_dict[k][3] /= (100*settings.obstacle_workspace_ratio)
        if max_drift_x > 0:
            point_dict[k][4][0] = (point_dict[k][4][0] - min_drift_x) / (max_drift_x - min_drift_x)
        '''
        if max_outbound > 0:
            point_dict[k][5] = (point_dict[k][5] - min_outbound) / (max_outbound - min_outbound)
        '''
        point_dict[k][5] /=100
        #point_dict[k][5] /= 100
        uncertainty = settings.beta*point_dict[k][2]+settings.gamma*point_dict[k][3]+settings.delta*point_dict[k][4][0]+settings.epsilon*point_dict[k][5]
        point_dict[k].append(uncertainty)
        tot_uncertainty += uncertainty

    # normalize to all add to 1
    test = 0
    for k in point_dict.keys():
        point_dict[k][-1] /= tot_uncertainty
        test += point_dict[k][-1]
    print(test)

    for k in point_dict.keys():
        # prevent log 0
        if (point_dict[k][-1] > 0):
            point_dict[k][-1] = -1 * point_dict[k][-1] * log(point_dict[k][-1])
    print("number of collision and out of point is: ", temp_col)
    return (point_dict)
'''
def normalize(point_dict):
    max_end_dis = max(point_dict.items(), key=lambda item: item[1][2])[1][2]
    max_drift_x = max(point_dict.items(), key=lambda item: item[1][4][0])[1][4][0]
    min_end_dis = min(point_dict.items(), key=lambda item: item[1][2])[1][2]
    min_drift_x = min(point_dict.items(), key=lambda item: item[1][4][0])[1][4][0]

    min_collision = min(point_dict.items(), key=lambda item: item[1][3])[1][3]
    max_collision = max(point_dict.items(), key=lambda item: item[1][3])[1][3]
    min_outbound = min(point_dict.items(), key=lambda item: item[1][5])[1][5]
    max_outbound = max(point_dict.items(), key=lambda item: item[1][5])[1][5]

    tot_uncertainty = 0
    # normalize each parameter
    for k in point_dict.keys():

        point_dict[k][2] = (point_dict[k][2] - min_end_dis) / (max_end_dis - min_end_dis)
        if max_collision > 0:
            point_dict[k][3] = (point_dict[k][3] - min_collision) / (max_collision-min_collision)
        #point_dict[k][3] /= (100*settings.obstacle_workspace_ratio)
        if max_drift_x > 0:
            point_dict[k][4][0] = (point_dict[k][4][0] - min_drift_x) / (max_drift_x - min_drift_x)
        if max_outbound > 0:
            point_dict[k][5] = (point_dict[k][5] - min_outbound) / (max_outbound - min_outbound)
        #point_dict[k][5] /= 100
        uncertainty = settings.beta*point_dict[k][2]+settings.gamma*point_dict[k][3]+settings.delta*point_dict[k][4][0]+settings.epsilon*point_dict[k][5]
        point_dict[k].append(uncertainty)
        tot_uncertainty += uncertainty

    # normalize to all add to 1
    test = 0
    for k in point_dict.keys():
        point_dict[k][-1] /= tot_uncertainty
        test += point_dict[k][-1]
    print(test)

    for k in point_dict.keys():
        # prevent log 0
        if (point_dict[k][-1] > 0):
            point_dict[k][-1] = -1 * point_dict[k][-1] * log(point_dict[k][-1])

    return (point_dict)
'''

def generate_dataframe(x_points, y_points, point_dict, max_u_pt):
    # initialize an empty dataframe
    init_df = np.zeros((len(y_points), len(x_points)))

    df = pd.DataFrame(data=init_df, index=y_points, columns=x_points)
    for y in y_points:
        for x in x_points:
            df.at[y, x] = point_dict[(x, y)][-1]  # /point_dict[max_u_pt][5]
    return df

def get_bic_score(start, end, demos_to_use):
    for i in range(start, end+1):
        tpgmm = TPGMM_Time(i, reg_covar=settings.reg, max_steps=200)  # initialize TPGMM_Time
        data = retrieve_demos(demos_to_use)  # obtain demos
        frames = get_frames(data, dims=2)  # generate frames
        # print ("frame is: " + str(frames))
        tpgmm.fit(data, frames)  # fit data to frames
        data_bic = data[:, :, 0]
        for j in range(1, data.shape[2]):
            data_temp = data[:, :, j]
            data_bic = np.concatenate((data_bic, data_temp))
        print("bic is on index  " + str(i))

        # plot1 = plt.figure(1)

        # x_points, y_points, point_dict, max_u_pt, error_traj = trajectory_grid(settings.increment, settings.end_goal_radius)  # generate grid of trajectories
        bic_score.append(tpgmm.bic_test(data_bic, np.linspace(0, 10.0, num=100)))

def get_teaching_effacacy(num_green, total):
    return num_green/total
def get_teaching_efficiency(efficacy, num_demo):
    return efficacy/num_demo

# --- MAIN --- #


settings = Params()
ax = add_rectangles()
#record the time taken
start_time = time.time()

increment = 0.03
x_points = np.linspace(-0.36+ settings.shift[0] + increment / 2, -0.215+ settings.shift[0] - increment / 2,
                           round(((0.36+ settings.shift[0] - increment / 2) - (0.215+ settings.shift[0] + increment / 2)) / increment))  # x-grid generated
y_points = np.linspace(0.04+ settings.shift[1] + increment / 2, 0.49+ settings.shift[1] - increment / 2,
                           round(((0.49+ settings.shift[1] - increment / 2) - (0.04+ settings.shift[1] + increment / 2)) / increment))  # y-grid generated

tgpmm_components = 10
bic_score = []
bic = []
'''
gmm = mixture.GaussianMixture(
            n_components=n_components, covariance_type=cv_type
        )
        gmm.fit(X)
        bic.append(gmm.bic(X))
'''
#get_bic_score(1,settings.max_gmm, settings.demos_to_use)
# get the minimum index
min_index = 25
#min_index = bic_score.index(min(bic_score[0:]))
#print("bic score is: ", bic_score)
print("min bic score num of elements is: ", min_index)

#fit the smallest bic score elements
tpgmm = TPGMM_Time(min_index, reg_covar=settings.reg, max_steps=200)  # initialize TPGMM_Time
data = retrieve_demos(settings.demos_to_use)  # obtain demos
settings.Data = data
frames = get_frames(data, dims=2)  # generate frames
# print ("frame is: " + str(frames))
priors, Mu_fit, Sigma_fit = tpgmm.fit(data, frames)  # fit data to frames
print("dimensions of fitted Mu: ", Mu_fit.shape, " and of Covar: ", Sigma_fit.shape)

plot1 = plt.figure(1)

x_points, y_points, point_dict, max_u_pt, jerk, pnt_distance, Mu, Sigma = trajectory_grid(settings.increment,
                                                           settings.end_goal_radius)  # generate grid of trajectories
plot_ellipsoids(Mu_fit, Sigma_fit, ax)
plot_ellipsoids(Mu, Sigma, ax)
df = generate_dataframe(x_points, y_points, point_dict, max_u_pt)

plot_demos(settings.demos_to_use,Mu, Sigma)  # visually plot demos extracted

plt.axis('scaled')  # ensure 1:1 scaling of the plot
plot2 = plt.figure(2)
plt.axis('scaled')
ax = sns.heatmap(df, cmap="coolwarm")
print("df is:", df)
ax.invert_yaxis()

# record the time taken
end_time = time.time()
print("time taken for entire operation is: " + str(end_time - start_time))
print("total jerk are: ", jerk, " and point distances are: ", pnt_distance)
print("efficacy is: ", get_teaching_effacacy(56 - settings.red, 56))
#plot error trajecories
#x_error = [row[0] for row in error_traj]
#y_error = [row[1] for row in error_traj]
#plt.figure("x error trajectories")
#plt.plot(np.arange(len(x_error)), x_error)
#plt.figure("y error trajectories")
#plt.plot(np.arange(len(y_error)), y_error)

plt.figure("bic score graph")
plt.plot(np.arange(len(bic_score)), bic_score)
plt.figure("gmm components ellipsoids")
#fig, ax = plt.subplots()
#plot_ellipsoids(Mu, Sigma, ax)
#plt.axis('scaled')
plt.show()  # ensure 1:1 scaling of the plot, and illustrate it