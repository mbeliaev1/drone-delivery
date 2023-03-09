import sys
sys.path.append('/home/mark/Documents/code/drone')
import os
import subprocess
import numpy as np
from sumo.utils.runSumo import runSumo
from sumo.utils.sumo_loop import sumo_loop
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
import pickle


class c_picker(object):
    def __init__(self,in_dir):
        self.in_dir = in_dir
        self.ratios = np.logspace(np.log10(0.001),np.log10(0.5),100)
        self.flows = np.arange(10,3000,10)
        self.r_vs_f = pickle.load(open(self.in_dir+"r_vs_f.p",'rb'))*3600
        print(self.r_vs_f.shape)
        print(self.flows.shape)
        self.click = None
        self.capacity = []

    def onclick(self,event):
        self.click = [event.ydata, event.ydata]
        print(self.click)
    def run(self):
        for r_idx in range(len(self.ratios)):
            # init new plot
            print(self.capacity)
            plt.ion()
            fig, axs = plt.subplots(1,1,figsize=(10, 10))
            axs.set_title(str(self.ratios[r_idx]))
            axs.plot(self.flows,self.flows,linestyle="dashed")
            axs.plot(self.flows,self.r_vs_f[r_idx,:])
            axs.set_xlabel("flow in")
            axs.set_ylabel("flow out")
            fig.canvas.mpl_connect('button_press_event', self.onclick)
            # this will not hang because plt.ion()
            plt.show()
            done = False
            # force hang while waiting for correct click
            #------------------------------------------------------------#
            while not done:
                choice = input(" d-display current point\n r-reset plot\n s-save current point\n")
                # display current point
                if choice == 'd':
                    temp = np.ones_like(self.flows)*self.click[0]
                    axs.plot(self.flows,temp,linestyle='dashed',color='red')
                # reset current plot
                elif choice == 'r':
                    plt.close()
                    fig, axs = plt.subplots(1,1,figsize=(10, 10))
                    axs.set_title(str(self.ratios[r_idx]))
                    axs.plot(self.flows,self.flows,linestyle="dashed")
                    axs.plot(self.flows,self.r_vs_f[r_idx,:])
                    axs.set_xlabel("flow in")
                    axs.set_ylabel("flow out")
                    fig.canvas.mpl_connect('button_press_event', self.onclick)
                    plt.show()
                elif choice == 's':
                    plt.close()
                    if len(self.capacity) != 0:
                        if self.capacity[-1] < self.click[0]:
                            print("WARNING: Previous capacity value: ",self.capacity[-1]," is smaller than current: ",self.click[0])
                            warning = input("continue?: y/n")
                            if warning == 'y':
                                print("saving capacity: ",self.click[0])
                                self.capacity.append(self.click[0])
                                done = True
                        break
                
                    print("saving capacity: ",self.click[0])
                    self.capacity.append(self.click[0])
                    done = True
                else:
                    print("invalid input try again: ")
            #-----------------------------------------------------------------#
        print(self.capacity)
        pickle.dump(self.capacity,open(self.in_dir+"c.p",'wb'))

if __name__ == "__main__":
    c_pick = c_picker("sim_3/")
    c_pick.run()


# # init
# sim_dir = "sim_1/"
# ratios = np.logspace(np.log10(0.001),np.log10(0.8),100)
# flows = np.arange(10,3600,10)

# # lets visualize these while picking out
# r_vs_f = pickle.load(open("sim_1/r_vs_f.p",'rb'))*3600
# capacity = []
# for r_idx in range(len(ratios)):
#     fig, axs = plt.subplots(1,1)
#     axs.set_title(str(ratios[r_idx]))
#     axs.plot(flows,flows,linestyle="dashed")
#     axs.plot(flows,r_vs_f[r_idx,:])
#     axs.set_xlabel("flow in")
#     axs.set_ylabel("flow out")
#     fig.canvas.mpl_connect('button_press_event', onclick)
#     # this will hang until plot is closed
#     plt.show()



# print(capacity)
# pickle.dump(capacity,open(sim_dir+"c.p",'wb'))