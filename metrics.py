from msilib.schema import Class
from queue import Empty
from re import X
import matplotlib.pyplot as plt
import datetime
import os
import wandb
from torch.utils.tensorboard import SummaryWriter

class metrics():
    def __init__(self):
        self.episodes = []
        self.rewards1 = []
        self.rewards2 = []
        self.paretoepisodes = []
        self.paretorewards1 = []
        self.paretorewards2 = []
        self.paretor0 = []
        self.paretor1 = []
        self.nonDominatedPoints = []
        self.ndPoints =[]
        self.pdict = {}
        self.xA0 = []
        self.yA0 = []        
        self.xA1 = []
        self.yA1 = []
        self.xA2 = []
        self.yA2 = []
        self.xA3 = []
        self.yA3 = []
        self.count = 0
        self.path = ''
        self.writer = SummaryWriter(f"pdn")
        
        


    def plotGraph(self):
        
        
        fig, ax = plt.subplots()
        ax.plot(self.episodes, self.rewards1)
        ax.set_title('Treasure reward x Episodes')
        #plt.show()
       
        
        fig, ax2 = plt.subplots()
        ax2.plot(self.episodes, self.rewards2)
        ax2.set_title('Time penalty x Episodes')
        
        
       
        plt.show()
        
    def setup_wandb(self, project_name: str, experiment_name: str):
        self.experiment_name = experiment_name
        import wandb

        wandb.init(
            project=project_name,
            sync_tensorboard=True,
            config=self.get_config(),
            name=self.experiment_name,
            monitor_gym=True,
            save_code=True,

        )
        self.writer = SummaryWriter(f"{self.experiment_name}")
        # The default "step" of wandb is not the actual time step (gloabl_step) of the MDP
        wandb.define_metric("*", step_metric="global_step")

    def close_wandb(self):
        import wandb
        self.writer.close()
        wandb.finish()
        
    def get_config(self) -> dict:
        """Generates dictionary of the algorithm parameters configuration
        Returns:
            dict: Config
        """
        
        
    def plot_pareto_frontier(self,Xs,Ys):
        import numpy as np

        frontier = []


        

        points = np.column_stack((Xs, Ys))
        uniques = np.unique(points,axis=0)
        
        inputPoints = uniques.tolist()
        paretoPoints, dominatedPoints = simple_cull(inputPoints, dominates)

        print ("*"*8 + " non-dominated answers " + ("*"*8))
        for p in paretoPoints:
            frontier.append(p)
            print (p)
        print ("*"*8 + " dominated answers " + ("*"*8))
        for p in dominatedPoints:
            pass
            #print (p)
        #print(arr)

        print(frontier)
        
  
        pf_Xx = [pair[0] for pair in frontier]
        pf_Yy = [pair[1] for pair in frontier]
        
        Xs = points[:,0]
        Ys = points[:,1]
        
       
        points = list(zip(pf_Xx, pf_Yy))
        points_sorted = sorted(points)
        x_sorted, y_sorted = zip(*points_sorted)

        # Print the sorted points
        print("Sorted points by x-coordinate:")
        print(list(points_sorted))
        plt.plot(x_sorted, y_sorted, '-o')

        xreal = [1,2,3,5,8,16,24,50,74,124]
        yreal = [-1,-3,-5,-7,-8,-9,-15,-16,-19,-21]
        
        plt.plot(xreal,yreal)
        
        plt.xlabel("Treasure Reward  " )
        plt.ylabel("Time Penalty " )

        plt.show()
        
def simple_cull(inputPoints, dominates): 
    paretoPoints = set()
    candidateRowNr = 0
    dominatedPoints = set()
    
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            if dominates(candidateRow, row):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow):
                nonDominated = False
                dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0:
            break
    return paretoPoints, dominatedPoints
def dominates(row, candidateRow):
    return sum([row[x] >= candidateRow[x] for x in range(len(row))]) == len(row) 
