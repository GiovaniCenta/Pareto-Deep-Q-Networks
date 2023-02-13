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
        


    def plotGraph(self):
        
        
        fig, ax = plt.subplots()
        ax.plot(self.episodes, self.rewards1)
        ax.set_title('Treasure reward x Episodes')
        #plt.show()
       
        
        fig, ax2 = plt.subplots()
        ax2.plot(self.episodes, self.rewards2)
        ax2.set_title('Time penalty x Episodes')
        
        
       
        plt.show()