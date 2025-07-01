import socket
import warnings
from pathlib import Path
from typing import final

import joblib
import mediapipe as mp
import numpy as np
import torch
import re
from typing import Dict, List, Optional
import threading
from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import DpipCommonGroundTrackingInterface, PropositionInterface
import tkinter as tk                    
from tkinter import ttk

@final
class DpipCommonGroundTracking(BaseFeature):
    """
    Visualize the common gorund for the DPIP task.

    Input interfaces are `PropositionInterface`, ...

    Output interface is `None`

    Keyword arguments:
    """

    def __init__(
        self, prop: BaseFeature[PropositionInterface]
    ):
        super().__init__(prop) 
        self.init = False
        self.t = threading.Thread(target=self.worker)
        self.t.start()
        self.rowX = {}

    def initialize(self):
        print("DPIP Interface")
    
    def get_output(self, prop: PropositionInterface):
        #if not prop.is_new(): #TODO update to run only when props come in
        #    return None
        
        self.t = threading.Thread(target=self.worker)
        self.t.start()
        
        #TODO output banks for the planner
        return DpipCommonGroundTrackingInterface(
            qbank=[],
            fbank=[],
            ebank=[],
        )
    
    def worker(self):
        #print("New DPIP Interface Update Thread Started")
        try:
            if self.init == False:
                self.init = True
                self.root = tk.Tk()
                self.root.title("DPIP Common Ground")
                self.tabControl = ttk.Notebook(self.root)

                self.tab1 = ttk.Frame(self.tabControl)
                self.tab2 = ttk.Frame(self.tabControl)
                self.tab3 = ttk.Frame(self.tabControl)

                self.tabControl.add(self.tab1, text ='D1')
                self.tabControl.add(self.tab2, text ='D2')
                self.tabControl.add(self.tab3, text ='D3')
                
                self.tabControl.pack(expand = 1, fill ="both")

                self.canvas1 = tk.Canvas(self.tab1, bg="white", height=250, width=300)
                self.canvas2 = tk.Canvas(self.tab2, bg="white", height=250, width=300)
                self.canvas3 = tk.Canvas(self.tab3, bg="white", height=250, width=300)

                self.canvas1.pack()
                self.canvas2.pack()
                self.canvas3.pack()
                
                self.root.mainloop()
            else:
                self.canvas1.delete("all")
                self.canvas2.delete("all")
                self.canvas3.delete("all")
                self.rowX = {}

                self.renderRectangles(1, 1, 1, 'orange')
                self.renderRectangles(1, 1, 2, 'white')
                self.renderRectangles(1, 1, 1, 'blue')
                self.renderRectangles(1, 2, 2, 'red')
                self.renderRectangles(1, 2, 1, 'blue')
                self.renderRectangles(1, 3, 2, 'purple')

                self.renderRectangles(2, 2, 1, 'red')
                self.renderRectangles(2, 2, 2, 'purple')
                self.renderRectangles(3, 3, 1, 'yellow')

                self.canvas1.pack()
                self.canvas2.pack()
                self.canvas3.pack()

        except Exception as e:
            print(f"DPIP FEATURE THREAD: An error occurred: {e}")

    def renderRectangles(self, side, row, size, color):
        key = str(side) + "_" + str(row)
        y = 10 + ((row - 1) * 60)
        if self.rowX.get(key) is None:
            self.rowX[key] = 10

        xOffset = 50 * size
        xStart = self.rowX.get(key)
        xEnd = self.rowX.get(key) + xOffset
        if(side == 1):
             self.canvas1.create_rectangle(xStart, y, xEnd, y + 50, fill=color, outline=color)
        if(side == 2):
             self.canvas2.create_rectangle(xStart, y, xEnd, y + 50, fill=color, outline=color)
        if(side == 3):
             self.canvas3.create_rectangle(xStart, y, xEnd, y + 50, fill=color, outline=color)
        self.rowX[key] = xEnd + 5
