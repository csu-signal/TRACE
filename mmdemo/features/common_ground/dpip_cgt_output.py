import json
import shutil
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
from mmdemo.interfaces import ColorImageInterface, DpipCommonGroundTrackingInterface, DpipFrictionOutputInterface, PropositionInterface
import tkinter as tk                    
from tkinter import ttk
from PIL import ImageGrab
import os

@final
class DpipCommonGroundTracking(BaseFeature):
    """
    Visualize the common gorund for the DPIP task.

    Input interfaces are `PropositionInterface`, ...

    Output interface is `None`

    Keyword arguments:
    """

    def __init__(
        self, prop: BaseFeature[DpipFrictionOutputInterface], color: BaseFeature[ColorImageInterface], saveCanvas: bool | None = False
    ):
        super().__init__(prop, color) 
        self.init = False
        self.t = threading.Thread(target=self.worker)
        self.t.start()
        self.rowX = {}
        self.currentCg = ''
        self.lastCg = ''
        self.saveCanvas = saveCanvas
        self.frameIndex = 0

        if(self.saveCanvas):
            if os.path.isdir("cg_output/"):
                shutil.rmtree("cg_output/")

            # Create the folder and any necessary parent directories
            # exist_ok=True prevents an error if the directory already exists
            os.makedirs("cg_output/", exist_ok=True)
            os.makedirs("cg_output/d1/", exist_ok=True)
            os.makedirs("cg_output/d2/", exist_ok=True)
            os.makedirs("cg_output/d3/", exist_ok=True)

    def initialize(self):
        print("DPIP Interface")

    def save_canvas(self, canvas_widget, filename):
        x = self.root.winfo_rootx() + canvas_widget.winfo_x()
        y = self.root.winfo_rooty() + canvas_widget.winfo_y()
        x1 = x + canvas_widget.winfo_width()
        y1 = y + canvas_widget.winfo_height()
        ImageGrab.grab().crop((x, y, x1, y1)).save(filename)
    
    def get_output(self, prop: DpipFrictionOutputInterface, color:ColorImageInterface):
        #if not prop.is_new(): #TODO update to run only when props come in
        #    return None
        self.frameIndex = color.frame_count
        if(self.init == False or (prop.cg_json != "None" and prop.cg_json != '' and self.lastCg != prop.cg_json)):
            self.lastCg = prop.cg_json
            self.currentCg = json.loads(prop.cg_json)
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

                self.renderSide("D1", "row_0")
                self.renderSide("D1", "row_1")
                self.renderSide("D1", "row_2")

                self.renderSide("D2", "row_0")
                self.renderSide("D2", "row_1")
                self.renderSide("D2", "row_2")

                self.renderSide("D3", "row_0")
                self.renderSide("D3", "row_1")
                self.renderSide("D3", "row_2")

                self.canvas1.pack()
                self.canvas2.pack()
                self.canvas3.pack()

                if(self.save_canvas):
                    self.tabControl.select(0) # Selects the second tab (index 0)
                    self.save_canvas(self.canvas1, f"cg_output/d1/{self.frameIndex}.png")
                    self.tabControl.select(1) # Selects the second tab (index 1)
                    self.save_canvas(self.canvas2, f"cg_output/d2/{self.frameIndex}.png")
                    self.tabControl.select(2) # Selects the second tab (index 2)
                    self.save_canvas(self.canvas3, f"cg_output/d3/{self.frameIndex}.png")

        except Exception as e:
            print(f"DPIP FEATURE THREAD: An error occurred: {e}")

    def renderSide(self, side, row):
        blocks = self.currentCg[side][row]
        uiRow = 2 if row == "row_0" else 1 if row == "row_1" else 0
        for b in blocks:
            self.renderRectangles(int(side.split("D")[1]), uiRow, b["size"], b["color"])

    def renderRectangles(self, side, row, size, color):
        if(color == "unknown"):
            color = "black"

        key = str(side) + "_" + str(row)
        y = 10 + ((row) * 60) #TODO I think this is upside down, row 0 is the bottom of the structure
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
