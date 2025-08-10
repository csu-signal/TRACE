import json
import shutil
import socket
import time
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
from mmdemo.interfaces import ColorImageInterface, DpipActionInterface, DpipCommonGroundTrackingInterface, DpipFrictionOutputInterface, PropositionInterface
import tkinter as tk
from tkinter import Button, ttk
from PIL import ImageGrab
import os

from mmdemo.utils.files import create_tmp_dir_with_featureName

@final
class DpipCommonGroundTracking(BaseFeature):
    """
    Visualize the common gorund for the DPIP task.

    Input interfaces are `PropositionInterface`, ...

    Output interface is `None`

    Keyword arguments:
    """

    def __init__(
        self, prop: BaseFeature[DpipFrictionOutputInterface], color: BaseFeature[ColorImageInterface], actions: BaseFeature[DpipActionInterface], saveCanvas: bool | None = False
    ):
        super().__init__(prop, color, actions)
        self.init = False
        self.useTabs = True
        self.redraw = True
        self.t = threading.Thread(target=self.worker)
        self.t.start()
        self.rowX = {}
        self.currentCg = {
            "D1": {
                "row_0": [
                    {"color":"none", "size":1},
                    {"color":"none", "size":1},
                    {"color":"none", "size":1},
                    ],
                "row_1": [
                    {"color":"none", "size":1},
                    {"color":"none", "size":1},
                    {"color":"none", "size":1},
                    ],
                "row_2": [
                    {"color":"none", "size":1},
                    {"color":"none", "size":1},
                    {"color":"none", "size":1},
                ]
            },
            "D2": {
                "row_0": [
                    {"color":"none", "size":1},
                    {"color":"none", "size":1},
                    {"color":"none", "size":1},
                    ],
                "row_1": [
                    {"color":"none", "size":1},
                    {"color":"none", "size":1},
                    {"color":"none", "size":1},
                    ],
                "row_2": [
                    {"color":"none", "size":1},
                    {"color":"none", "size":1},
                    {"color":"none", "size":1},
                ]
            },
            "D3": {
                "row_0": [
                    {"color":"none", "size":1},
                    {"color":"none", "size":1},
                    {"color":"none", "size":1},
                    ],
                "row_1": [
                    {"color":"none", "size":1},
                    {"color":"none", "size":1},
                    {"color":"none", "size":1},
                    ],
                "row_2": [
                    {"color":"none", "size":1},
                    {"color":"none", "size":1},
                    {"color":"none", "size":1},
                ]
            }
        }
        self.lastCgLLM = ''
        self.lastCgStruct = ''

        self.saveCanvas = saveCanvas
        self.frameIndex = 0

        if(self.saveCanvas):
            self.outputDir = create_tmp_dir_with_featureName("commonGround")

    def initialize(self):
        print("DPIP Interface")

    def saveUi(self, filename):
        x = self.root.winfo_rootx()
        y = self.root.winfo_rooty()
        x1 = x + self.root.winfo_width()
        y1 = y + self.root.winfo_height()
        ImageGrab.grab().crop((x, y, x1, y1)).save(filename)

    def get_output(self, prop: DpipFrictionOutputInterface, color:ColorImageInterface, actions:DpipActionInterface):
        #if not prop.is_new(): #TODO update to run only when props come in
        #    return None
        self.frameIndex = color.frame_count
        if(self.init == True):
            if(prop.cg_json != "None" and prop.cg_json != ''):
                self.currentCg = json.loads(prop.cg_json)

            if(self.redraw or self.lastCgStruct != actions.jsonStructure or self.lastCgLLM != prop.cg_json):
                self.redraw = False
                self.lastCgLLM = prop.cg_json
                self.lastCgStruct = actions.jsonStructure
                self.t = threading.Thread(target=self.worker)
                self.t.start()

        #TODO output banks for the planner
        return DpipCommonGroundTrackingInterface(
            qbank=[],
            fbank=[],
            ebank=[],
        )

    def toggleTabs(self):
        self.init = not self.init
        self.useTabs = not self.useTabs
        self.redraw = True
        self.root.destroy()

        self.t = threading.Thread(target=self.worker)
        self.t.start()

    def tabChanged(self, event):
        notebook = event.widget
        selected_tab_id = notebook.select()
        tab_index = notebook.index(selected_tab_id)

        if(tab_index == 3):
            self.root.geometry("190x590") 
            self.root.mainloop()
        else:
            self.root.geometry("190x210") 
            self.root.mainloop()

    def worker(self):
        #print("New DPIP Interface Update Thread Started")
        try:
            if self.init == False:
                self.init = True

                self.root = tk.Tk()
                self.root.title("CG")
                self.tabControl = ttk.Notebook(self.root)

                if(self.saveCanvas): #keep ui on top if we are saving images
                    self.root.wm_attributes("-topmost", True)

                self.tab1 = ttk.Frame(self.tabControl)
                self.tab2 = ttk.Frame(self.tabControl)
                self.tab3 = ttk.Frame(self.tabControl)
                self.tab4 = ttk.Frame(self.tabControl)

                self.tabControl.add(self.tab1, text ='D1' if self.useTabs else 'CG')
                self.tabControl.add(self.tab2, text ='D2')
                self.tabControl.add(self.tab3, text ='D3')
                self.tabControl.add(self.tab4, text ='All')

                self.tabControl.pack(expand = 1, fill ="both")

                self.canvas1 = tk.Canvas(self.tab1, bg="white", height=185, width=185)
                self.canvas2 = tk.Canvas(self.tab2, bg="white", height=185, width=185)
                self.canvas3 = tk.Canvas(self.tab3, bg="white", height=185, width=185)
                
                self.canvas1All = tk.Canvas(self.tab4, bg="white", height=185, width=185)
                self.canvas2All = tk.Canvas(self.tab4, bg="white", height=185, width=185)
                self.canvas3All = tk.Canvas(self.tab4, bg="white", height=185, width=185)

                #btn.pack(side = 'top', anchor="ne")
                self.canvas1.pack()
                self.canvas2.pack()
                self.canvas3.pack()

                self.canvas1All.pack()
                self.canvas2All.pack()
                self.canvas3All.pack()

                self.tabControl.bind("<<NotebookTabChanged>>", self.tabChanged)

                self.root.mainloop()
            else:
                self.canvas1.delete("all")
                self.canvas2.delete("all")
                self.canvas3.delete("all")

                self.canvas1All.delete("all")
                self.canvas2All.delete("all")
                self.canvas3All.delete("all")
                self.rowX = {}

                if(self.lastCgStruct != {}):
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

                self.canvas1All.pack()
                self.canvas2All.pack()
                self.canvas3All.pack()

                if(self.saveCanvas):
                    self.tabControl.select(3)
                    frame = self.frameIndex
                    time.sleep(1) 
                    self.saveUi(f"{self.outputDir}/{frame}.png")


        except Exception as e:
            print(f"DPIP FEATURE THREAD: An error occurred: {e}")

    def renderSide(self, side, row):
        blocks = self.lastCgStruct[side][row]
        blocksLLM = self.currentCg[side][row]
        uiRow = 2 if row == "row_0" else 1 if row == "row_1" else 0
        length = len(blocks)
        skip = False

        for i, b in enumerate(blocks):
            if(blocksLLM[i]["color"] == "unknown"):
                if skip:
                    skip = False
                    continue
                size = blocksLLM[i]["size"]
                color = blocksLLM[i]["color"]
                if(size == 2 and i + 1 < length and blocksLLM[i + 1]["size"] == size and blocksLLM[i + 1]["color"] == color):
                    self.renderRectangles(int(side.split("D")[1]), uiRow, size, color, b["color"], True)
                    skip = True
                else:
                    self.renderRectangles(int(side.split("D")[1]), uiRow, 1, color, b["color"], True)
            else:
                if skip:
                    skip = False
                    continue
                size = b["size"]
                color = b["color"]
                if(size == 2 and i + 1 < length and blocks[i + 1]["size"] == size and blocks[i + 1]["color"] == color):
                    self.renderRectangles(int(side.split("D")[1]), uiRow, size, color, b["color"], False)
                    skip = True
                else:
                    self.renderRectangles(int(side.split("D")[1]), uiRow, 1, color, b["color"], False)

    def renderRectangles(self, side, row, size, color, outline, llm):
        if (llm and color != "unknown"):
            return

        if(color == "none"):
            color = "white"

        if(color == "unknown"):
            if(llm):
                color = "black"
            else:
                color = 'gray'

        if(outline == "unknown"):
            outline = 'gray'

        if(not llm):
            outline = color

        #colorLabel = color
        font = "black"
        if color == "black":
            font = "white"

        key = str(side) + "_" + str(row)
        y = 10 + ((row) * 60)
        if self.rowX.get(key) is None:
            self.rowX[key] = 10

        xOffset = (50 * size)
        if(size == 2):
            xOffset = xOffset + 10
        xStart = self.rowX.get(key)
        xEnd = self.rowX.get(key) + xOffset
        #shape = "r" if size == 2 else "s"
        if(side == 1):
             self.canvas1.create_rectangle(xStart, y, xEnd, y + 50, fill=color, outline=outline, width=5)
             self.canvas1All.create_rectangle(xStart, y, xEnd, y + 50, fill=color, outline=outline, width=5)
             #self.canvas1.create_text(xStart + 15, y + 15, text=f"{colorLabel[0]}{shape}", fill=font)
        if(side == 2):
             self.canvas2.create_rectangle(xStart, y, xEnd, y + 50, fill=color, outline=outline, width=5)
             self.canvas2All.create_rectangle(xStart, y, xEnd, y + 50, fill=color, outline=outline, width=5)
             #self.canvas2.create_text(xStart + 15, y + 15, text=f"{colorLabel[0]}{shape}", fill=font)
        if(side == 3):
             self.canvas3.create_rectangle(xStart, y, xEnd, y + 50, fill=color, outline=outline, width=5)
             self.canvas3All.create_rectangle(xStart, y, xEnd, y + 50, fill=color, outline=outline, width=5)
             #self.canvas3.create_text(xStart + 15, y + 15, text=f"{colorLabel[0]}{shape}", fill=font)
        self.rowX[key] = xEnd + 10
