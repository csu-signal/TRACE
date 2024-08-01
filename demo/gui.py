from abc import ABC, abstractmethod
from tkinter import LEFT, RIGHT, Checkbutton, Frame, IntVar, Label, Tk
from typing import final

import numpy as np
from PIL import Image, ImageTk


class BaseGui(ABC):
    @property
    @abstractmethod
    def running(self) -> bool:
        """Whether the GUI is running or not"""
        raise NotImplementedError

    @abstractmethod
    def mainloop(self):
        """
        Run the main gui loop. This will only be run from the main
        thread.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """
        Manually close the GUI. This must work from any thread.
        """
        raise NotImplementedError

    @abstractmethod
    def new_image(self, frame):
        """
        Update the most recent output image from the demo.
        """
        raise NotImplementedError

    @abstractmethod
    def feature_active(self, name:str) -> bool:
        """
        Check if the feature with the given name should be processed. See
        FeatureManager.processFrame for the inputs that must be supported.
        """
        raise NotImplementedError


class GuiFeatureInfo:
    def __init__(self, var_val: bool):
        self.var = IntVar(value=int(var_val))
        self.var_val = var_val


@final
class Gui(BaseGui):
    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("Demo")

        self.feature_vars = {
            "gesture": GuiFeatureInfo(True),
            "objects": GuiFeatureInfo(True),
            "gaze": GuiFeatureInfo(True),
            "asr": GuiFeatureInfo(True),
            "dense paraphrasing": GuiFeatureInfo(True),
            "pose": GuiFeatureInfo(False),
            "prop": GuiFeatureInfo(True),
            "move": GuiFeatureInfo(True),
            "common ground": GuiFeatureInfo(True),
        }

        self.output_frame = np.zeros((500, 500, 3), dtype=np.float32)
        self.image_label = Label(self.root)
        self.image_label.pack(side=LEFT)
        self._update_image_callback()

        button_frame = Frame(self.root).pack(side=RIGHT)
        for i, (text, var) in enumerate(self.feature_vars.items()):
            Checkbutton(
                button_frame,
                text=text,
                variable=var.var,
                onvalue=1,
                offvalue=0,
            ).pack()
        self._update_vars_callback()

        self._running = False

    def _update_vars_callback(self):
        # We need to update the stored variable values
        # so they can be accessed from a non-main thread
        for var in self.feature_vars.values():
            var.var_val = bool(var.var.get())
        self.root.after(100, self._update_vars_callback)

    def _update_image_callback(self):
        img = Image.fromarray(self.output_frame.astype(np.uint8))
        imgtk = ImageTk.PhotoImage(image=img)
        self.image_label.imgtk = imgtk
        self.image_label.configure(image=imgtk)
        self.image_label.after(1, self._update_image_callback)

    @property
    def running(self):
        return self._running

    def mainloop(self):
        self._running = True
        self.root.mainloop()
        self._running = False

    def close(self):
        # needs to be called like this so it can work
        # from non-main threads
        self.root.after(0, self.root.destroy)

    def new_image(self, frame):
        self.output_frame = frame

    def feature_active(self, name):
        assert name in self.feature_vars, "Feature button does not exist"
        return self.feature_vars[name].var_val
