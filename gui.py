from tkinter import Tk, IntVar, Checkbutton

class Gui:
    def __init__(self):

        self.root = Tk()
        # self.root.geometry('350x200')
        self.root.title("Output Options")

        self.vars = {
                "gesture": IntVar(value=1),
                "objects": IntVar(value=1),
                "gaze": IntVar(value=1),
                "asr": IntVar(value=1),
                "pose": IntVar(value=1),
                "prop": IntVar(value=1),
                "move": IntVar(value=1),
                }


    def create_buttons(self):
        for i,j in self.vars.items():
            self._make_button(i, j).pack()

    def _make_button(self, text, var):
        return Checkbutton(self.root, text=text, variable=var, onvalue=1, offvalue=0, height=2, width=10)

    def update(self):
        self.root.update()

    def should_process(self, var):
        return self.vars[var].get()
