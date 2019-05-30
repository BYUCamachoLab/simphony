import tkinter as tk
from tkinter import ttk
from .models.components import Component
from importlib import import_module

# TODO: Make it an on-change to save, rather than a on-close
class SettingsGUI(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.withdraw()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.title("Settings")

        padx = 5
        pady = 5

        bbox = tk.Frame(padx=padx, pady=pady)
        bbox.pack()

        notebook = ttk.Notebook(bbox)
        notebook.pack(fill=tk.BOTH)
        
        models = tk.Frame(notebook, padx=padx, pady=pady)
        mod = import_module('models')
        Comp = mod.components.Component
        i = 0
        self.references = {}
        self.selections = {}
        for class_ in Comp.__subclasses__():
            tk.Label(models, text=class_.__name__, justify=tk.LEFT, anchor='w').grid(row=i, column=0, sticky='ew')
            self.references[class_.__name__] = class_
            self.selections[class_.__name__] = tk.StringVar(self)
            self.selections[class_.__name__].set(class_._selected_model)
            om = tk.OptionMenu(models, self.selections[class_.__name__], *class_._simulation_models.keys())
            om.configure(width=25, anchor='w')
            om.grid(row=i, column=1, padx=padx, pady=pady)
            i += 1
        notebook.add(models, text="Models")

        self.after(0, self.deiconify)

    def on_closing(self):
        for key, val in self.selections.items():
            if self.references[key]._selected_model != val.get():
                self.references[key].set_model(val.get())

        self.withdraw()
        self.quit()
        self.destroy()

def settings_gui():
    app = SettingsGUI()
    app.mainloop()

if __name__ == "__main__":
    settings_gui()