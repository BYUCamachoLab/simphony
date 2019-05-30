"""
monte_carlo_simulation.py

Author:
    Sequoia Ploeg

Dependencies:
- tkinter
- SiEPIC.ann.simulation
- os

This file mainly provides the GUI for running monte carlo simulations. It 
creates a MCSimulation object and runs it, provides various parameters from its
GUI. Presently, the MCSimulation object handles displaying the results itself.
"""

import os
import tkinter as tk
from tkinter import filedialog

from .simulation import monte_carlo_sim

DEF_NUM_SIMS = 10
DEF_MU_WIDTH = 0.5
DEF_SIGMA_WIDTH = 0.005
DEF_MU_THICKNESS = 0.22
DEF_SIGMA_THICKNESS = 0.002
DEF_MU_LENGTH = 0
DEF_SIGMA_LENGTH = 0
DEF_DPIN = 1
DEF_DPOUT = 0
DEF_SAVEDATA = True
DEF_DISPTIME = True
DEF_FILENAME = "monte_carlo.mat"

class MonteCarloGUI(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.withdraw()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.title("Monte Carlo Simulation")

        padx = 5
        pady = 5

        bbox = tk.Frame(padx=padx, pady=pady)
        bbox.pack()

        sim_group = tk.LabelFrame(bbox, text="Simulation", padx=padx, pady=pady)
        sim_group.pack(fill=tk.BOTH)
        tk.Label(sim_group, text="Number of simulations: ").pack(side=tk.LEFT)
        self.num_sims = tk.Entry(sim_group)
        self.num_sims.pack(side=tk.LEFT)
        self.num_sims.insert(0, str(DEF_NUM_SIMS))
        
        var_group = tk.Frame(bbox)
        var_group.pack(fill=tk.BOTH)

        stdev_group = tk.LabelFrame(var_group, text="Standard Deviations", padx=padx, pady=pady)
        stdev_group.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        tk.Label(stdev_group, text="Width:").grid(row=0, column=0, sticky=tk.W)
        self.sigma_width = tk.Entry(stdev_group)
        self.sigma_width.grid(row=0, column=1)
        self.sigma_width.insert(0, str(DEF_SIGMA_WIDTH))
        tk.Label(stdev_group, text="Thickness:").grid(row=1, column=0, sticky=tk.W)
        self.sigma_thickness = tk.Entry(stdev_group)
        self.sigma_thickness.grid(row=1, column=1)
        self.sigma_thickness.insert(0, str(DEF_SIGMA_THICKNESS))
        tk.Label(stdev_group, text="Length:").grid(row=2, column=0, sticky=tk.W)
        self.sigma_length = tk.Entry(stdev_group)
        self.sigma_length.grid(row=2, column=1)
        self.sigma_length.insert(0, str(DEF_SIGMA_LENGTH))

        mean_group = tk.LabelFrame(var_group, text="Means", padx=padx, pady=pady)
        mean_group.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        tk.Label(mean_group, text="Width (um):").grid(row=0, column=0, sticky=tk.W)
        self.mean_width = tk.Entry(mean_group)
        self.mean_width.grid(row=0, column=1)
        self.mean_width.insert(0, str(DEF_MU_WIDTH))
        tk.Label(mean_group, text="Thickness (um):").grid(row=1, column=0, sticky=tk.W)
        self.mean_thickness = tk.Entry(mean_group)
        self.mean_thickness.grid(row=1, column=1)
        self.mean_thickness.insert(0, str(DEF_MU_THICKNESS))
        tk.Label(mean_group, text="Length (um):").grid(row=2, column=0, sticky=tk.W)
        self.mean_length = tk.Entry(mean_group)
        self.mean_length.grid(row=2, column=1)
        self.mean_length.insert(0, str(DEF_MU_LENGTH))

        io_group = tk.LabelFrame(bbox, text="I/O (0-indexed)", padx=padx, pady=pady)
        io_group.pack(fill=tk.BOTH, expand=1)
        tk.Label(io_group, text="Input port:").grid(row=0, column=0)
        self.in_port = tk.Entry(io_group)
        self.in_port.grid(row=0, column=1)
        self.in_port.insert(0, str(DEF_DPIN))
        tk.Label(io_group, text="Output port:").grid(row=0, column=2)
        self.out_port = tk.Entry(io_group)
        self.out_port.grid(row=0, column=3)
        self.out_port.insert(0, str(DEF_DPOUT))

        extras_group = tk.LabelFrame(bbox, text="Extras", padx=padx, pady=pady)
        extras_group.pack(fill=tk.BOTH, expand=1)
        save_group = tk.Frame(extras_group)
        save_group.pack(fill=tk.BOTH, expand=1)
        self.savefile = tk.IntVar()
        tk.Checkbutton(save_group, text="Save output?", variable=self.savefile).grid(row=0, column=0)
        tk.Label(save_group, text="     Location:").grid(row=0, column=1)
        self.savefilename = tk.Entry(save_group, width=30)
        self.savefilename.grid(row=0, column=2, padx=5)
        self.savefilename.insert(0, os.path.join(os.path.expanduser('~'), DEF_FILENAME))
        tk.Button(save_group, text="Browse", command=self.saveasdialog).grid(row=0, column=3)
        self.show_time = tk.IntVar()
        tk.Checkbutton(extras_group, text="Show simulation time", variable=self.show_time).pack(side=tk.LEFT)

        out_group = tk.LabelFrame(bbox, text="Output", padx=padx, pady=pady)
        out_group.pack(fill=tk.BOTH, expand=1)
        self.out_text = tk.Text(out_group, width=50, height=5)
        self.out_text.bind("<Key>", lambda e: "break")
        self.out_text.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)

        run_group = tk.Frame(bbox, padx=padx, pady=pady)
        run_group.pack(fill=tk.BOTH, expand=1)
        self.run_sim = tk.Button(run_group, text="Run Simulation", command=self.run_simulation)
        self.run_sim.pack(side=tk.RIGHT)

        self.after(0, self.deiconify)
        
    def on_closing(self):
        self.withdraw()
        self.quit()
        self.destroy()

    def saveasdialog(self):
        filename = filedialog.asksaveasfilename(initialdir = os.path.expanduser('~'),title = "Select file",filetypes = (("MATLAB Files","*.mat"),("All files","*.*")))
        if filename:
            self.savefilename.insert(0, filename)

    # TODO: Select input/output ports AFTER simulation is run, and plotting them.
    def run_simulation(self):
        sims = int(self.num_sims.get())
        s_width = float(self.sigma_width.get())
        s_thick = float(self.sigma_thickness.get())
        s_length = float(self.sigma_length.get())
        m_width = float(self.mean_width.get())
        m_thick = float(self.mean_thickness.get())
        m_length = float(self.mean_length.get())
        iport = int(self.in_port.get())
        oport = int(self.out_port.get())
        save = self.savefile.get() == 1
        location = self.savefilename
        timeit = self.show_time.get() == 1

        monte_carlo_sim(num_sims=sims, mu_width=m_width, sigma_width=s_width, mu_thickness=m_thick, sigma_thickness=s_thick, mu_length=m_length,
            sigma_length=s_length, dpin=iport, dpout=oport, saveData=save, filename=location, dispTime=timeit, printer=self.print_to_output)

    def print_to_output(self, message):
        self.out_text.insert(tk.END, message + "\n")

def monte_carlo_simulation():
    app = MonteCarloGUI()
    app.mainloop()
  
if __name__ == "__main__":
    monte_carlo_simulation()
