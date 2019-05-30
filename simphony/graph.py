"""
graph.py : Tkinter Graph Wrapper

Author: 
    Sequoia Ploeg

Dependencies:
- tkinter
- matplotlib
- numpy

This module allows tkinter graphs to be easily created from a parent tkinter 
object in their own separate windows.
"""

import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import numpy as np
import scipy.io as sio
import os # For filedialog to start in user's /~ instead of /.


class ListSelectDeleteDialog:
    """Opens a dialog window presenting a list of items passed in as a 
    parameter.

    Delete button removes items from the list. Deleted items are the return 
    value.

    Usage: 
    ListSelectDeleteDialog(master: tk.Toplevel).askdeletelist(startlist: list)
    """

    def __init__(self, master: tk.Toplevel):
        self.master = master

    def askdeletelist(self, startlist: list):
        self.listbox = tk.Listbox(self.master)
        self.listbox.pack()
        self.deleted = []
        self.leftover = startlist
        self.success = True
        self.master.protocol("WM_DELETE_WINDOW", self._cancel)

        for item in startlist:
            self.listbox.insert(tk.END, item)

        frame_okCancel = tk.Frame(self.master)
        deleteBtn = tk.Button(frame_okCancel, text="Delete", command=lambda: self._delete(self.listbox))
        deleteBtn.grid(column=0, row=0)
        okBtn = tk.Button(frame_okCancel, text="Ok", command=self._on_close)
        okBtn.grid(column=1, row=0)
        cancelBtn = tk.Button(frame_okCancel, text="Cancel", command=self._cancel)
        cancelBtn.grid(column=2, row=0)
        frame_okCancel.pack()

        self.master.grab_set()
        self.master.wait_window(self.master)
        if self.success:
            return self.deleted
        else:
            return None

    def _delete(self, listbox: tk.Listbox):
        selection = listbox.curselection()[0]
        self.deleted.append(self.leftover.pop(selection))
        listbox.delete(selection)        

    def _cancel(self):
        self.success = False
        self._on_close()

    def _on_close(self):
        self.master.destroy()


class ListSelectRenameDialog:
    """Opens a dialog window presenting a list of items passed in as a 
    parameter.

    Rename button opens a text dialog to rename an item from the list. Returns
    a tuple: the original name and the new name.

    Usage: 
    ListSelectRenameDialog(master: tk.Toplevel).askrenamelist(startlist: list)
    """

    def __init__(self, master: tk.Toplevel):
        self.master = master

    def askrenamelist(self, startlist: list):
        self.listbox = tk.Listbox(self.master)
        self.listbox.pack()
        self.leftover = startlist
        self.success = True
        self.master.protocol("WM_DELETE_WINDOW", self._cancel)

        for item in startlist:
            self.listbox.insert(tk.END, item)

        frame_okCancel = tk.Frame(self.master)
        deleteBtn = tk.Button(frame_okCancel, text="Rename", command=lambda: self._rename(self.listbox))
        deleteBtn.grid(column=0, row=0)
        cancelBtn = tk.Button(frame_okCancel, text="Cancel", command=self._cancel)
        cancelBtn.grid(column=1, row=0)
        frame_okCancel.pack()

        self.master.grab_set()
        self.master.wait_window(self.master)
        if self.success:
            return self.original, self.final
        else:
            return None, None

    def _rename(self, listbox: tk.Listbox):
        selection = listbox.curselection()[0]
        self.original = self.leftover.pop(selection)
        self.master.withdraw()
        self.final = simpledialog.askstring("Rename", "Enter new name for line: " + self.original)
        self.master.destroy()

    def _cancel(self):
        self.success = False
        self._on_close()

    def _on_close(self):
        self.master.destroy()

class MenuItem:
    def __init__(self, label=None, callback=None):
        self.label = label
        self.callback = callback

class MenuGroup:
    """
    TODO: Implement such that classing importing graph don't have to create 
    their own dictionaries but can just create a MenuGroup and pass that in to
    the Graph.
    """

    def __init__(self):
        self.menuitems = []

    def add(self, item: MenuItem):
        self.menuitems.append(item)

class DataSet:
    """
    The DataSet class is used to allow the Graph to conveniently store and 
    access multiple lines.

    This class is not intended to be used by any other class besides Graph.
    Stored within the DataSet class is the xdata, the ydata, and an optional 
    name. Graph generates a DataSet object when xdata, ydata, and an optional 
    name is passed to its plot function. Upon graphing the object, it stores a
    reference to the axis line object. Graph stores a list of DataSet objects.
    In this way, it can iterate through all the axis line objects to delete or 
    otherwise modify them.
    """

    def __init__(self, x: np.array, y: np.array, name: str = None):
        """Stores the x and y values of the plot, as well as an (optional) 
        name for the DataSet"""

        self.x = x
        self.y = y
        self.name = name

    def setObjectID(self, id):
        """The objectID member is intended to hold a reference to a matplotlib
        Axes line."""

        self.objectID = id[0]

    def getObjectID(self):
        """The objectID member holds a reference to a matplotlib Axes line."""

        return self.objectID

    def to_mat(self):
        xdata_name = "x_" + self.name.replace(" ", "_")
        ydata_name = "y_" + self.name.replace(" ", "_")
        return {xdata_name: self.x, ydata_name: self.y}

    def __str__(self):
        return self.name

class Graph:
    """
    The Graph class presents a tkinter interface for Matplotlib plots.

    It requires a Toplevel tkinter object for initialization. 
    
    Attributes
    ----------
    default_title : str
        The default title for a new or reset window and plot.
    master : tk.Toplevel
        The Toplevel tkinter object that must exist for Graph to exist.
    menubar : tk.menu
        The tkinter menubar object for user operations.
    fig : matplotlib.figure.Figure
        The matplotlib figure object contained within Graph.
    canvas : matplotlib.backends.backend_tkagg.FigureCanvasTkAgg
        The matplotlib drawing canvas object contained within Graph.
    toolbar: matplotlib.backends.backend_tkagg.NavigationToolbar2Tk
        The matplotlib toolbar object contained within Graph.
    ax : matplotlib.axes.Axes
        The matplotlib axis upon which line and legend operations are 
        performed.
    hasLegend : tk.BooleanVar
        A state variable used to maintain the legend's existence state between
        graph updates (default is false)
    lines : DataSet[]
        A list of DataSet objects containing the data points and their formal 
        names.
    line_counter : int
        An autoincrementing counter to assign numeric names to lines plotted 
        without a specified name.

    Methods
    -------
    plot(x=None, y=None, name=None)
        Plots x and y data on a Graph.
    clear(value)
        Deletes a stored DataSet value from the graph's self.lines DataSet 
        objects list and removes its line and legend from the plot.
    legend(include=None)
        Updates the legend's values and maintains its state. Can be used to 
        activate/deactivate the legend.
    linewidth(size)
        Changes the linewidth of all plotted lines.
    title(title)
        Sets the window title and the graph title.
    xlabel(xlabel)
        Sets the x-axis label.
    ylabel(ylabel)
        Sets the y-axis label.
    """

    # Default window and plot title
    default_title = "Graph"

    #########################################################################
    #                                                                       #
    #                 GRAPH INITIALIZATION FUNCTIONS                        #
    #                                                                       #
    #########################################################################

    def __init__(self, parent: tk.Toplevel, window_title=None, additional_menus=None, onCloseCallback=None):
        # The master tk object
        self.parent = parent
        self.master = tk.Toplevel(parent)
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        # self.master = parent
        if window_title == None:
            self.master.title(Graph.default_title)
        else:
            self.master.title(window_title)
        self.menubar = tk.Menu(self.master)
        self.hasLegend = tk.BooleanVar()

        # The only real objects we'll need to interact with to plot and unplot
        self.fig = Figure(figsize=(5, 4), dpi=100)

        # Objects needed simply for the sake of embedding the graph in tk
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.master)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas.mpl_connect("key_press_event", self.on_key_press)
        
        # Setup the plot area, stored lines, and setup the menu now that all 
        # variables exist.
        self.reset()
        self.init_menu(additional_menus=additional_menus)
        self.master.config(menu=self.menubar)
        self.onCloseCallback = onCloseCallback

    def reset(self):
        """Clears the figure, adds a new axis, resets the title, and clears all
        stored DataSet lines."""

        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.hasLegend.set(False)
        self.title(Graph.default_title)
        # Lines is a list of DataSet objects. The user should take care to make
        # DataSet names unique, as there is no error checking done by Graph. 
        # If a DataSet line is deleted by its formal name, Graph will delete the
        # first line in the list that matches the name.
        self.lines = {}
        self.line_counter = 1

    def raise_window(self):
        self.master.attributes('-topmost', True)
        self.master.attributes('-topmost', False)

    def on_closing(self):
        if self.onCloseCallback is not None:
            self.onCloseCallback()
        self.master.destroy()

    #########################################################################
    #                                                                       #
    #                     MENU BUTTON IMPLEMENTATIONS                       #
    #                                                                       #
    #########################################################################

    def init_menu(self, additional_menus=None):
        filemenu = tk.Menu(self.menubar, tearoff=0)
        filemenu.add_command(label="New", command=self.filemenu_New)
        #filemenu.add_command(label="Open", command=self.filemenu_Open)
        filemenu.add_command(label="Close", command=self.filemenu_Close)
        filemenu.add_separator()
        filemenu.add_command(label="Export to...", command=self.filemenu_Export)
        #filemenu.add_separator()
        #filemenu.add_command(label="Print")
        self.menubar.add_cascade(label="File", menu=filemenu)

        editmenu = tk.Menu(self.menubar, tearoff=0)
        editmenu.add_command(label="Delete line...", command=self.editmenu_DeleteLine)
        editmenu.add_command(label="Rename line...", command=self.editmenu_RenameLine)
        linewidth_submenu = tk.Menu(editmenu, tearoff=0)
        linewidth_submenu.add_command(label="Ultrathin", command=lambda: self.linewidth(0.5))
        linewidth_submenu.add_command(label="Thin", command=lambda: self.linewidth(1.0))
        linewidth_submenu.add_command(label="Default", command=lambda: self.linewidth(1.5))
        linewidth_submenu.add_command(label="Thick", command=lambda: self.linewidth(2.0))
        linewidth_submenu.add_command(label="Ultrathick", command=lambda: self.linewidth(2.5))
        editmenu.add_cascade(label="Set linewidth", menu=linewidth_submenu)
        windowsize_submenu = tk.Menu(editmenu, tearoff=0)
        windowsize_submenu.add_command(label="Small")
        windowsize_submenu.add_command(label="Default")
        windowsize_submenu.add_command(label="Large")
        editmenu.add_cascade(label="Resize window", menu=windowsize_submenu)
        editmenu.add_command(label="Tight layout", command=self.tight_layout)
        self.menubar.add_cascade(label="Edit", menu=editmenu)

        insertmenu = tk.Menu(self.menubar, tearoff=0)
        insertmenu.add_command(label="X Label", command=self.insertmenu_XLabel)
        insertmenu.add_command(label="Y Label", command=self.insertmenu_YLabel)
        insertmenu.add_command(label="Title", command=self.insertmenu_Title)
        insertmenu.add_separator()
        insertmenu.add_checkbutton(label="Legend", onvalue=True, offvalue=False, variable=self.hasLegend, command=self.legend)
        self.menubar.add_cascade(label="Insert", menu=insertmenu)

        # An "additional_menu" is a dictionary where each key is the name of 
        # the cascade to be added, and its corresponding value is a list of 
        # MenuItem objects (each as a name and a callback function).
        if additional_menus != None:
            for cascade in additional_menus:
                cascade_menu = tk.Menu(self.menubar, tearoff=0)
                commands = additional_menus[cascade]
                for item in commands:
                    cascade_menu.add_command(label=item.label, command=item.callback)
                self.menubar.add_cascade(label=cascade, menu=cascade_menu)

        helpmenu = tk.Menu(self.menubar, tearoff=0)
        helpmenu.add_command(label="About")
        helpmenu.add_command(label="Keyboard shortcuts")
        helpmenu.add_command(label="LaTeX")
        self.menubar.add_cascade(label="Help", menu=helpmenu)

    def filemenu_New(self):
        Graph(self.parent)
        #Graph(tk.Toplevel())

    def filemenu_Open(self):
        options = {}
        options['initialdir'] = os.path.expanduser('~')
        options['parent'] = self.master
        f = filedialog.askopenfilename(**options)
        print("TODO")        

    def filemenu_Export(self):
        """
        If saved as .npz, note that it's a dictionary of lines being saved.
        On loading, you must specify allow_pickle=True.
        To access items in the dictionary, first take them out of the array.

        Example:
        --------
        a = np.load('sample.npz')
        contents = a['lines'].item()
        """
        line_dict = {}
        for line in self.lines.values():
            for name, arr in line.to_mat().items():
                line_dict[name] = arr
        fileTypes = [("MATLAB file","*.mat"), ("NumPy file","*.npz")]
        options = {}
        options['initialdir'] = os.path.expanduser('~')
        options['filetypes'] = fileTypes
        options['parent'] = self.master
        filename = filedialog.asksaveasfilename(**options)
        if filename:
            _, ext = os.path.splitext(filename)
            if ext == ".mat":
                sio.savemat(filename, line_dict)
            elif ext == ".npz":
                np.savez(filename, lines=line_dict)
            
    def editmenu_DeleteLine(self):
        linelist = []
        for line in self.lines.values():
            linelist.append(line.name)
        child_window = tk.Toplevel(self.master)
        delete = ListSelectDeleteDialog(child_window).askdeletelist(linelist)
        if delete != None:
            for item in delete:
                self.clear(item)

    def editmenu_RenameLine(self):
        linelist = []
        for line in self.lines.values():
            linelist.append(line.name)
        child_window = tk.Toplevel(self.master)
        orig, final = ListSelectRenameDialog(child_window).askrenamelist(linelist)
        if final:
            line = self.lines.pop(orig)
            line.name = final
            self.lines[final] = line
            self.legend()

    def insertmenu_XLabel(self):
        label = simpledialog.askstring("Edit X Label","Wrap LaTeX in $", parent=self.master, initialvalue=self.ax.get_xlabel())
        if label is not None:
            self.xlabel(label)

    def insertmenu_YLabel(self):
        label = simpledialog.askstring("Edit Y Label","Wrap LaTeX in $", parent=self.master, initialvalue=self.ax.get_ylabel())
        if label is not None:
            self.ylabel(label)
            
    def insertmenu_Title(self):
        label = simpledialog.askstring("Edit Title","Wrap LaTeX in $", parent=self.master, initialvalue=self.ax.get_title())
        if label is not None:
            self.title(label)

    def on_key_press(self, event):
        """Registers a key press event (default matplotlib keybindings are 
        implemented).

        Parameters
        ----------
        event : Event
            An event like a key press that is passed to the matplotlib key 
            press handler.
        """

        #print("you pressed {}".format(event.key))
        key_press_handler(event, self.canvas, self.toolbar)

    def filemenu_Close(self):
        """Destroys the child tkinter object upon closing."""

        self.on_closing()

    #########################################################################
    #                                                                       #
    #               PLOTTING AND AXIS MANIPULATION FUNCTIONS                #
    #                                                                       #
    #########################################################################

    def plot(self, x: np.array, y: np.array, name: str = None):
        """Plots x and y data on a Graph.

        Parameters
        ----------
        x : np.array
            The x axis values
        y : np.array
            The y axis values
        name : str, optional
            The name for this line (default = None). Line names are required 
            to be unique, and Graph raises a ValueError if the unique name 
            constraint is not satisfied.

        Raises
        ------
        ValueError
            If the shapes of x or y are different.
        """

        if x.shape == y.shape and name not in self.lines:
            if name == None:
                name = "Line " + str(self.line_counter)
                self.line_counter += 1
            dataset = DataSet(x, y, name)
            dataset.setObjectID(self.ax.plot(dataset.x, dataset.y))
            self.lines[dataset.name] = dataset
            self.legend()
            self.canvas.draw()
        else:
            if x.shape != y.shape:
                raise ValueError("x and y array shapes do not match.")
            if(name in self.lines):
                raise ValueError("line with specified name already exists (unique constraint failed).")
            raise ValueError("Error in required arguments for plotting.")

    # Much help derived from https://stackoverflow.com/questions/4981815/how-to-remove-lines-in-a-matplotlib-plot
    def clear(self, value=None):
        """Deletes a stored DataSet value from the graph's self.lines DataSet 
        objects list and removes its line and legend from the plot.

        The user should take care to make DataSet names unique, as Graph will 
        raise a value error if attempts are made to plot a duplicate.

        Parameters
        ----------
        value : str
            The line with the specified name is deleted (no effect if it 
            doesn't exist). If None, all lines are cleared from the graph.
        """

        if value is not None:
            for line in self.lines.values():
                if line.name == value:
                    self.ax.lines.remove(line.getObjectID())
                    self.lines.pop(line.name)
                    break
        else:
            self.reset()
        # Remove the lines that have been cleared from the legend.
        self.legend()
        self.canvas.draw()

    def legend(self, include: bool = None):
        """Updates the legend's values and maintains its state.

        Parameters
        ----------
        include : bool, optional
            If not specified, default behavior is to maintain the legend's 
            present state (self.hasLegend).
            If true, a draggable legend is placed onto the Graph.
            If false, the legend is turned off.
        """
        
        if include == None:
            if self.hasLegend.get() == True:
                include = True
            else:
                include = False
            
        if include == True:
            labels = []
            for line in self.lines.values():
                labels.append(line.name)
            self.ax.legend(labels).set_draggable(True)
            self.hasLegend.set(True)
        else:
            self.ax.legend().remove() # This line complains to the console if no legend exists when it's removed
            self.hasLegend.set(False)
        self.canvas.draw()
    
    def linewidth(self, size: float):
        """Changes the linewidth of all plotted lines.

        Some suggested line sizes:
        Ultrathin   Thin    Default     Thick   Ultrathick      Custom
        0.5         1.0     1.5         2.0     2.5             _._

        Parameters
        ----------
        size : float
            A floating point value of the thickness to use.
        """
        for line in self.ax.lines:
            line.set_linewidth(size)
        self.canvas.draw()

    def title(self, title: str):
        """Sets the window title and the graph title.
        
        Parameters
        ----------
        title : str
            The graph and window Title.
        """

        #self.master.title(title)
        self.ax.set_title(title)
        self.canvas.draw()

    def xlabel(self, xlabel: str):
        """Sets the x-axis label.

        Parameters
        ----------
        xlabel : str
            The x-axis label.
        """

        self.ax.set_xlabel(xlabel)
        self.canvas.draw()

    def ylabel(self, ylabel: str):
        """Sets the y-axis label.

        Parameters
        ----------
        ylabel : str
            The y-axis label.
        """
        self.ax.set_ylabel(ylabel)
        self.canvas.draw()

    def get_lines(self):
        """Returns all lines (DataSet) objects stored within the graph
        Since these contain references to the line objects, they can be
        altered and updated. This allows externally implemented menus to
        access the data within the graph and perform operations on it, 
        e.g. converting the x-axis from frequency to wavelength.
        """

        return self.lines

    def refresh(self):
        """Refreshes the axis plot. If data within the lines has changed,
        the lines are redrawn and the plot rescaled to fit them.
        """

        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    def tight_layout(self):
        """Applies the tight layout to the figure.
        """
        self.fig.tight_layout()
        self.canvas.draw()

#########################################################################
#                                                                       #
#              TEST FUNCTIONS IF CODE IS RUN AS A SCRIPT                #
#                                                                       #
#########################################################################

def sayHello():
    print("Hello, world!")

def sayGoodbye():
    print("Goodbye, world!")

def greet():
    print("What's up?")

def snooze():
    print("Snoozing for 15 minutes")

def alarm():
    print("annoying sounds!")

def sleep():
    print("Time for bed!")

def test(): 
    root = tk.Tk()

    talkingMenu = [MenuItem("Hello", sayHello), MenuItem("Goodbye", sayGoodbye), MenuItem("Greeting", greet)]
    clockMenu = [MenuItem("Snooze", snooze), MenuItem("Alarm", alarm), MenuItem("Sleep", sleep)]
    newmenus = {"Chatterbox": talkingMenu, "Clock": clockMenu}

    #app = Graph(tk.Toplevel(root))
    app = Graph(root, additional_menus=newmenus)

    t1 = np.arange(0, 3, .01)
    y1 = 2 * np.sin(2 * np.pi * t1)
    app.plot(t1, y1, "Sine")

    t2 = np.arange(0, 6, .01)
    y2 = 5 * np.cos(2 * np.pi * t2)
    app.plot(t2, y2, "Cosine")

    app.title("Frequency Response")
    app.xlabel("Time (s)")
    app.ylabel("Amplitude")
    #app.legend(True)

    root.mainloop()

def stretch(x):
    return -x * 5

def test_conversion():
    root = tk.Tk()
    app = Graph(root)

    t1 = np.arange(0, 3, .01)
    y1 = 2 * np.sin(2 * np.pi * t1)
    app.plot(t1, y1, "Sine")

    t2 = np.arange(0, 6, .01)
    y2 = 5 * np.cos(2 * np.pi * t2)
    app.plot(t2, y2, "Cosine")

    app.title("Frequency Response")
    app.xlabel("Time (s)")
    app.ylabel("Amplitude")

    lines = app.get_lines()
    for line in lines.values():
        x = line.objectID.get_xdata()
        line.objectID.set_xdata(stretch(x))
    app.refresh()

    root.mainloop()

if __name__=="__main__":
    test()
    test_conversion()