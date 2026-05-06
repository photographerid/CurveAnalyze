
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
from pathlib import Path
import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import brentq, curve_fit

APP_DIR = Path(__file__).resolve().parent
THREE_PL_D = 0.0
APP_NAME = "ELISA Data Analyzer"
MAX_RECENT_PROJECTS = 5


def get_resource_dir():
    """Return the directory containing bundled app resources."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", APP_DIR))
    return APP_DIR


def get_user_data_dir():
    """Return a writable per-user directory for persistent app state."""
    if sys.platform.startswith("win"):
        base_dir = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base_dir = Path.home() / "Library" / "Application Support"
    else:
        base_dir = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))

    return base_dir / "CurveAnalyze"


RESOURCE_DIR = get_resource_dir()
USER_DATA_DIR = get_user_data_dir()
RECENT_PROJECTS_FILE = USER_DATA_DIR / "recent_projects.json"

# ========== MATHEMATICAL FUNCTIONS ==========
def four_param_logistic(x, A, B, C, D):
    """4-Parameter Logistic Function"""
    with np.errstate(invalid='ignore', divide='ignore'):
        return D + (A - D) / (1 + (x / C)**B)

def three_param_logistic(x, A, B, C, D=THREE_PL_D):
    """3-Parameter Logistic Function with D fixed at 0"""
    with np.errstate(invalid='ignore', divide='ignore'):
        return D + (A - D) / (1 + (C / x)**B)

def calculate_r_squared(y_true, y_pred):
    """Calculate R-squared value"""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


def normalize_column_name(name):
    """Normalize column names for loose matching."""
    return ''.join(ch.lower() for ch in str(name) if ch.isalnum())

# ========== STYLE MANAGER ==========
class StyleManager:
    def __init__(self):
        self.bg_color = "#eef2f7"
        self.card_color = "#ffffff"
        self.panel_color = "#f8fafc"
        self.border_color = "#d5dde8"
        self.primary_color = "#315c8f"
        self.primary_hover_color = "#284d79"
        self.accent_color = "#2a8f6a"
        self.accent_hover_color = "#217456"
        self.error_color = "#c94c4c"
        self.error_hover_color = "#ab3d3d"
        self.text_color = "#1f2d3d"
        self.muted_text_color = "#5f6f82"
        self.input_bg = "#fbfcfe"
        self.font_family = "Segoe UI"
        
    def configure_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # General styles
        style.configure('.', 
                      background=self.bg_color,
                      foreground=self.text_color,
                      font=(self.font_family, 10))
        
        # Frame styles
        style.configure('Card.TFrame', 
                      background=self.card_color,
                      borderwidth=1,
                      relief='flat',
                      bordercolor=self.border_color)

        style.configure('Section.TFrame',
                      background=self.panel_color,
                      borderwidth=1,
                      relief='flat',
                      bordercolor=self.border_color)
        
        # Button styles
        style.configure('Primary.TButton',
                      foreground='white',
                      background=self.primary_color,
                      borderwidth=1,
                      relief='raised',
                      bordercolor="#5d7ea8",
                      lightcolor="#5d7ea8",
                      darkcolor=self.primary_hover_color,
                      focusthickness=0,
                      padding=(16, 6))
        style.map('Primary.TButton',
                  foreground=[('disabled', '#f0f0f0'), ('active', 'white'), ('pressed', 'white')],
                  background=[('disabled', '#b8c4d6'), ('active', self.primary_hover_color), ('pressed', self.primary_hover_color)],
                  bordercolor=[('disabled', '#b8c4d6'), ('active', self.primary_hover_color), ('pressed', self.primary_hover_color)],
                  lightcolor=[('disabled', '#b8c4d6'), ('active', self.primary_hover_color), ('pressed', self.primary_hover_color)],
                  darkcolor=[('disabled', '#b8c4d6'), ('active', self.primary_hover_color), ('pressed', self.primary_hover_color)])
        
        style.configure('Accent.TButton',
                      foreground='white',
                      background=self.accent_color,
                      borderwidth=1,
                      relief='raised',
                      bordercolor="#56a586",
                      lightcolor="#56a586",
                      darkcolor=self.accent_hover_color,
                      focusthickness=0,
                      padding=(16, 6))
        style.map('Accent.TButton',
                  foreground=[('disabled', '#f0f0f0'), ('active', 'white'), ('pressed', 'white')],
                  background=[('disabled', '#b7ddc0'), ('active', self.accent_hover_color), ('pressed', self.accent_hover_color)],
                  bordercolor=[('disabled', '#b7ddc0'), ('active', self.accent_hover_color), ('pressed', self.accent_hover_color)],
                  lightcolor=[('disabled', '#b7ddc0'), ('active', self.accent_hover_color), ('pressed', self.accent_hover_color)],
                  darkcolor=[('disabled', '#b7ddc0'), ('active', self.accent_hover_color), ('pressed', self.accent_hover_color)])
        
        style.configure('Danger.TButton',
                      foreground='white',
                      background=self.error_color,
                      borderwidth=1,
                      relief='raised',
                      bordercolor="#d77777",
                      lightcolor="#d77777",
                      darkcolor=self.error_hover_color,
                      focusthickness=0,
                      padding=(16, 6))
        style.map('Danger.TButton',
                  foreground=[('disabled', '#f0f0f0'), ('active', 'white'), ('pressed', 'white')],
                  background=[('disabled', '#e5b8bf'), ('active', self.error_hover_color), ('pressed', self.error_hover_color)],
                  bordercolor=[('disabled', '#e5b8bf'), ('active', self.error_hover_color), ('pressed', self.error_hover_color)],
                  lightcolor=[('disabled', '#e5b8bf'), ('active', self.error_hover_color), ('pressed', self.error_hover_color)],
                  darkcolor=[('disabled', '#e5b8bf'), ('active', self.error_hover_color), ('pressed', self.error_hover_color)])
        
        # Label styles
        style.configure('Title.TLabel',
                      background=self.card_color,
                      font=(self.font_family, 13, 'bold'),
                      foreground=self.primary_color)
        
        style.configure('Subtitle.TLabel',
                      background=self.card_color,
                      font=(self.font_family, 10),
                      foreground=self.muted_text_color)

        style.configure('SectionTitle.TLabel',
                      background=self.panel_color,
                      font=(self.font_family, 10, 'bold'),
                      foreground=self.primary_color)

        style.configure('TLabelframe',
                      background=self.panel_color,
                      borderwidth=1,
                      relief='solid',
                      bordercolor=self.border_color,
                      padding=10)
        style.configure('TLabelframe.Label',
                      background=self.panel_color,
                      font=(self.font_family, 10, 'bold'),
                      foreground=self.primary_color)

        style.configure('TNotebook',
                      background=self.card_color,
                      borderwidth=0,
                      tabmargins=(4, 4, 4, 0))
        style.configure('TNotebook.Tab',
                      background="#dfe7f1",
                      foreground=self.muted_text_color,
                      padding=(16, 8),
                      font=(self.font_family, 10, 'bold'))
        style.map('TNotebook.Tab',
                  background=[('selected', self.card_color), ('active', '#e9eef5')],
                  foreground=[('selected', self.primary_color), ('active', self.text_color)])

        style.configure('TEntry',
                      fieldbackground=self.input_bg,
                      foreground=self.text_color,
                      bordercolor=self.border_color,
                      lightcolor=self.border_color,
                      darkcolor=self.border_color,
                      padding=6)

        style.configure('TCheckbutton',
                      background=self.panel_color,
                      foreground=self.text_color)
        style.configure('TRadiobutton',
                      background=self.panel_color,
                      foreground=self.text_color)

# ========== DATA ENTRY TABLE ==========
class DataEntryTable(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.conc_vars = []
        self.od_vars = []
        self.configure(style='Section.TFrame', padding=8)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        
        # Create header
        ttk.Label(self, text="Concentration", style='SectionTitle.TLabel', width=15).grid(row=0, column=0, padx=8, pady=(4, 6), sticky='w')
        ttk.Label(self, text="OD Value", style='SectionTitle.TLabel', width=15).grid(row=0, column=1, padx=8, pady=(4, 6), sticky='w')
        
        # Add initial rows
        for i in range(5):
            self.add_row(i+1)
            
        # Add control buttons
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=100, column=0, columnspan=2, pady=5)
        
        ttk.Button(btn_frame, text="+ Add Row", 
                  command=lambda: self.add_row(len(self.conc_vars)+1),
                  style='Primary.TButton').pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="- Remove Row", 
                  command=self.remove_row,
                  style='Danger.TButton').pack(side=tk.LEFT, padx=5)
    
    def add_row(self, row_num):
        """Add a new data entry row"""
        conc_var = tk.StringVar()
        od_var = tk.StringVar()
        
        ttk.Entry(self, textvariable=conc_var, width=15).grid(row=row_num, column=0, padx=5, pady=2)
        ttk.Entry(self, textvariable=od_var, width=15).grid(row=row_num, column=1, padx=5, pady=2)
        
        self.conc_vars.append(conc_var)
        self.od_vars.append(od_var)
    
    def remove_row(self):
        """Remove the last data entry row"""
        if len(self.conc_vars) > 1:
            self.conc_vars.pop()
            self.od_vars.pop()
            
            # Destroy the last row widgets
            for widget in self.grid_slaves():
                if int(widget.grid_info()["row"]) == len(self.conc_vars)+1:
                    widget.destroy()
    
    def get_data(self):
        """Get entered data as arrays"""
        concentrations = []
        od_values = []
        
        for conc_var, od_var in zip(self.conc_vars, self.od_vars):
            try:
                conc = float(conc_var.get())
                od = float(od_var.get())
                concentrations.append(conc)
                od_values.append(od)
            except (ValueError, TypeError):
                continue
                
        return np.array(concentrations), np.array(od_values)

    def set_data(self, concentrations, od_values):
        """Populate the table with concentration and OD values."""
        target_rows = max(1, len(concentrations), len(od_values))

        while len(self.conc_vars) < target_rows:
            self.add_row(len(self.conc_vars) + 1)

        while len(self.conc_vars) > target_rows:
            self.remove_row()

        for index in range(target_rows):
            conc_value = concentrations[index] if index < len(concentrations) else ""
            od_value = od_values[index] if index < len(od_values) else ""
            self.conc_vars[index].set("" if conc_value == "" else str(conc_value))
            self.od_vars[index].set("" if od_value == "" else str(od_value))

# ========== MAIN APPLICATION ==========
class ELISAApplication:
    def __init__(self, root):
        self.root = root
        self.style = StyleManager()
        self.style.configure_styles()
        
        # Configure main window
        self.root.title(APP_NAME)
        self.root.geometry("1100x750")
        self.root.configure(bg=self.style.bg_color)
        
        # Initialize variables
        self.x = None
        self.y = None
        self.popt = None
        self.r_squared = None
        self.fitted_model = None
        self.welcome_dialog = None
        self.license_dialog = None
        self.readme_dialog = None
        self.current_project_path = None
        self.last_saved_state = None
        self.recent_projects = self.load_recent_projects()
        self.model_var = tk.StringVar(value='4PL')
        self.show_formula = tk.BooleanVar(value=True)
        self.show_r2 = tk.BooleanVar(value=True)
        
        # Build UI
        self.create_widgets()
        self.bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self.handle_app_close)
        
        # Show about dialog on first run
        self.show_about()

    def get_model_function(self):
        """Return the currently selected logistic model."""
        return four_param_logistic if self.model_var.get() == '4PL' else three_param_logistic

    def serialize_project_state(self):
        """Serialize project state for unsaved-change detection."""
        return json.dumps(self.get_project_state(), sort_keys=True)

    def has_unsaved_changes(self):
        """Return whether the current session should prompt for save."""
        if self.current_project_path is None:
            return True

        if self.last_saved_state is None:
            return True

        return self.serialize_project_state() != self.last_saved_state

    def handle_app_close(self):
        """Prompt to save unsaved work before closing the application."""
        if not self.confirm_discard_unsaved_changes("Do you want to save your project before closing?"):
            return

        self.root.destroy()

    def confirm_discard_unsaved_changes(self, prompt_message):
        """Prompt to save unsaved work before discarding the current session."""
        if not self.has_unsaved_changes():
            return True

        response = messagebox.askyesnocancel("Save Project", prompt_message)

        if response is None:
            return False

        if response:
            return self.save_project(show_feedback=False)

        return True

    def update_window_title(self):
        """Show the current project name in the main window title bar."""
        if self.current_project_path:
            self.root.title(Path(self.current_project_path).stem)
        else:
            self.root.title(APP_NAME)

    def load_recent_projects(self):
        """Load the recent project list from disk."""
        try:
            with open(RECENT_PROJECTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return []

        if not isinstance(data, list):
            return []

        return [str(path) for path in data[:MAX_RECENT_PROJECTS]]

    def save_recent_projects(self):
        """Persist the recent project list to disk."""
        try:
            RECENT_PROJECTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(RECENT_PROJECTS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.recent_projects[:MAX_RECENT_PROJECTS], f, indent=2)
        except OSError:
            pass

    def add_recent_project(self, filepath):
        """Add a project path to the recent-project list."""
        normalized_path = str(Path(filepath))
        self.recent_projects = [path for path in self.recent_projects if path != normalized_path]
        self.recent_projects.insert(0, normalized_path)
        self.recent_projects = self.recent_projects[:MAX_RECENT_PROJECTS]
        self.save_recent_projects()
        self.refresh_recent_projects_menu()

    def remove_recent_project(self, filepath):
        """Remove a missing or unwanted path from the recent-project list."""
        normalized_path = str(Path(filepath))
        self.recent_projects = [path for path in self.recent_projects if path != normalized_path]
        self.save_recent_projects()
        self.refresh_recent_projects_menu()

    def refresh_recent_projects_menu(self):
        """Rebuild the Recent Projects submenu."""
        self.recent_projects_menu.delete(0, tk.END)

        if not self.recent_projects:
            self.recent_projects_menu.add_command(label="No recent projects", state=tk.DISABLED)
            return

        for filepath in self.recent_projects:
            self.recent_projects_menu.add_command(
                label=Path(filepath).name,
                command=lambda target=filepath: self.open_recent_project(target),
            )

    def open_recent_project(self, filepath):
        """Open a project from the recent-project list."""
        if not self.confirm_discard_unsaved_changes(
            "Do you want to save your current project before loading another one?"
        ):
            return

        if not Path(filepath).exists():
            messagebox.showerror("Error", f"Project file not found:\n{filepath}")
            self.remove_recent_project(filepath)
            return

        self.load_project(filepath=filepath, prompt_to_save=False)

    def get_fitted_model_function(self):
        """Return the logistic model used by the current fit."""
        if self.fitted_model == '4PL':
            return four_param_logistic
        if self.fitted_model == '3PL':
            return three_param_logistic
        raise ValueError("Please fit the curve first.")

    def reset_analysis_state(self):
        """Clear the current fit, plot, and derived outputs."""
        self.popt = None
        self.r_squared = None
        self.fitted_model = None
        self.result_label.config(text="Estimated concentration: ")
        self.clear_bulk_data(update_status=False)
        self.ax.clear()
        self.canvas.draw()

    def validate_standard_data(self, x, y):
        """Validate, sort, and normalize standard curve input data."""
        if len(x) < 3:
            raise ValueError("At least 3 data points are required for curve fitting.")

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("All concentration and OD values must be finite numbers.")

        if np.any(x <= 0):
            raise ValueError("All concentrations must be greater than 0 for log-scale curve fitting.")

        if len(np.unique(x)) < 3:
            raise ValueError("At least 3 distinct concentration values are required.")

        order = np.argsort(x)
        return x[order], y[order]

    def set_standard_data(self, x, y, source_label):
        """Store validated standard data and clear stale fit results."""
        self.x, self.y = self.validate_standard_data(x, y)
        self.reset_analysis_state()
        self.status_bar.config(text=f"Using {len(self.x)} data points from {source_label}")

    def find_matching_column(self, columns, kind):
        """Select the best matching concentration or OD column from a CSV."""
        normalized = {col: normalize_column_name(col) for col in columns}

        if kind == 'concentration':
            preferred = (
                'concentration',
                'concentrations',
                'conc',
                'stdconcentration',
                'standardconcentration',
            )
            contains_tokens = ('concentration', 'conc')
        else:
            preferred = (
                'od',
                'odvalue',
                'opticaldensity',
                'absorbance',
                'absorbancevalue',
                'abs',
            )
            contains_tokens = ('opticaldensity', 'absorbance', 'od')

        for target in preferred:
            for col, norm in normalized.items():
                if norm == target:
                    return col

        for col, norm in normalized.items():
            if any(token in norm for token in contains_tokens):
                return col

        return None

    def solve_concentration_from_od(self, od_value):
        """Estimate concentration within the fitted calibration range."""
        if self.popt is None or self.x is None:
            raise ValueError("Please fit the curve first.")

        if not np.isfinite(od_value):
            raise ValueError("OD value must be numeric.")

        model_func = self.get_fitted_model_function()
        x_low = float(np.min(self.x))
        x_high = float(np.max(self.x))

        if self.fitted_model == '3PL':
            fit_x_low = max(np.nextafter(0.0, 1.0), x_low * 1e-9)
            fit_x_high = max(x_high * 1e9, float(self.popt[2]) * 1e6, fit_x_low * 10)
            y_min = min(THREE_PL_D, float(self.popt[0]))
            y_max = max(THREE_PL_D, float(self.popt[0]))
        else:
            fit_x_low = x_low
            fit_x_high = x_high
            y_low = float(model_func(fit_x_low, *self.popt))
            y_high = float(model_func(fit_x_high, *self.popt))
            y_min = min(y_low, y_high)
            y_max = max(y_low, y_high)

        if od_value < y_min or od_value > y_max:
            raise ValueError(
                f"OD value is outside the fitted calibration range ({y_min:.4f} to {y_max:.4f})."
            )

        if self.fitted_model == '3PL' and np.isclose(od_value, THREE_PL_D):
            if self.popt[1] > 0:
                return 0.0
            raise ValueError(
                "OD value is at the 3PL lower asymptote and does not correspond to a finite concentration."
            )

        residual = lambda conc: model_func(conc, *self.popt) - od_value
        conc = brentq(residual, fit_x_low, fit_x_high)

        if not np.isfinite(conc) or conc <= 0:
            raise ValueError("Estimated concentration is outside the valid calibration range.")

        return conc
    
    def create_widgets(self):
        """Create all interface components"""
        self.setup_menu()

        # Main container
        self.main_frame = ttk.Frame(self.root, style='Card.TFrame', padding=15)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.setup_data_tab()
        self.setup_analysis_tab()
        self.setup_visualization_tab()
        
        # Status bar
        self.status_bar = ttk.Label(self.root, 
                                  text="Ready", 
                                  relief=tk.SUNKEN,
                                  background="#dde5ef",
                                  foreground=self.style.text_color,
                                  anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def show_text_context_menu(self, event, widget, editable=True):
        """Show a right-click context menu for text widgets."""
        widget.focus_force()
        menu = tk.Menu(widget, tearoff=0)

        if editable:
            menu.add_command(label="Cut", command=lambda: widget.event_generate("<<Cut>>"))

        menu.add_command(label="Copy", command=lambda: widget.event_generate("<<Copy>>"))

        if editable:
            menu.add_command(label="Paste", command=lambda: widget.event_generate("<<Paste>>"))

        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def bind_text_context_menu(self, widget, editable=True):
        """Bind right-click context menu behavior to a text widget."""
        widget.bind(
            "<Button-3>",
            lambda event, target=widget, can_edit=editable: self.show_text_context_menu(
                event, target, can_edit
            ),
        )

    def setup_menu(self):
        """Create the application menu."""
        menu_bar = tk.Menu(self.root)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="New Project", command=self.new_project, accelerator="Ctrl+N")
        file_menu.add_separator()
        file_menu.add_command(label="Save", command=self.save_project, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As...", command=self.save_project_as)
        file_menu.add_command(label="Load Project", command=self.load_project, accelerator="Ctrl+O")
        self.recent_projects_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Recent Projects", menu=self.recent_projects_menu)
        self.refresh_recent_projects_menu()
        menu_bar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="License", command=self.show_license)
        help_menu.add_command(label="User Guide", command=self.show_readme)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=menu_bar)

    def bind_shortcuts(self):
        """Bind application keyboard shortcuts."""
        self.root.bind_all("<Control-n>", lambda event: self.new_project())
        self.root.bind_all("<Control-s>", lambda event: self.save_project())
        self.root.bind_all("<Control-o>", lambda event: self.load_project())

    def reset_to_new_project(self):
        """Clear the current session back to a new blank project."""
        self.current_project_path = None
        self.last_saved_state = None
        self.x = None
        self.y = None
        self.model_var.set('4PL')
        self.show_formula.set(True)
        self.show_r2.set(True)

        self.data_table.set_data(["", "", "", "", ""], ["", "", "", "", ""])
        self.preview_text.delete("1.0", tk.END)

        self.od_entry.delete(0, tk.END)
        self.plot_title.delete(0, tk.END)
        self.plot_title.insert(0, "ELISA Standard Curve")
        self.plot_xlabel.delete(0, tk.END)
        self.plot_xlabel.insert(0, "Concentration")
        self.plot_ylabel.delete(0, tk.END)
        self.plot_ylabel.insert(0, "OD Value")

        self.reset_analysis_state()
        self.update_window_title()
        self.status_bar.config(text="New project")

    def new_project(self):
        """Start a new blank project, prompting to save current work if needed."""
        if not self.confirm_discard_unsaved_changes(
            "Do you want to save your current project before starting a new one?"
        ):
            return

        self.reset_to_new_project()

    def choose_project_save_path(self):
        """Prompt for a project file path."""
        return filedialog.asksaveasfilename(
            title="Save Project As",
            defaultextension=".curvealyze.json",
            filetypes=[
                ("CurveAnalyze Project", "*.curvealyze.json"),
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
        )

    def get_project_state(self):
        """Collect the current application state for project save/load."""
        return {
            "project_version": 1,
            "data_table": {
                "concentrations": [var.get() for var in self.data_table.conc_vars],
                "od_values": [var.get() for var in self.data_table.od_vars],
            },
            "standard_data": {
                "x": None if self.x is None else self.x.tolist(),
                "y": None if self.y is None else self.y.tolist(),
            },
            "analysis": {
                "selected_model": self.model_var.get(),
                "fitted_model": self.fitted_model,
                "show_formula": self.show_formula.get(),
                "show_r2": self.show_r2.get(),
                "popt": None if self.popt is None else self.popt.tolist(),
                "r_squared": self.r_squared,
                "single_od": self.od_entry.get(),
                "single_result": self.result_label.cget("text"),
                "bulk_od_text": self.bulk_od_text.get("1.0", tk.END),
                "bulk_results_text": self.bulk_results.get("1.0", tk.END),
            },
            "plot": {
                "title": self.plot_title.get(),
                "xlabel": self.plot_xlabel.get(),
                "ylabel": self.plot_ylabel.get(),
            },
        }

    def restore_project_state(self, state):
        """Restore the application state from a saved project."""
        data_table = state.get("data_table", {})
        concentrations = data_table.get("concentrations", [])
        od_values = data_table.get("od_values", [])
        self.data_table.set_data(concentrations, od_values)

        plot_state = state.get("plot", {})
        self.plot_title.delete(0, tk.END)
        self.plot_title.insert(0, plot_state.get("title", "ELISA Standard Curve"))
        self.plot_xlabel.delete(0, tk.END)
        self.plot_xlabel.insert(0, plot_state.get("xlabel", "Concentration"))
        self.plot_ylabel.delete(0, tk.END)
        self.plot_ylabel.insert(0, plot_state.get("ylabel", "OD Value"))

        analysis_state = state.get("analysis", {})
        self.model_var.set(analysis_state.get("selected_model", "4PL"))
        self.show_formula.set(analysis_state.get("show_formula", True))
        self.show_r2.set(analysis_state.get("show_r2", True))
        self.od_entry.delete(0, tk.END)
        self.od_entry.insert(0, analysis_state.get("single_od", ""))
        self.result_label.config(text=analysis_state.get("single_result", "Estimated concentration: "))

        self.bulk_od_text.delete("1.0", tk.END)
        self.bulk_od_text.insert("1.0", analysis_state.get("bulk_od_text", ""))

        self.bulk_results.config(state='normal')
        self.bulk_results.delete("1.0", tk.END)
        self.bulk_results.insert("1.0", analysis_state.get("bulk_results_text", ""))
        self.bulk_results.config(state='disabled')

        standard_data = state.get("standard_data", {})
        x_values = standard_data.get("x")
        y_values = standard_data.get("y")

        self.popt = None
        self.r_squared = None
        self.fitted_model = None

        if x_values is not None and y_values is not None:
            self.x = np.asarray(x_values, dtype=float)
            self.y = np.asarray(y_values, dtype=float)

            popt = analysis_state.get("popt")
            fitted_model = analysis_state.get("fitted_model")
            if popt is not None and fitted_model in {"4PL", "3PL"}:
                self.popt = np.asarray(popt, dtype=float)
                self.fitted_model = fitted_model
                self.r_squared = analysis_state.get("r_squared")
                self.update_plot()
            else:
                self.ax.clear()
                self.canvas.draw()
        else:
            self.x = None
            self.y = None
            self.ax.clear()
            self.canvas.draw()

    def save_project(self, show_feedback=True):
        """Save the current analysis state to the current project file."""
        filepath = self.current_project_path or self.choose_project_save_path()

        if not filepath:
            return False

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.get_project_state(), f, indent=2)

            self.current_project_path = filepath
            self.last_saved_state = self.serialize_project_state()
            self.add_recent_project(filepath)
            self.update_window_title()
            self.status_bar.config(text=f"Project saved: {filepath}")
            if show_feedback:
                messagebox.showinfo("Success", f"Project saved successfully to:\n{filepath}")
            return True
        except OSError as e:
            messagebox.showerror("Error", f"Failed to save project:\n{str(e)}")
            return False

    def save_project_as(self):
        """Save the current analysis state to a new project file."""
        filepath = self.choose_project_save_path()
        if not filepath:
            return False

        self.current_project_path = filepath
        return self.save_project()

    def load_project(self, filepath=None, prompt_to_save=True):
        """Load a saved analysis state from a project file."""
        if prompt_to_save and not self.confirm_discard_unsaved_changes(
            "Do you want to save your current project before loading another one?"
        ):
            return

        if filepath is None:
            filepath = filedialog.askopenfilename(
                title="Load Project",
                filetypes=[
                    ("CurveAnalyze Project", "*.curvealyze.json"),
                    ("JSON files", "*.json"),
                    ("All files", "*.*"),
                ],
            )

        if not filepath:
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                state = json.load(f)

            self.restore_project_state(state)
            self.current_project_path = filepath
            self.last_saved_state = self.serialize_project_state()
            self.add_recent_project(filepath)
            self.update_window_title()
            self.status_bar.config(text=f"Project loaded: {filepath}")
            messagebox.showinfo("Success", f"Project loaded successfully from:\n{filepath}")
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as e:
            messagebox.showerror("Error", f"Failed to load project:\n{str(e)}")
    
    def setup_data_tab(self):
        """Data input tab"""
        data_tab = ttk.Frame(self.notebook)
        self.notebook.add(data_tab, text="Data Input")
        
        # Notebook for input methods
        input_methods = ttk.Notebook(data_tab)
        input_methods.pack(fill=tk.BOTH, expand=True)
        
        # CSV Import
        csv_frame = ttk.Frame(input_methods, style='Section.TFrame', padding=12)
        ttk.Label(csv_frame, text="CSV Import", style='Title.TLabel').pack(pady=10)
        
        btn_frame = ttk.Frame(csv_frame)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Browse CSV", 
                  command=self.load_csv,
                  style='Primary.TButton').pack()
        
        # Preview area
        preview_frame = ttk.LabelFrame(csv_frame, text="Data Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.preview_text = tk.Text(preview_frame, height=8, wrap=tk.NONE)
        scroll_y = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview_text.yview)
        scroll_x = ttk.Scrollbar(preview_frame, orient="horizontal", command=self.preview_text.xview)
        self.preview_text.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.preview_text.pack(fill=tk.BOTH, expand=True)
        
        input_methods.add(csv_frame, text="CSV Import")
        
        # Manual Entry
        manual_frame = ttk.Frame(input_methods, style='Section.TFrame', padding=12)
        ttk.Label(manual_frame, text="Manual Entry", style='Title.TLabel').pack(pady=10)
        
        self.data_table = DataEntryTable(manual_frame)
        self.data_table.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        ttk.Button(manual_frame, text="Use This Data",
                  command=self.use_manual_data,
                  style='Accent.TButton').pack(pady=10)
        
        input_methods.add(manual_frame, text="Manual Entry")
    
    def setup_analysis_tab(self):
        """Analysis controls tab"""
        analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(analysis_tab, text="Analysis")

        analysis_tab.columnconfigure(0, weight=0)
        analysis_tab.columnconfigure(1, weight=1)
        analysis_tab.rowconfigure(0, weight=1)

        control_frame = ttk.Frame(analysis_tab, style='Section.TFrame', padding=12)
        control_frame.grid(row=0, column=0, sticky="ns", padx=(10, 5), pady=10)

        results_frame = ttk.Frame(analysis_tab, style='Section.TFrame', padding=12)
        results_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(2, weight=1)

        # Model selection
        model_frame = ttk.LabelFrame(control_frame, text="Model Selection")
        model_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Radiobutton(model_frame, text="4-Parameter Logistic (4PL)", 
                       variable=self.model_var, value='4PL').pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(model_frame, text="3-Parameter Logistic (3PL)", 
                       variable=self.model_var, value='3PL').pack(anchor=tk.W, padx=5, pady=2)
        
        # Plot options
        opt_frame = ttk.LabelFrame(control_frame, text="Plot Options")
        opt_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(opt_frame, text="Show Formula",
                       variable=self.show_formula).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Checkbutton(opt_frame, text="Show R² Value",
                       variable=self.show_r2).pack(anchor=tk.W, padx=5, pady=2)
        
        # Fit button
        ttk.Button(control_frame, text="Fit Curve", 
                  command=self.fit_curve,
                  style='Accent.TButton').pack(fill=tk.X, pady=10)
        
        # Unknown concentration estimation
        unknown_frame = ttk.LabelFrame(control_frame, text="Concentration Estimation")
        unknown_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(unknown_frame, text="Enter OD Value:").grid(row=0, column=0, padx=5, pady=2)
        self.od_entry = ttk.Entry(unknown_frame, width=15)
        self.od_entry.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Button(unknown_frame, text="Calculate", 
                  command=self.estimate_concentration,
                  style='Primary.TButton').grid(row=0, column=2, padx=5, pady=2)
        
        self.result_label = ttk.Label(unknown_frame, text="Estimated concentration: ")
        self.result_label.grid(row=1, column=0, columnspan=3, pady=5)
        
        # Bulk OD estimation section
        bulk_frame = ttk.LabelFrame(results_frame, text="Bulk OD Estimation")
        bulk_frame.grid(row=0, column=0, sticky="nsew")
        bulk_frame.columnconfigure(0, weight=1)
        bulk_frame.rowconfigure(3, weight=1)

        ttk.Label(bulk_frame, text="Enter multiple OD values (one per line):").grid(
            row=0, column=0, sticky="w", padx=8, pady=(6, 2)
        )

        self.bulk_od_text = tk.Text(
            bulk_frame,
            height=6,
            width=40,
            wrap=tk.NONE,
            background=self.style.input_bg,
            foreground=self.style.text_color,
            insertbackground=self.style.text_color,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=self.style.border_color,
            highlightcolor=self.style.primary_color,
            padx=8,
            pady=8,
        )
        self.bulk_od_text.grid(row=1, column=0, sticky="ew", padx=8, pady=5)
        self.bind_text_context_menu(self.bulk_od_text, editable=True)
        
        # Bulk buttons
        bulk_btn_frame = ttk.Frame(bulk_frame)
        bulk_btn_frame.grid(row=2, column=0, sticky="w", padx=8, pady=5)
        
        ttk.Button(bulk_btn_frame, text="Estimate All", 
                  command=self.estimate_bulk_concentrations,
                  style='Primary.TButton').pack(side=tk.LEFT, padx=5)

        ttk.Button(bulk_btn_frame, text="Export CSV",
                  command=self.export_bulk_results_to_csv,
                  style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        
        ttk.Button(bulk_btn_frame, text="Clear", 
                  command=self.clear_bulk_data,
                  style='Danger.TButton').pack(side=tk.LEFT, padx=5)
        
        # Results display
        results_text_frame = ttk.Frame(bulk_frame)
        results_text_frame.grid(row=3, column=0, sticky="nsew", padx=8, pady=(5, 8))
        results_text_frame.columnconfigure(0, weight=1)
        results_text_frame.rowconfigure(0, weight=1)

        self.bulk_results = tk.Text(
            results_text_frame,
            height=14,
            width=70,
            state='disabled',
            wrap=tk.NONE,
            background="#f7f9fc",
            foreground=self.style.text_color,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=self.style.border_color,
            highlightcolor=self.style.primary_color,
            padx=8,
            pady=8,
        )
        self.bind_text_context_menu(self.bulk_results, editable=False)
        bulk_scroll_y = ttk.Scrollbar(results_text_frame, orient="vertical", command=self.bulk_results.yview)
        bulk_scroll_x = ttk.Scrollbar(results_text_frame, orient="horizontal", command=self.bulk_results.xview)
        self.bulk_results.configure(yscrollcommand=bulk_scroll_y.set, xscrollcommand=bulk_scroll_x.set)

        bulk_scroll_y.grid(row=0, column=1, sticky="ns")
        bulk_scroll_x.grid(row=1, column=0, sticky="ew")
        self.bulk_results.grid(row=0, column=0, sticky="nsew")
    
    def setup_visualization_tab(self):
        """Results visualization tab"""
        vis_tab = ttk.Frame(self.notebook, style='Section.TFrame', padding=12)
        self.notebook.add(vis_tab, text="Standard Curve")
        
        # Create plot
        self.fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=vis_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Plot controls
        ctrl_frame = ttk.Frame(vis_tab, style='Section.TFrame', padding=8)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(ctrl_frame, text="Save Plot", 
                  command=self.save_plot,
                  style='Primary.TButton').pack(side=tk.LEFT, padx=5)
        
        # Plot labels
        label_frame = ttk.Frame(ctrl_frame, style='Section.TFrame')
        label_frame.pack(side=tk.LEFT, padx=20)
        
        ttk.Label(label_frame, text="Title:").grid(row=0, column=0, sticky='e', pady=2)
        self.plot_title = ttk.Entry(label_frame, width=30)
        self.plot_title.insert(0, "ELISA Standard Curve")
        self.plot_title.grid(row=0, column=1, pady=2)
        
        ttk.Label(label_frame, text="X-label:").grid(row=1, column=0, sticky='e', pady=2)
        self.plot_xlabel = ttk.Entry(label_frame, width=30)
        self.plot_xlabel.insert(0, "Concentration")
        self.plot_xlabel.grid(row=1, column=1, pady=2)
        
        ttk.Label(label_frame, text="Y-label:").grid(row=2, column=0, sticky='e', pady=2)
        self.plot_ylabel = ttk.Entry(label_frame, width=30)
        self.plot_ylabel.insert(0, "OD Value")
        self.plot_ylabel.grid(row=2, column=1, pady=2)
    
    def show_about(self):
        """Show startup welcome dialog"""
        if self.welcome_dialog is not None and self.welcome_dialog.winfo_exists():
            self.welcome_dialog.deiconify()
            self.welcome_dialog.lift()
            self.welcome_dialog.focus_force()
            return

        about = tk.Toplevel(self.root)
        self.welcome_dialog = about
        about.title("Welcome to CurveAnalyze")
        about.resizable(False, False)
        about.transient(self.root)
        about.grab_set()

        def close_dialog():
            if self.welcome_dialog is not None and self.welcome_dialog.winfo_exists():
                self.welcome_dialog.destroy()
            self.welcome_dialog = None

        about.protocol("WM_DELETE_WINDOW", close_dialog)
        
        # Content
        content = ttk.Frame(about, padding=15)
        content.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(content, text="CurveAnalyze",
                 style='Title.TLabel').pack(pady=10)
        
        ttk.Label(content, text="Author: Abdul Halim Sadikin",
                 font=("Segoe UI", 10)).pack()

        ttk.Label(content, text="©2025",
                 font=("Segoe UI", 10)).pack()

        info_text = """CurveAnalyze helps you fit ELISA standard curves and
estimate unknown sample concentrations.

Quick start:
1. Load standards from a CSV file or enter them manually.
2. Choose a 4PL or 3PL model in the Analysis tab.
3. Fit the curve and review the plotted result.
4. Estimate single or bulk concentrations from OD values.

Notes:
- Concentrations must be greater than 0 because the curve is plotted on a log x-axis.
- OD values outside the fitted calibration range are reported as out of range.

Models:
- 4PL: y = D + (A-D)/(1+(x/C)^B)
- 3PL: y = D + (A-D)/(1+(C/x)^B), with D = 0"""
        
        ttk.Label(
            content,
            text=info_text,
            justify=tk.LEFT,
            wraplength=455
        ).pack(fill=tk.BOTH, expand=True, pady=10)
        
        start_button = ttk.Button(content, text="Start", 
                                 command=close_dialog,
                                 style='Primary.TButton')
        start_button.pack(pady=10)
        start_button.focus_set()
        about.bind("<Return>", lambda event: start_button.invoke())

        # Size the dialog to its content, then center it on screen.
        about.update_idletasks()
        w = about.winfo_reqwidth()
        h = about.winfo_reqheight()
        ws = self.root.winfo_screenwidth()
        hs = self.root.winfo_screenheight()
        x = (ws - w) // 2
        y = (hs - h) // 2
        about.geometry(f"{w}x{h}+{x}+{y}")

    def show_license(self):
        """Show the license text in a scrollable dialog."""
        if self.license_dialog is not None and self.license_dialog.winfo_exists():
            self.license_dialog.deiconify()
            self.license_dialog.lift()
            self.license_dialog.focus_force()
            return

        license_dialog = tk.Toplevel(self.root)
        self.license_dialog = license_dialog
        license_dialog.title("CurveAnalyze License")
        license_dialog.resizable(True, True)
        license_dialog.transient(self.root)

        def close_dialog():
            if self.license_dialog is not None and self.license_dialog.winfo_exists():
                self.license_dialog.destroy()
            self.license_dialog = None

        license_dialog.protocol("WM_DELETE_WINDOW", close_dialog)

        content = ttk.Frame(license_dialog, padding=15)
        content.pack(fill=tk.BOTH, expand=True)

        ttk.Label(content, text="License", style='Title.TLabel').pack(pady=(0, 10))

        text_frame = ttk.Frame(content)
        text_frame.pack(fill=tk.BOTH, expand=True)

        license_text = tk.Text(text_frame, wrap=tk.WORD, height=20, width=80)
        scroll_y = ttk.Scrollbar(text_frame, orient="vertical", command=license_text.yview)
        license_text.configure(yscrollcommand=scroll_y.set)

        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        license_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        try:
            with open(RESOURCE_DIR / "LICENSE", "r", encoding="utf-8") as f:
                license_content = f.read()
        except OSError as e:
            license_content = f"Failed to load LICENSE file:\n{str(e)}"

        license_text.insert("1.0", license_content)
        license_text.config(state='disabled')

        close_button = ttk.Button(content, text="Close", command=close_dialog, style='Primary.TButton')
        close_button.pack(pady=(10, 0))
        close_button.focus_set()
        license_dialog.bind("<Return>", lambda event: close_button.invoke())

        license_dialog.update_idletasks()
        w = max(680, license_dialog.winfo_reqwidth())
        h = max(500, license_dialog.winfo_reqheight())
        ws = self.root.winfo_screenwidth()
        hs = self.root.winfo_screenheight()
        x = (ws - w) // 2
        y = (hs - h) // 2
        license_dialog.geometry(f"{w}x{h}+{x}+{y}")

    def show_readme(self):
        """Show the README text in a scrollable dialog."""
        if self.readme_dialog is not None and self.readme_dialog.winfo_exists():
            self.readme_dialog.deiconify()
            self.readme_dialog.lift()
            self.readme_dialog.focus_force()
            return

        readme_dialog = tk.Toplevel(self.root)
        self.readme_dialog = readme_dialog
        readme_dialog.title("CurveAnalyze README")
        readme_dialog.resizable(True, True)
        readme_dialog.transient(self.root)

        def close_dialog():
            if self.readme_dialog is not None and self.readme_dialog.winfo_exists():
                self.readme_dialog.destroy()
            self.readme_dialog = None

        readme_dialog.protocol("WM_DELETE_WINDOW", close_dialog)

        content = ttk.Frame(readme_dialog, padding=15)
        content.pack(fill=tk.BOTH, expand=True)

        ttk.Label(content, text="README", style='Title.TLabel').pack(pady=(0, 10))

        text_frame = ttk.Frame(content)
        text_frame.pack(fill=tk.BOTH, expand=True)

        readme_text = tk.Text(text_frame, wrap=tk.WORD, height=20, width=80)
        scroll_y = ttk.Scrollbar(text_frame, orient="vertical", command=readme_text.yview)
        readme_text.configure(yscrollcommand=scroll_y.set)

        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        readme_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        try:
            with open(RESOURCE_DIR / "README.md", "r", encoding="utf-8") as f:
                readme_content = f.read()
        except OSError as e:
            readme_content = f"Failed to load README.md file:\n{str(e)}"

        readme_text.insert("1.0", readme_content)
        readme_text.config(state='disabled')

        close_button = ttk.Button(content, text="Close", command=close_dialog, style='Primary.TButton')
        close_button.pack(pady=(10, 0))
        close_button.focus_set()
        readme_dialog.bind("<Return>", lambda event: close_button.invoke())

        readme_dialog.update_idletasks()
        w = max(680, readme_dialog.winfo_reqwidth())
        h = max(500, readme_dialog.winfo_reqheight())
        ws = self.root.winfo_screenwidth()
        hs = self.root.winfo_screenheight()
        x = (ws - w) // 2
        y = (hs - h) // 2
        readme_dialog.geometry(f"{w}x{h}+{x}+{y}")
    
    # ========== CORE FUNCTIONALITY ==========
    def load_csv(self):
        """Load data from CSV file"""
        filepath = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not filepath:
            return
            
        try:
            # Try reading with different delimiters
            try:
                df = pd.read_csv(filepath)
            except (pd.errors.ParserError, UnicodeDecodeError):
                try:
                    df = pd.read_csv(filepath, delimiter=';')
                except (pd.errors.ParserError, UnicodeDecodeError):
                    df = pd.read_csv(filepath, delimiter='\t')
                
            conc_col = self.find_matching_column(df.columns, 'concentration')
            od_col = self.find_matching_column(df.columns, 'od')

            if conc_col is None or od_col is None:
                messagebox.showerror("Error", 
                    "CSV must contain columns with 'Concentration' (or 'Conc') and 'OD' (or 'Absorbance')")
                return
            
            # Extract data and remove NaN values
            data = df[[conc_col, od_col]].dropna()
            self.set_standard_data(data[conc_col].values, data[od_col].values, f"CSV: {filepath}")
            
            # Update preview
            self.preview_text.delete(1.0, tk.END)
            preview_str = f"Columns found: {conc_col}, {od_col}\n"
            preview_str += f"Data points: {len(self.x)}\n\n"
            preview_df = pd.DataFrame({
                conc_col: self.x,
                od_col: self.y,
            })
            preview_str += preview_df.head(10).to_string(index=False)
            self.preview_text.insert(tk.END, preview_str)
            
            self.status_bar.config(text=f"Loaded {len(self.x)} data points from {filepath}")
            messagebox.showinfo("Success", f"Successfully loaded {len(self.x)} data points")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV:\n{str(e)}")
    
    def use_manual_data(self):
        """Use manually entered data"""
        x, y = self.data_table.get_data()

        try:
            self.set_standard_data(x, y, "manual entry")
            messagebox.showinfo("Success", f"Using {len(self.x)} entered data points")
        except ValueError as e:
            messagebox.showwarning("Warning", str(e))
    
    def fit_curve(self):
        """Perform curve fitting with selected model"""
        if self.x is None or len(self.x) < 3:
            messagebox.showwarning("Warning", "Please load at least 3 valid data points first")
            return
            
        try:
            model = self.model_var.get()
            self.popt = None
            self.r_squared = None
            self.fitted_model = None
            
            # Prepare bounds and initial guesses
            if model == '4PL':
                increasing = self.y[-1] >= self.y[0]
                slope_guess = -1.0 if increasing else 1.0

                # 4PL parameter order: [A, B, C, D]
                p0 = [
                    max(self.y),           # A (upper asymptote)
                    slope_guess,           # B (Hill slope)
                    np.median(self.x),     # C (inflection point)
                    min(self.y)            # D (lower asymptote)
                ]
                bounds = (
                    [0, -10, 0, 0],                    # Lower bounds
                    [np.inf, 10, np.inf, np.inf]       # Upper bounds
                )
            else:  # 3PL
                increasing = self.y[-1] >= self.y[0]
                slope_guess = 1.0 if increasing else -1.0

                # 3PL parameter order: [A, B, C] (D fixed at 0)
                p0 = [
                    max(self.y),           # A (upper asymptote)
                    slope_guess,           # B (Hill slope)
                    np.median(self.x)      # C (inflection point)
                ]
                if increasing:
                    bounds = (
                        [0, 0.1, 0],           # Lower bounds
                        [np.inf, 10, np.inf]   # Upper bounds
                    )
                else:
                    bounds = (
                        [0, -10, 0],           # Lower bounds
                        [np.inf, -0.1, np.inf] # Upper bounds
                    )

            # Perform the curve fitting
            if model == '4PL':
                self.popt, _ = curve_fit(four_param_logistic, self.x, self.y,
                                       p0=p0, bounds=bounds, maxfev=5000)
            else:  # 3PL
                self.popt, _ = curve_fit(three_param_logistic, self.x, self.y,
                                       p0=p0, bounds=bounds, maxfev=5000)

            self.fitted_model = model
            model_func = self.get_fitted_model_function()
            y_pred = model_func(self.x, *self.popt)
            
            # Calculate R-squared
            self.r_squared = calculate_r_squared(self.y, y_pred)
            
            # Update the plot
            self.update_plot()
            
            # Show success message with parameters
            param_text = f"Curve fitting successful ({model})\n\n"
            param_labels = ['A (Upper)', 'B (Slope)', 'C (IC50)', 'D (Lower)']
            
            for i, (param, label) in enumerate(zip(self.popt, param_labels[:len(self.popt)])):
                param_text += f"{label}: {param:.6f}\n"

            if model == '3PL':
                param_text += f"D (Lower): {THREE_PL_D:.6f}\n"
            
            param_text += f"\nR² = {self.r_squared:.6f}"
            
            messagebox.showinfo("Curve Fit Results", param_text)
            self.status_bar.config(text=f"{model} fit complete - R² = {self.r_squared:.4f}")
            
        except RuntimeError as e:
            messagebox.showerror("Curve Fitting Error", 
                f"Failed to fit {model} curve:\n{str(e)}\n\n"
                "Possible solutions:\n"
                "• Check if data follows expected curve shape\n"
                "• Verify data quality and remove outliers\n"
                "• Try the alternative model (3PL vs 4PL)")
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error during curve fitting:\n{str(e)}")
    
    def update_plot(self):
        """Update the plot with current data and fit"""
        if self.popt is None:
            return
            
        self.ax.clear()
        
        # Plot raw data
        self.ax.scatter(self.x, self.y, color='red', s=50, alpha=0.7, label='Data', zorder=5)
        
        # Plot fitted curve
        x_min, x_max = min(self.x), max(self.x)
        x_fit = np.logspace(np.log10(x_min), np.log10(x_max), 200)
        
        y_fit = self.get_fitted_model_function()(x_fit, *self.popt)
            
        self.ax.plot(x_fit, y_fit, 'b-', linewidth=2, label=f'{self.fitted_model} Fit')
        
        # Format plot
        self.ax.set_xscale('log')
        self.ax.set_title(self.plot_title.get(), fontsize=12, fontweight='bold')
        self.ax.set_xlabel(self.plot_xlabel.get(), fontsize=10)
        self.ax.set_ylabel(self.plot_ylabel.get(), fontsize=10)
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        # Add annotations if enabled
        annotations = []
        
        if self.show_formula.get():
            if self.fitted_model == '4PL':
                formula = r'$y = D + \frac{A-D}{1+(x/C)^B}$'
                params = f"A = {self.popt[0]:.4f}\nB = {self.popt[1]:.4f}\n"
                params += f"C = {self.popt[2]:.4f}\nD = {self.popt[3]:.4f}"
            else:
                formula = r'$y = D + \frac{A-D}{1+(C/x)^B}$'
                params = f"A = {self.popt[0]:.4f}\nB = {self.popt[1]:.4f}\n"
                params += f"C = {self.popt[2]:.4f}\nD = {THREE_PL_D:.4f}"
            
            annotations.extend([formula, params])
        
        if self.show_r2.get() and self.r_squared is not None:
            annotations.append(f'R² = {self.r_squared:.4f}')
            
        if annotations:
            self.ax.text(0.02, 0.98, '\n'.join(annotations),
                        transform=self.ax.transAxes,
                        verticalalignment='top',
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        self.canvas.draw()
    
    def estimate_concentration(self):
        """Estimate concentration from single OD value"""
        if self.popt is None:
            messagebox.showwarning("Warning", "Please fit the curve first")
            return
            
        try:
            od_text = self.od_entry.get().strip()

            try:
                od_value = float(od_text)
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid numeric OD value")
                return
            
            if od_value < 0:
                messagebox.showerror("Error", "OD value must be 0 or positive")
                return

            conc = self.solve_concentration_from_od(od_value)
            self.result_label.config(text=f"Estimated concentration: {conc:.6f}")
            self.status_bar.config(text=f"Concentration: {conc:.6f} (OD: {od_value:.4f})")
            
        except ValueError as e:
            self.result_label.config(text="Estimated concentration: Out of calibration range")
            messagebox.showerror("Error", str(e))
        except RuntimeError as e:
            messagebox.showerror("Error", f"Failed to estimate concentration:\n{str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to estimate concentration:\n{str(e)}")
    
    def estimate_bulk_concentrations(self):
        """Estimate concentrations for multiple OD values"""
        if self.popt is None:
            messagebox.showwarning("Warning", "Please fit the curve first")
            return
            
        od_text = self.bulk_od_text.get(1.0, tk.END).strip()
        if not od_text:
            messagebox.showwarning("Warning", "Please enter OD values")
            return
            
        try:
            # Parse OD values (support multiple formats)
            od_values = []
            for line in od_text.split('\n'):
                line = line.strip()
                if line:
                    # Handle comma-separated values in a line
                    for val in line.replace(',', ' ').split():
                        try:
                            od_values.append(float(val))
                        except ValueError:
                            continue
            
            if not od_values:
                messagebox.showwarning("Warning", "No valid OD values found")
                return
            
            # Calculate concentrations
            results = []
            
            for od in od_values:
                try:
                    conc = self.solve_concentration_from_od(od)
                    results.append((od, f"{conc:.6f}"))
                except ValueError:
                    results.append((od, "Out of range"))
                except RuntimeError:
                    results.append((od, "Error"))
            
            # Display results
            result_text = "OD Value\t\tConcentration\n"
            result_text += "-" * 40 + "\n"
            
            for od, conc in results:
                result_text += f"{od:.4f}\t\t{conc}\n"
            
            self.bulk_results.config(state='normal')
            self.bulk_results.delete(1.0, tk.END)
            self.bulk_results.insert(tk.END, result_text)
            self.bulk_results.config(state='disabled')
            
            self.status_bar.config(text=f"Processed {len(od_values)} OD values")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process bulk OD values:\n{str(e)}")

    def get_bulk_results_rows(self):
        """Parse the displayed bulk results into exportable rows."""
        result_text = self.bulk_results.get("1.0", tk.END).strip()
        if not result_text:
            return []

        rows = []
        for line in result_text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("OD Value") or stripped.startswith("-"):
                continue

            parts = stripped.split()
            if len(parts) < 2:
                continue

            od_value = parts[0]
            concentration = " ".join(parts[1:])
            rows.append((od_value, concentration))

        return rows

    def export_bulk_results_to_csv(self):
        """Export the current bulk estimation results to a CSV file."""
        rows = self.get_bulk_results_rows()
        if not rows:
            messagebox.showwarning("Warning", "No bulk results available to export")
            return

        filepath = filedialog.asksaveasfilename(
            title="Export Bulk Results",
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
        )

        if not filepath:
            return

        try:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["OD Value", "Concentration"])
                writer.writerows(rows)

            self.status_bar.config(text=f"Bulk results exported: {filepath}")
            messagebox.showinfo("Success", f"Bulk results exported successfully to:\n{filepath}")
        except OSError as e:
            messagebox.showerror("Error", f"Failed to export bulk results:\n{str(e)}")
    
    def clear_bulk_data(self, update_status=True):
        """Clear bulk OD input and results"""
        self.bulk_od_text.delete(1.0, tk.END)
        self.bulk_results.config(state='normal')
        self.bulk_results.delete(1.0, tk.END)
        self.bulk_results.config(state='disabled')
        if update_status:
            self.status_bar.config(text="Bulk data cleared")
    
    def save_plot(self):
        """Save the current plot to file"""
        if self.popt is None:
            messagebox.showwarning("Warning", "No plot to save. Please fit a curve first.")
            return
            
        filetypes = [
            ('PNG Image', '*.png'),
            ('JPEG Image', '*.jpg'),
            ('PDF Document', '*.pdf'),
            ('SVG Vector', '*.svg'),
            ('All Files', '*.*')
        ]
        
        filepath = filedialog.asksaveasfilename(
            title="Save Plot As",
            defaultextension='.png',
            filetypes=filetypes
        )
        
        if filepath:
            try:
                self.fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                               facecolor='white', edgecolor='none')
                messagebox.showinfo("Success", f"Plot saved successfully to:\n{filepath}")
                self.status_bar.config(text=f"Plot saved: {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plot:\n{str(e)}")

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    root = tk.Tk()
    
    # Set window icon (optional)
    try:
        root.iconbitmap(str(RESOURCE_DIR / "icon.ico"))
    except tk.TclError:
        pass
    
    app = ELISAApplication(root)
    root.mainloop()
