
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import curve_fit, fsolve

# ========== MATHEMATICAL FUNCTIONS ==========
def four_param_logistic(x, A, B, C, D):
    """4-Parameter Logistic Function"""
    with np.errstate(invalid='ignore', divide='ignore'):
        return D + (A - D) / (1 + (x / C)**B)

def three_param_logistic(x, A, B, C):
    """3-Parameter Logistic Function (D fixed at 0)"""
    with np.errstate(invalid='ignore', divide='ignore'):
        return A / (1 + (x / C)**B)

def calculate_r_squared(y_true, y_pred):
    """Calculate R-squared value"""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

# ========== STYLE MANAGER ==========
class StyleManager:
    def __init__(self):
        self.bg_color = "#f5f6f8"
        self.card_color = "#ffffff"
        self.primary_color = "#4a6fa5"
        self.accent_color = "#28a745"
        self.error_color = "#dc3545"
        self.font_family = "Segoe UI"
        
    def configure_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # General styles
        style.configure('.', 
                      background=self.bg_color,
                      font=(self.font_family, 10))
        
        # Frame styles
        style.configure('Card.TFrame', 
                      background=self.card_color,
                      borderwidth=1,
                      bordercolor='#e0e0e0')
        
        # Button styles
        style.configure('Primary.TButton',
                      foreground='white',
                      background=self.primary_color,
                      borderwidth=0,
                      padding=6)
        
        style.configure('Accent.TButton',
                      foreground='white',
                      background=self.accent_color,
                      borderwidth=0,
                      padding=6)
        
        style.configure('Danger.TButton',
                      foreground='white',
                      background=self.error_color,
                      borderwidth=0,
                      padding=6)
        
        # Label styles
        style.configure('Title.TLabel',
                      font=(self.font_family, 12, 'bold'),
                      foreground=self.primary_color)
        
        style.configure('Subtitle.TLabel',
                      font=(self.font_family, 10),
                      foreground=self.primary_color)

# ========== DATA ENTRY TABLE ==========
class DataEntryTable(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.conc_vars = []
        self.od_vars = []
        
        # Create header
        ttk.Label(self, text="Concentration", width=15).grid(row=0, column=0, padx=5, pady=2)
        ttk.Label(self, text="OD Value", width=15).grid(row=0, column=1, padx=5, pady=2)
        
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

# ========== MAIN APPLICATION ==========
class ELISAApplication:
    def __init__(self, root):
        self.root = root
        self.style = StyleManager()
        self.style.configure_styles()
        
        # Configure main window
        self.root.title("ELISA Data Analyzer")
        self.root.geometry("1100x750")
        self.root.configure(bg=self.style.bg_color)
        
        # Initialize variables
        self.x = None
        self.y = None
        self.popt = None
        self.r_squared = None
        self.model_var = tk.StringVar(value='4PL')
        self.show_formula = tk.BooleanVar(value=True)
        self.show_r2 = tk.BooleanVar(value=True)
        
        # Build UI
        self.create_widgets()
        
        # Show about dialog on first run
        self.show_about()
    
    def create_widgets(self):
        """Create all interface components"""
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
                                  anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    def setup_data_tab(self):
        """Data input tab"""
        data_tab = ttk.Frame(self.notebook)
        self.notebook.add(data_tab, text="Data Input")
        
        # Notebook for input methods
        input_methods = ttk.Notebook(data_tab)
        input_methods.pack(fill=tk.BOTH, expand=True)
        
        # CSV Import
        csv_frame = ttk.Frame(input_methods)
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
        manual_frame = ttk.Frame(input_methods)
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
        
        # Model selection
        model_frame = ttk.LabelFrame(analysis_tab, text="Model Selection")
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Radiobutton(model_frame, text="4-Parameter Logistic (4PL)", 
                       variable=self.model_var, value='4PL').pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(model_frame, text="3-Parameter Logistic (3PL)", 
                       variable=self.model_var, value='3PL').pack(anchor=tk.W, padx=5, pady=2)
        
        # Plot options
        opt_frame = ttk.LabelFrame(analysis_tab, text="Plot Options")
        opt_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Checkbutton(opt_frame, text="Show Formula",
                       variable=self.show_formula).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Checkbutton(opt_frame, text="Show R² Value",
                       variable=self.show_r2).pack(anchor=tk.W, padx=5, pady=2)
        
        # Fit button
        ttk.Button(analysis_tab, text="Fit Curve", 
                  command=self.fit_curve,
                  style='Accent.TButton').pack(pady=10)
        
        # Unknown concentration estimation
        unknown_frame = ttk.LabelFrame(analysis_tab, text="Concentration Estimation")
        unknown_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(unknown_frame, text="Enter OD Value:").grid(row=0, column=0, padx=5, pady=2)
        self.od_entry = ttk.Entry(unknown_frame, width=15)
        self.od_entry.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Button(unknown_frame, text="Calculate", 
                  command=self.estimate_concentration,
                  style='Primary.TButton').grid(row=0, column=2, padx=5, pady=2)
        
        self.result_label = ttk.Label(unknown_frame, text="Estimated concentration: ")
        self.result_label.grid(row=1, column=0, columnspan=3, pady=5)
        
        # Bulk OD estimation section
        bulk_frame = ttk.LabelFrame(analysis_tab, text="Bulk OD Estimation")
        bulk_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(bulk_frame, text="Enter multiple OD values (one per line):").pack(pady=2)
        
        self.bulk_od_text = tk.Text(bulk_frame, height=5, width=40)
        self.bulk_od_text.pack(pady=5)
        
        # Bulk buttons
        bulk_btn_frame = ttk.Frame(bulk_frame)
        bulk_btn_frame.pack(pady=5)
        
        ttk.Button(bulk_btn_frame, text="Estimate All", 
                  command=self.estimate_bulk_concentrations,
                  style='Primary.TButton').pack(side=tk.LEFT, padx=5)
        
        ttk.Button(bulk_btn_frame, text="Clear", 
                  command=self.clear_bulk_data,
                  style='Danger.TButton').pack(side=tk.LEFT, padx=5)
        
        # Results display
        self.bulk_results = tk.Text(bulk_frame, height=5, width=60, state='disabled')
        self.bulk_results.pack(pady=5)
    
    def setup_visualization_tab(self):
        """Results visualization tab"""
        vis_tab = ttk.Frame(self.notebook)
        self.notebook.add(vis_tab, text="Results")
        
        # Create plot
        self.fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=vis_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Plot controls
        ctrl_frame = ttk.Frame(vis_tab)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(ctrl_frame, text="Save Plot", 
                  command=self.save_plot,
                  style='Primary.TButton').pack(side=tk.LEFT, padx=5)
        
        # Plot labels
        label_frame = ttk.Frame(ctrl_frame)
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
        """Show about dialog"""
        about = tk.Toplevel(self.root)
        about.title("About ELISA Analyzer")
        about.resizable(False, False)
        about.transient(self.root)
        about.grab_set()
        
        # Center window
        w = 400
        h = 350
        ws = self.root.winfo_screenwidth()
        hs = self.root.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        about.geometry(f"{w}x{h}+{int(x)}+{int(y)}")
        
        # Content
        content = ttk.Frame(about, padding=15)
        content.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(content, text="ELISA Data Analyzer", 
                 style='Title.TLabel').pack(pady=10)
        
        ttk.Label(content, text="Version 2.0",
                 font=("Segoe UI", 10)).pack()
        
        info_text = """This application performs curve fitting and 
concentration estimation for ELISA assays.

Features:
• 4PL and 3PL curve fitting
• CSV data import with preview
• Manual data entry
• Interactive plot visualization
• Single and bulk concentration estimation
• High-resolution plot export

Mathematical Models:
• 4PL: y = D + (A-D)/(1+(x/C)^B)
• 3PL: y = A/(1+(x/C)^B)

Developed for biomedical research applications."""
        
        ttk.Label(content, text=info_text, justify=tk.LEFT).pack(fill=tk.X, pady=10)
        
        ttk.Button(content, text="Close", 
                  command=about.destroy,
                  style='Primary.TButton').pack(pady=10)
    
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
            except:
                try:
                    df = pd.read_csv(filepath, delimiter=';')
                except:
                    df = pd.read_csv(filepath, delimiter='\t')
                
            # Validate required columns (case insensitive)
            cols = [col.lower() for col in df.columns]
            conc_found = any('concentration' in col or 'conc' in col for col in cols)
            od_found = any('od' in col or 'absorbance' in col or 'abs' in col for col in cols)
            
            if not (conc_found and od_found):
                messagebox.showerror("Error", 
                    "CSV must contain columns with 'Concentration' (or 'Conc') and 'OD' (or 'Absorbance')")
                return
                
            # Find the actual column names
            conc_col = None
            od_col = None
            
            for col in df.columns:
                if 'concentration' in col.lower() or 'conc' in col.lower():
                    conc_col = col
                if 'od' in col.lower() or 'absorbance' in col.lower() or 'abs' in col.lower():
                    od_col = col
            
            # Extract data and remove NaN values
            data = df[[conc_col, od_col]].dropna()
            self.x = data[conc_col].values.astype(float)
            self.y = data[od_col].values.astype(float)
            
            # Update preview
            self.preview_text.delete(1.0, tk.END)
            preview_str = f"Columns found: {conc_col}, {od_col}\n"
            preview_str += f"Data points: {len(data)}\n\n"
            preview_str += data.head(10).to_string(index=False)
            self.preview_text.insert(tk.END, preview_str)
            
            self.status_bar.config(text=f"Loaded {len(data)} data points from {filepath}")
            messagebox.showinfo("Success", f"Successfully loaded {len(data)} data points")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV:\n{str(e)}")
    
    def use_manual_data(self):
        """Use manually entered data"""
        x, y = self.data_table.get_data()
        
        if len(x) < 3:
            messagebox.showwarning("Warning", "At least 3 data points required for curve fitting")
            return
            
        self.x = x
        self.y = y
        self.status_bar.config(text=f"Using {len(x)} manually entered data points")
        messagebox.showinfo("Success", f"Using {len(x)} entered data points")
    
    def fit_curve(self):
        """Perform curve fitting with selected model"""
        if self.x is None or len(self.x) < 3:
            messagebox.showwarning("Warning", "Please load at least 3 valid data points first")
            return
            
        try:
            model = self.model_var.get()
            
            # Prepare bounds and initial guesses
            if model == '4PL':
                # 4PL parameter order: [A, B, C, D]
                p0 = [
                    max(self.y),           # A (upper asymptote)
                    1.0,                   # B (Hill slope)
                    np.median(self.x),     # C (inflection point)
                    min(self.y)            # D (lower asymptote)
                ]
                bounds = (
                    [0, 0.1, 0, 0],                    # Lower bounds
                    [np.inf, 10, np.inf, np.inf]       # Upper bounds
                )
            else:  # 3PL
                # 3PL parameter order: [A, B, C] (D fixed at 0)
                p0 = [
                    max(self.y),           # A (upper asymptote)
                    1.0,                   # B (Hill slope)
                    np.median(self.x)      # C (inflection point)
                ]
                bounds = (
                    [0, 0.1, 0],           # Lower bounds
                    [np.inf, 10, np.inf]   # Upper bounds
                )

            # Perform the curve fitting
            if model == '4PL':
                self.popt, _ = curve_fit(four_param_logistic, self.x, self.y, 
                                       p0=p0, bounds=bounds, maxfev=5000)
                y_pred = four_param_logistic(self.x, *self.popt)
            else:  # 3PL
                self.popt, _ = curve_fit(three_param_logistic, self.x, self.y,
                                       p0=p0, bounds=bounds, maxfev=5000)
                y_pred = three_param_logistic(self.x, *self.popt)
            
            # Calculate R-squared
            self.r_squared = calculate_r_squared(self.y, y_pred)
            
            # Update the plot
            self.update_plot()
            
            # Show success message with parameters
            param_text = f"Curve fitting successful ({model})\n\n"
            param_labels = ['A (Upper)', 'B (Slope)', 'C (IC50)', 'D (Lower)']
            
            for i, (param, label) in enumerate(zip(self.popt, param_labels[:len(self.popt)])):
                param_text += f"{label}: {param:.6f}\n"
            
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
        
        if self.model_var.get() == '4PL':
            y_fit = four_param_logistic(x_fit, *self.popt)
        else:
            y_fit = three_param_logistic(x_fit, *self.popt)
            
        self.ax.plot(x_fit, y_fit, 'b-', linewidth=2, label=f'{self.model_var.get()} Fit')
        
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
            if self.model_var.get() == '4PL':
                formula = r'$y = D + \frac{A-D}{1+(x/C)^B}$'
                params = f"A = {self.popt[0]:.4f}\nB = {self.popt[1]:.4f}\n"
                params += f"C = {self.popt[2]:.4f}\nD = {self.popt[3]:.4f}"
            else:
                formula = r'$y = \frac{A}{1+(x/C)^B}$'
                params = f"A = {self.popt[0]:.4f}\nB = {self.popt[1]:.4f}\n"
                params += f"C = {self.popt[2]:.4f}"
            
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
            od_value = float(self.od_entry.get())
            
            if od_value <= 0:
                messagebox.showerror("Error", "OD value must be positive")
                return
            
            # Get reasonable initial guess for concentration
            x0 = np.median(self.x) if len(self.x) > 0 else 1.0
            
            if self.model_var.get() == '4PL':
                conc = fsolve(lambda x: four_param_logistic(x, *self.popt) - od_value, x0=x0)[0]
            else:
                conc = fsolve(lambda x: three_param_logistic(x, *self.popt) - od_value, x0=x0)[0]
                
            if conc < 0:
                self.result_label.config(text="Estimated concentration: Outside curve range")
            else:
                self.result_label.config(text=f"Estimated concentration: {conc:.6f}")
                self.status_bar.config(text=f"Concentration: {conc:.6f} (OD: {od_value:.4f})")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid numeric OD value")
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
            x0 = np.median(self.x) if len(self.x) > 0 else 1.0
            
            for od in od_values:
                try:
                    if self.model_var.get() == '4PL':
                        conc = fsolve(lambda x: four_param_logistic(x, *self.popt) - od, x0=x0)[0]
                    else:
                        conc = fsolve(lambda x: three_param_logistic(x, *self.popt) - od, x0=x0)[0]
                    
                    if conc < 0:
                        results.append((od, "Out of range"))
                    else:
                        results.append((od, f"{conc:.6f}"))
                except:
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
    
    def clear_bulk_data(self):
        """Clear bulk OD input and results"""
        self.bulk_od_text.delete(1.0, tk.END)
        self.bulk_results.config(state='normal')
        self.bulk_results.delete(1.0, tk.END)
        self.bulk_results.config(state='disabled')
        self.status_bar.config(text="Bulk data cleared")
    
    def save_plot(self):
        """Save the current plot to file"""
        if not hasattr(self, 'ax') or len(self.ax.lines) == 0:
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
        root.iconbitmap('icon.ico')  # Add an icon file if available
    except:
        pass
    
    app = ELISAApplication(root)
    root.mainloop()


