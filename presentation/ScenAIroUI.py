"""
ScenAIroUI - Main User Interface
=================================

This module contains the main UI class for the ScenAIro application.
It provides a comprehensive graphical interface for:
    - Airport/runway configuration
    - Trajectory and point generation parameters
    - Environment settings (time, weather)
    - 3D visualization of sampling points
    - Data generation controls

The UI follows a clean, organized design with:
    - Left sidebar with tabbed configuration panels
    - Right content area with 3D plot and distribution preview
    - Bottom control bar with action buttons
    - Settings popup accessible via gear icon

Dependencies:
    - tkinter: GUI framework
    - matplotlib: 3D plotting and visualization
    - Custom tools: Backend logic modules

Author: ScenAIro Team
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json

# Plotting Libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Custom Tool Imports (Backend Logic)
from tools.MetadataFileReader import MetadataFileReader
from tools import SamplingPointGenerator, AircraftPositioningAgent
from tools.RunwayGeometryCalculator import RunwayGeometryCalculator
from tools.SettingsManager import SettingsManager
from presentation.SettingsPopup import SettingsPopup

class JSONManager:
    """
    Static utility class to handle JSON file operations (Save/Load).
    Provides standardized dialogs for file selection.
    """
    @staticmethod
    def save_to_file(data, filetypes=("JSON files", "*.json")):
        """Opens a file dialog to save a dictionary as a JSON file."""
        try:
            file = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[filetypes])
            if file:
                with open(file, "w") as f:
                    json.dump(data, f, indent=4)
                return file
            return None
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")

    @staticmethod
    def load_from_file(filetypes=("JSON files", "*.json")):
        """Opens a file dialog to load a JSON file and returns it as a dictionary."""
        try:
            file = filedialog.askopenfilename(defaultextension=".json", filetypes=[filetypes])
            if file:
                with open(file, "r") as f:
                    return json.load(f)
            return None
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
            return None


class ScenAIroUI(tk.Frame):
    """
    Main UI Class for the ScenAIro Application.
    
    Structure:
    - Left Sidebar (Fixed Width): Contains configuration tabs (Airport, Trajectory, Environment).
    - Right Content Area: Contains the 3D Visualization and 2D Distribution plots.
    - Settings Button: Located in bottom-left corner, opens settings popup.
    
    Design Style:
    - Classic Tkinter look with pastel color coding for different sections.
    - Settings are managed through SettingsManager for persistence.
    """

    def __init__(self, parent):
        """
        Initialize the UI components.
        
        Args:
            parent: The root Tkinter window or parent frame.
        """
        super().__init__(parent)
        
        # ----------------------------------------------------------------
        # Initialize Settings Manager
        # ----------------------------------------------------------------
        # The settings manager provides access to all configurable values
        self.settings_manager = SettingsManager()
        
        # Store references
        self.airport = None
        self.parent = parent
        self.jsonmanager = JSONManager()

        # ----------------------------------------------------------------
        # Global Design Constants
        # ----------------------------------------------------------------
        # Colors are defined here for consistency across the UI
        self.bg_main = "#f0f4f8"  # Light Grey-Blue background

        # --- Main Layout Frames ---
        main_frame = tk.Frame(self.parent, bg=self.bg_main)
        main_frame.pack(fill="both", expand=True)

        # ----------------------------------------------------------------
        # 1. Left Sidebar
        # ----------------------------------------------------------------
        # Width is now configurable via settings
        # 'pack_propagate(False)' prevents the frame from shrinking
        left_width = self.settings_manager.get("ui_layout", "left_sidebar_width")
        self.left_frame = tk.Frame(main_frame, bg=self.bg_main, width=left_width)
        self.left_frame.pack(side="left", fill="y", padx=5, pady=5)
        self.left_frame.pack_propagate(False) 

        # ----------------------------------------------------------------
        # 2. Right Content Area (Plot)
        # ----------------------------------------------------------------
        # Expands to fill the remaining space
        self.right_frame = tk.Frame(main_frame, bg=self.bg_main)
        self.right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        # --- Tab Configuration (Notebook) ---
        self.notebook = ttk.Notebook(self.left_frame)
        self.notebook.pack(fill="both", expand=True, padx=2, pady=2)

        # Define Tab Container Frames
        self.tab_preset = tk.Frame(self.notebook, bg=self.bg_main)
        self.tab_scenery = tk.Frame(self.notebook, bg=self.bg_main)
        self.tab_trajectory = tk.Frame(self.notebook, bg=self.bg_main)
        self.tab_env = tk.Frame(self.notebook, bg=self.bg_main)

        # Add Tabs to Notebook
        self.notebook.add(self.tab_preset, text="Presets")
        self.notebook.add(self.tab_scenery, text="Airport")
        self.notebook.add(self.tab_trajectory, text="Trajectory")
        self.notebook.add(self.tab_env, text="Environment")

        # --- Section Initialization ---
        # Each section is initialized via helper methods to maintain clean code 
        # and consistent styling (LabelFrames, Colors).

        # Tab 0: Presets
        self.metadataFileReader = self.__initializeMetadataSection(
            "Load Metadata Files",
            parent=self.tab_preset,
            bg_color="#e8e8ff" # Light Purple
        )
        
        # Tab 1: Airport (Pastel Blue Theme)
        self.airport_entries = self.__initializeInputSection(
            "Airport Parameters",
            ["Airport Name", "ICAO Code", "Runway Name", "Width", "Length", "Heading", "Latitude", "Longitude",
             "Altitude", "Start Height", "End Height"],
            bg_color="#e6f7ff",
            parent=self.tab_scenery,
            load_command=self.loadAirport,
            save_command=self.saveAirport
        )

        # Tab 2: Trajectory (Red & Green Theme)
        self.point_entries = self.__initializeInputSection(
            "Point Generation Parameters",
            ["Apex X", "Apex Y", "Apex Z", "Lateral Angle Left", "Lateral Angle Right",
             "Vertical Min Angle", "Vertical Max Angle", "Maximum Distance", "Number of Points"],
            bg_color="#ffede6", # Pastel Red
            parent=self.tab_trajectory,
            load_command=self.loadParameters,
            save_command=self.saveParameters
        )

        self.angle_entries = self.__initializeInputSection(
            "Angle Parameters (Pitch, Yaw, Bank)",
            ["Pitch Min", "Pitch Max", "Yaw Min", "Yaw Max", "Bank Min", "Bank Max"],
            bg_color="#e6ffe6", # Pastel Green
            parent=self.tab_trajectory,
            load_command=self.loadAngles,
            save_command=self.saveAngles
        )

        # Distribution Settings (Custom UI Layout)
        self.__initializeDistributionSection(self.tab_trajectory, bg_color="#f3e6ff") # Lavender


        # Tab 3: Environment (Orange & Purple Theme)
        self.time_entries = self.__initializeInputSection(
            "Time of Day",
            ["Hours", "Minutes"],
            bg_color="#FFEDE0", # Pastel Orange
            parent=self.tab_env,
            load_command=self.loadTime,
            save_command=self.saveTime
        )

        self.weather_frame = self.__initializeWeatherSection(
            "Weather Settings",
            bg_color="#E6E6FA", # Lavender
            parent=self.tab_env
        )

        # --- Right Side Setup (Plots) ---
        self.right_frame_top = tk.Frame(self.right_frame, bg=self.bg_main)
        self.right_frame_top.pack(side="top", fill="both", expand=True, padx=5, pady=5)

        self.right_frame_bottom = tk.Frame(self.right_frame, bg=self.bg_main)
        self.right_frame_bottom.pack(side="bottom", fill="x", padx=5, pady=5)

        # ----------------------------------------------------------------
        # 3D Plot Container
        # ----------------------------------------------------------------
        # Figure size is now configurable via settings
        plot_figsize = self.settings_manager.get("ui_layout", "plot_figsize")
        plot_frame = tk.Frame(self.right_frame_top, bg=self.bg_main)
        plot_frame.pack(side="left", fill="both", expand=True, padx=5)

        # Matplotlib Initialization
        self.fig = plt.figure(figsize=plot_figsize)
        self.ax = self.fig.add_subplot(111, projection="3d")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # ----------------------------------------------------------------
        # Right Sidebar (Legend & 2D Distribution Plot)
        # ----------------------------------------------------------------
        # Width is now configurable via settings
        right_sidebar_width = self.settings_manager.get("ui_layout", "right_sidebar_width")
        right_sidebar = tk.Frame(self.right_frame_top, bg=self.bg_main, width=right_sidebar_width)
        right_sidebar.pack(side="right", fill="y", padx=5)
        right_sidebar.pack_propagate(False)

        # Text Legend
        desc_frame = tk.LabelFrame(right_sidebar, text="Plot Legend", font=("Helvetica", 10, "bold"),
                                   bg="#e6f7ff", fg="#333")
        desc_frame.pack(fill="x", pady=5)
        
        self.plot_description = tk.Text(desc_frame, wrap="word", height=8, width=30, 
                                        font=("Helvetica", 9), bg="white")
        self.plot_description.pack(fill="both", expand=True, padx=5, pady=5)
        self.plot_description.insert("1.0", "Waiting for data generation...")

        # 2D Distribution Preview
        dist_frame = tk.LabelFrame(right_sidebar, text="2D Distribution", font=("Helvetica", 10, "bold"),
                                   bg="#ffede6", fg="#333")
        dist_frame.pack(fill="both", expand=True, pady=5)

        # Figure size is now configurable via settings
        dist_figsize = self.settings_manager.get("ui_layout", "dist_figsize")
        self.dist_fig, self.dist_ax = plt.subplots(figsize=dist_figsize)
        self.dist_canvas = FigureCanvasTkAgg(self.dist_fig, master=dist_frame)
        self.dist_canvas.get_tk_widget().pack(fill="both", expand=True)

        # ----------------------------------------------------------------
        # Bottom Control Bar
        # ----------------------------------------------------------------
        # Contains action buttons and settings access
        self.__setupButtons(self.right_frame_bottom)
        self.__setupStatusBar(main_frame)
        
        # ----------------------------------------------------------------
        # Settings Button (Bottom Left Corner)
        # ----------------------------------------------------------------
        # Provides access to the settings popup dialog
        self._create_settings_button(main_frame)
        
        # Bind Events for Auto-Updates
        self._bind_automatic_updates()
        self._trigger_update()


    # =========================================================================
    # CORE LOGIC & EVENT HANDLING
    # =========================================================================

    def _trigger_update(self, event=None):
        """
        Triggered whenever a UI value changes.
        Calls the parent controller to regenerate sample data (if available)
        and updates the 2D distribution preview.
        """
        if hasattr(self.parent, "generateSampleDataset"):
            self.parent.generateSampleDataset(silent=True)
            self.status_var.set("Preview updated")
        self.__plotSamplingPointDistribution()

    def _bind_automatic_updates(self):
        """
        Binds 'FocusOut' (user leaves field) and 'Return' (user hits enter)
        events to all input fields to trigger auto-updates.
        """
        entry_groups = [self.airport_entries, self.point_entries, self.angle_entries, self.time_entries]
        for group in entry_groups:
            for entry in group.values():
                entry.bind("<FocusOut>", self._trigger_update)
                entry.bind("<Return>", self._trigger_update)
        # Bind Combobox selection event
        self.distribution_menu.bind("<<ComboboxSelected>>", self._trigger_update)

    # =========================================================================
    # UI COMPONENT FACTORIES (Helper Methods)
    # =========================================================================

    def __initializeInputSection(self, title, fields, bg_color, parent, save_command=None, load_command=None):
        """
        Creates a standardized section with LabelFrame, input fields, and optional Save/Load buttons.
        
        :param title: Title of the LabelFrame.
        :param fields: List of strings (field names).
        :param bg_color: Background color (Hex).
        :param parent: Parent widget.
        :return: Dictionary containing the created Entry widgets { "FieldName": EntryObject }.
        """
        section_frame = tk.LabelFrame(parent, text=title, font=("Helvetica", 10, "bold"), 
                                      bg=bg_color, fg="#333")
        section_frame.pack(fill="x", padx=3, pady=3)

        section_entries = {}
        total_fields = len(fields)

        for idx, field in enumerate(fields):
            row = tk.Frame(section_frame, bg=bg_color)
            # Grid layout: 2 columns logic (idx // 2, idx % 2)
            row.grid(row=idx // 2, column=idx % 2, padx=2, pady=2, sticky="w")

            tk.Label(row, text=field, bg=bg_color, anchor="w", width=13, font=("Helvetica", 9)).pack(side="left")
            
            entry = ttk.Entry(row, width=9) # Small width to fit the sidebar
            entry.pack(side="right", fill="x")
            section_entries[field] = entry

        if save_command and load_command:
            button_frame = tk.Frame(section_frame, bg=bg_color)
            button_frame.grid(row=(total_fields // 2) + 1, columnspan=2, pady=5)
            ttk.Button(button_frame, text="Save", command=save_command).pack(side="left", padx=5)
            ttk.Button(button_frame, text="Load", command=load_command).pack(side="right", padx=5)

        return section_entries
    
    def __initializeMetadataSection(self, title, parent, bg_color):
        """Creates the Metadata loading section."""
        section_frame = tk.LabelFrame(parent, text=title, font=("Helvetica", 10, "bold"),
                                    bg=bg_color, fg="#333")
        section_frame.pack(fill="x", padx=3, pady=3)

        self.metadata_path_var = tk.StringVar()
        row = tk.Frame(section_frame, bg=bg_color)
        row.pack(fill="x", padx=2, pady=2)

        entry = ttk.Entry(row, textvariable=self.metadata_path_var, width=18)
        entry.pack(side="left", fill="x", expand=True)

        def browse_folder():
            folder = filedialog.askdirectory(title="Select Folder")
            if folder: self.metadata_path_var.set(folder)

        ttk.Button(row, text="Browse", command=browse_folder).pack(side="left", padx=4)

        button_row = tk.Frame(section_frame, bg=bg_color)
        button_row.pack(fill="x", pady=4)

        ttk.Button(button_row, text="Generate From Folder",
                command=lambda: self.generateImagesFromFolder(self.metadata_path_var.get())
                ).pack(side="left", padx=2)

        return section_frame

    def __initializeDistributionSection(self, parent, bg_color):
        """Creates the custom layout for Distribution settings (Combobox + Checkboxes)."""
        dummy_frame = tk.LabelFrame(parent, text="Distribution Settings",
                                         font=("Helvetica", 10, "bold"), bg=bg_color, fg="#333")
        dummy_frame.pack(fill="x", padx=3, pady=3)

        row1 = tk.Frame(dummy_frame, bg=bg_color)
        row1.pack(fill="x", padx=5, pady=2)
        row2 = tk.Frame(dummy_frame, bg=bg_color)
        row2.pack(fill="x", padx=5, pady=2)

        # Dropdown for Distribution Type
        tk.Label(row1, text="Distribution:", bg=bg_color, font=("Helvetica", 9)).pack(side="left", padx=5)
        self.distribution_var = tk.StringVar(value="Normal Distribution")
        
        self.distribution_menu = ttk.Combobox(row1, textvariable=self.distribution_var,
                                         values=["Normal Distribution", "Parabel", "Exponentiell"],
                                         state="readonly", width=14)
        self.distribution_menu.pack(side="left", padx=5)
        
        # Checkboxes for Axis application
        tk.Label(row2, text="Apply to:", bg=bg_color, font=("Helvetica", 9)).pack(side="left", padx=5)
        self.apply_x = tk.BooleanVar(value=True)
        self.apply_y = tk.BooleanVar(value=True)
        
        tk.Checkbutton(row2, text="X-Axis", variable=self.apply_x, bg=bg_color, 
                       command=self._trigger_update).pack(side="left", padx=5)
        tk.Checkbutton(row2, text="Y-Axis", variable=self.apply_y, bg=bg_color, 
                       command=self._trigger_update).pack(side="left", padx=5)

    def __initializeWeatherSection(self, title, parent, bg_color):
        """Creates the Weather selection dropdown."""
        section_frame = tk.LabelFrame(parent, text=title, font=("Helvetica", 10, "bold"), 
                                      bg=bg_color, fg="#333")
        section_frame.pack(fill="x", padx=3, pady=3)
        
        row = tk.Frame(section_frame, bg=bg_color)
        row.pack(fill="x", padx=5, pady=5)

        tk.Label(row, text="Condition:", bg=bg_color, anchor="w", font=("Helvetica", 9)).pack(side="left", padx=5)

        self.weather_var = tk.StringVar(value="Clear Skies")
        weather_options = ["Clear Skies", "Few Clouds", "Scattered Clouds", "Broken Clouds", "High Level Clouds", "Overcast", "Rain", "Snow", "Light Thunderstorm"]
        
        self.weather_menu = ttk.Combobox(row, textvariable=self.weather_var, 
                                         values=weather_options, state="readonly", width=15)
        self.weather_menu.pack(side="right", fill="x", padx=5)
        self.weather_menu.bind("<<ComboboxSelected>>", self._trigger_update)

        return section_frame

    def __setupButtons(self, frame):
        """Sets up the bottom control area (Data Creation buttons)."""
        separator = ttk.Separator(frame, orient="horizontal")
        separator.pack(fill="x", pady=10)

        main_section_frame = tk.Frame(frame, bg="#f0f4f8")
        main_section_frame.pack(fill="x", padx=10, pady=5)

        labeling_data_frame = tk.LabelFrame(main_section_frame, text="Data Creation",
                                            font=("Helvetica", 10, "bold"), bg="#ececec", fg="#333")
        labeling_data_frame.pack(side="left", expand=True, fill="both", padx=5, pady=0)

        labeling_data_row = tk.Frame(labeling_data_frame, bg="#ececec")
        labeling_data_row.pack(anchor="w", padx=10, pady=5)
        
        self.labeling_var = tk.BooleanVar(value=False)
        self.labeling_exclImg = tk.BooleanVar(value=False)
        self.validation_var = tk.BooleanVar(value=False)

        tk.Checkbutton(labeling_data_row, text="Enable Labeling", variable=self.labeling_var, bg="#ececec").pack(side="left", padx=(0, 10))
        tk.Checkbutton(labeling_data_row, text="Visual Overlay", variable=self.validation_var, bg="#ececec").pack(side="left", padx=(0, 10))
        tk.Checkbutton(labeling_data_row, text="Exclude Images", variable=self.labeling_exclImg, bg="#ececec").pack(side="left", padx=(0, 10))
        
        ttk.Button(labeling_data_row, text="Create Data", command=self.run_generation).pack(side="left")
    
    def __setupStatusBar(self, parent):
        """
        Adds a status bar at the very bottom of the window.
        
        The status bar displays the current application state and
        feedback messages for user actions.
        """
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(self.parent, textvariable=self.status_var, relief="sunken", anchor="w", bg="#e0e0e0")
        status_bar.pack(side="bottom", fill="x")
    
    def _create_settings_button(self, parent):
        """
        Creates the settings button in the bottom-left corner.
        
        This button opens the SettingsPopup dialog when clicked,
        allowing users to configure application-wide settings.
        
        Args:
            parent: The parent frame to place the button in
        """
        # Create a frame to hold the settings button
        # Positioned at the bottom-left of the main frame
        settings_frame = tk.Frame(parent, bg=self.bg_main)
        settings_frame.place(relx=0.0, rely=1.0, anchor="sw", x=10, y=-30)
        
        # Settings button with gear icon
        settings_btn = tk.Button(
            settings_frame,
            text="⚙ Settings",
            font=("Helvetica", 9),
            bg="#e0e0e0",
            fg="#333",
            relief="raised",
            padx=10,
            pady=2,
            command=self._open_settings_popup
        )
        settings_btn.pack(side="left")
        
        # Add hover effect
        def on_enter(e):
            settings_btn.config(bg="#d0d0d0", relief="groove")
        
        def on_leave(e):
            settings_btn.config(bg="#e0e0e0", relief="raised")
        
        settings_btn.bind("<Enter>", on_enter)
        settings_btn.bind("<Leave>", on_leave)
    
    def _open_settings_popup(self):
        """
        Opens the settings popup dialog.
        
        Creates and displays the SettingsPopup window with a callback
        to handle settings changes after saving.
        """
        # Create the popup with a callback for when settings are saved
        SettingsPopup(self.parent, on_save_callback=self._on_settings_changed)
    
    def _on_settings_changed(self):
        """
        Callback function called when settings are saved.
        
        This method is triggered after the user saves settings in the
        SettingsPopup. It updates the UI to reflect the new settings.
        """
        # Update status bar
        self.status_var.set("Settings updated - Some changes may require restart")
        
        # Refresh the plot with new settings
        if hasattr(self.parent, "generateSampleDataset"):
            self.parent.generateSampleDataset(silent=True)

    def run_generation(self):
        """Executes the data generation process in the parent controller."""
        self.status_var.set("Generating Data...")
        self.update_idletasks() # Force UI refresh
        try:
            self.parent.generateData()
            self.status_var.set("Generation Complete")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    # =========================================================================
    # PLOTTING FUNCTIONS
    # =========================================================================

    def refreshPlot(self, points, airport, apex):
        """
        Updates the 3D Plot with new data (Runway, Points, Apex).
        
        This method clears the current plot and redraws all elements
        using the current settings for visualization parameters.
        
        Args:
            points: numpy array of 3D sampling points
            airport: RunwayGeometryCalculator instance with runway data
            apex: Tuple of (x, y, z) coordinates for the cone apex
        """
        self.ax.clear()
        legend_entries = []

        # Get plot settings from SettingsManager
        point_size = self.settings_manager.get("plot", "point_size")
        point_alpha = self.settings_manager.get("plot", "point_alpha")
        runway_alpha = self.settings_manager.get("plot", "runway_alpha")
        apex_point_size = self.settings_manager.get("plot", "apex_point_size")

        # 1. Plot Apex (Curve Center)
        if apex is not None:
            self.ax.scatter(apex[0], apex[1], apex[2], color="red", s=apex_point_size, label="Apex")
            legend_entries.append(f"🔴 Apex: ({apex[0]:.1f}, {apex[1]:.1f}, {apex[2]:.1f})")

        # 2. Plot Generated Points
        if points is not None:
            self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=point_size, c="blue", alpha=point_alpha, label="Points")
            legend_entries.append(f"🔵 Points: {len(points)}")

        # 3. Plot Runway Geometry
        if airport:
            corners = airport.calculateRunwayCorners()
            runway_points = [
                (corners["top_left"][0], corners["top_left"][1], 0),
                (corners["top_right"][0], corners["top_right"][1], 0),
                (corners["bottom_right"][0], corners["bottom_right"][1], 0),
                (corners["bottom_left"][0], corners["bottom_left"][1], 0)
            ]
            self.ax.scatter([p[0] for p in runway_points], [p[1] for p in runway_points],
                            [p[2] for p in runway_points], c="red", s=20)
            
            poly = Poly3DCollection([runway_points], color="gray", alpha=runway_alpha)
            self.ax.add_collection3d(poly)
            legend_entries.append(f"⬜️ Runway: {airport.runway_width}x{airport.runway_length}m")

        # Set labels and redraw
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.canvas.draw()
        
        # Update Text Legend
        self.plot_description.config(state="normal")
        self.plot_description.delete("1.0", tk.END)
        for entry in legend_entries:
            self.plot_description.insert("end", entry + "\n")
        self.plot_description.config(state="disabled")

        self.__plotSamplingPointDistribution()

    def __plotSamplingPointDistribution(self):
        """
        Updates the small 2D plot showing the selected distribution curve.
        """
        distribution = self.distribution_var.get()
        apply_y = self.apply_y.get()

        self.dist_ax.clear()

        x = np.linspace(-1, 1, 100)
        y = np.zeros_like(x)
        title = ""
        color = "#007acc"

        if distribution == "Normal Distribution":
            y = np.ones_like(x)
            title = "Uniform"
        elif distribution == "Parabel":
            y = x**2
            title = "Parabolic"
        elif distribution == "Exponentiell":
            if apply_y:
                y = np.exp(-4 * x**2)
                title = "Exponential (Center)"
            else:
                x = np.linspace(0, 3, 100)
                y = np.exp(-x)
                title = "Exponential (Decay)"

        self.dist_ax.plot(x, y, color=color, linewidth=2)
        self.dist_ax.fill_between(x, y, color=color, alpha=0.3)
        self.dist_ax.set_title(title, fontsize=9)
        self.dist_ax.grid(True, linestyle=":", alpha=0.6)
        self.dist_ax.axis('on') 

        self.dist_canvas.draw()
    
    # =========================================================================
    # DATA PERSISTENCE (Save/Load)
    # =========================================================================

    def saveAirport(self):
        """Reads inputs, creates a Runway object, and saves it to JSON."""
        try:
            vals = {k: v.get() for k, v in self.airport_entries.items()}
            if not all(vals.values()): raise ValueError("All fields must be filled!")
            num_keys = ["Width", "Length", "Heading", "Latitude", "Longitude", "Altitude", "Start Height", "End Height"]
            args = {k: float(v) for k, v in vals.items() if k in num_keys}
            
            self.airport = RunwayGeometryCalculator(
                vals["Airport Name"], vals["ICAO Code"], vals["Runway Name"],
                args["Width"], args["Length"], args["Heading"], 
                args["Latitude"], args["Longitude"], args["Altitude"], 
                args["Start Height"], args["End Height"], {}
            )
            file = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
            if file:
                self.airport.saveAirport(file)
                self.status_var.set(f"Saved: {os.path.basename(file)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def loadAirport(self):
        """Loads airport JSON and populates input fields."""
        try:
            file = filedialog.askopenfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
            if not file: return
            self.airport = RunwayGeometryCalculator.loadAirport(file)
            self.__populateEntryFields(self.airport_entries, {
                "Airport Name": self.airport.name,
                "ICAO Code": self.airport.icao_code,
                "Runway Name": self.airport.runway_name,
                "Width": self.airport.runway_width,
                "Length": self.airport.runway_length,
                "Heading": self.airport.runway_heading,
                "Latitude": self.airport.runway_center["latitude"],
                "Longitude": self.airport.runway_center["longitude"],
                "Altitude": self.airport.runway_center["altitude"],
                "Start Height": self.airport.start_height,
                "End Height": self.airport.end_height,
            })
            self._trigger_update()
            self.status_var.set(f"Loaded Airport: {self.airport.name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load airport: {e}")

    def saveParameters(self):
        """Saves Point Generation parameters to JSON."""
        params = {key: self.point_entries[key].get() for key in self.point_entries}
        file = JSONManager.save_to_file(params)
        if file: self.status_var.set("Parameters saved.")

    def loadParameters(self):
        """Loads Point Generation parameters from JSON."""
        params = JSONManager.load_from_file()
        if params: 
            self.__populateEntryFields(self.point_entries, params)
            self._trigger_update()
            
    def saveAngles(self):
        """Saves Angle parameters to JSON."""
        angles = {key: self.angle_entries[key].get() for key in self.angle_entries}
        file = JSONManager.save_to_file(angles)

    def loadAngles(self):
        """Loads Angle parameters from JSON."""
        angles = JSONManager.load_from_file()
        if angles: 
            self.__populateEntryFields(self.angle_entries, angles)
            self._trigger_update()

    def saveTime(self):
        """Saves Time settings to JSON."""
        times = {key: self.time_entries[key].get() for key in self.time_entries}
        file = JSONManager.save_to_file(times)

    def loadTime(self):
        """Loads Time settings from JSON."""
        times = JSONManager.load_from_file()
        if times: self.__populateEntryFields(self.time_entries, times)
        
    def generateImagesFromFolder(self, folder_path):
        """Triggers batch generation from a folder of JSON files."""
        if not folder_path:
            folder_path = filedialog.askdirectory(title="Select folder with JSON files")
            if not folder_path: return
        try:
            self.status_var.set("Batch processing images...")
            reader = MetadataFileReader(file_path="", screenshot_dir=folder_path)
            out_paths = reader.process_folder(folder_path, use_sim=True)
            self.status_var.set(f"Batch complete. {len(out_paths)} images.")
            messagebox.showinfo("Success", f"{len(out_paths)} images generated.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {e}")

    def __populateEntryFields(self, entry_fields, values):
        """Helper to fill multiple Entry widgets from a dictionary."""
        for key, value in values.items():
            if key in entry_fields:
                entry_fields[key].delete(0, tk.END)
                entry_fields[key].insert(0, str(value))