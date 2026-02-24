"""
SettingsPopup - Settings Configuration Dialog
==============================================

This module provides a popup dialog window for configuring application settings.
The dialog follows the same visual style as the main ScenAIro UI with:
    - Pastel color themes for different sections
    - Organized tabs for logical grouping
    - Save/Cancel/Reset functionality

The settings are organized into the following categories:
    - Paths: File saving locations
    - Screen: Screenshot resolution settings
    - Camera: Field of view settings
    - UI Layout: Sidebar and panel dimensions
    - Plot: Visualization parameters

Author: ScenAIro Team
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Callable, Optional

# Import the settings manager
from tools.SettingsManager import SettingsManager


class SettingsPopup(tk.Toplevel):
    """
    Settings configuration popup dialog.
    
    This dialog provides a user-friendly interface for modifying all
    configurable application settings. It follows the same design language
    as the main ScenAIro UI with pastel colors and organized sections.
    
    Attributes:
        settings_manager: Reference to the SettingsManager singleton
        on_save_callback: Optional callback function to execute after saving
        entries: Dictionary storing all input field widgets
    """
    
    # =========================================================================
    # COLOR THEME CONSTANTS
    # =========================================================================
    # These colors match the main ScenAIro UI design language
    
    BG_MAIN = "#f0f4f8"        # Main background (light grey-blue)
    BG_PATHS = "#e6f7ff"       # Paths section (pastel blue)
    BG_SCREEN = "#ffede6"      # Screen section (pastel red/orange)
    BG_CAMERA = "#e6ffe6"      # Camera section (pastel green)
    BG_LAYOUT = "#f3e6ff"      # Layout section (lavender)
    BG_PLOT = "#FFEDE0"        # Plot section (pastel orange)
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def __init__(self, parent, on_save_callback: Optional[Callable] = None):
        """
        Initialize the settings popup dialog.
        
        Args:
            parent: The parent window (main application window)
            on_save_callback: Optional callback function to execute after settings are saved
        """
        super().__init__(parent)
        
        # Store references
        self.settings_manager = SettingsManager()
        self.on_save_callback = on_save_callback
        self.entries = {}
        
        # ----------------------------------------------------------------
        # Window Configuration
        # ----------------------------------------------------------------
        self.title("Settings")
        self.geometry("500x600")
        self.configure(bg=self.BG_MAIN)
        self.resizable(False, False)
        
        # Make the popup modal (block interaction with parent)
        self.transient(parent)
        self.grab_set()
        
        # Center the popup on the parent window
        self._center_on_parent(parent)
        
        # Build the UI components
        self._create_widgets()
        
        # Load current settings into the UI
        self._load_current_settings()
        
        # Bind escape key to close
        self.bind("<Escape>", lambda e: self.destroy())
    
    def _center_on_parent(self, parent) -> None:
        """
        Center the popup window on the parent window.
        
        Args:
            parent: The parent window to center on
        """
        self.update_idletasks()
        
        # Get parent position and dimensions
        parent_x = parent.winfo_x()
        parent_y = parent.winfo_y()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()
        
        # Calculate center position
        popup_width = 500
        popup_height = 600
        x = parent_x + (parent_width - popup_width) // 2
        y = parent_y + (parent_height - popup_height) // 2
        
        self.geometry(f"{popup_width}x{popup_height}+{x}+{y}")
    
    # =========================================================================
    # UI CREATION METHODS
    # =========================================================================
    
    def _create_widgets(self) -> None:
        """
        Create all UI widgets for the settings dialog.
        
        The UI is organized into:
            1. Header with title
            2. Scrollable content area with setting sections
            3. Footer with action buttons
        """
        # ----------------------------------------------------------------
        # Header Section
        # ----------------------------------------------------------------
        header_frame = tk.Frame(self, bg=self.BG_MAIN)
        header_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        # Settings icon and title
        tk.Label(
            header_frame,
            text="⚙ Application Settings",
            font=("Helvetica", 14, "bold"),
            bg=self.BG_MAIN,
            fg="#333"
        ).pack(side="left")
        
        # ----------------------------------------------------------------
        # Content Area (Scrollable)
        # ----------------------------------------------------------------
        # Create a canvas with scrollbar for the settings content
        content_container = tk.Frame(self, bg=self.BG_MAIN)
        content_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Canvas for scrolling
        canvas = tk.Canvas(content_container, bg=self.BG_MAIN, highlightthickness=0)
        scrollbar = ttk.Scrollbar(content_container, orient="vertical", command=canvas.yview)
        
        # Frame inside canvas to hold content
        self.content_frame = tk.Frame(canvas, bg=self.BG_MAIN)
        
        # Configure canvas scrolling
        self.content_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.content_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Enable mouse wheel scrolling
        self._bind_mousewheel(canvas)
        
        # ----------------------------------------------------------------
        # Create Settings Sections
        # ----------------------------------------------------------------
        self._create_paths_section()
        self._create_screen_section()
        self._create_camera_section()
        self._create_layout_section()
        self._create_plot_section()
        
        # ----------------------------------------------------------------
        # Footer Section (Buttons)
        # ----------------------------------------------------------------
        self._create_footer()
    
    def _bind_mousewheel(self, canvas) -> None:
        """
        Bind mouse wheel events to the canvas for scrolling.
        
        Args:
            canvas: The canvas widget to enable scrolling on
        """
        # Windows and MacOS mouse wheel events
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
        # Unbind when window is closed
        self.bind("<Destroy>", lambda e: canvas.unbind_all("<MouseWheel>"))
    
    # =========================================================================
    # SECTION CREATION METHODS
    # =========================================================================
    
    def _create_paths_section(self) -> None:
        """
        Create the Paths settings section.
        
        Contains settings for:
            - Screenshot save path
            - Config directory path
        """
        section = self._create_section(
            "📁 File Paths",
            self.BG_PATHS,
            "Configure file saving locations"
        )
        
        # Screenshot Path
        self._create_path_entry(
            section,
            "Screenshot Path",
            "screenshot_path",
            "Directory where screenshots will be saved",
            self.BG_PATHS
        )
    
    def _create_screen_section(self) -> None:
        """
        Create the Screen settings section.
        
        Contains settings for:
            - Screen width (pixels)
            - Screen height (pixels)
        """
        section = self._create_section(
            "🖥️ Screen Resolution",
            self.BG_SCREEN,
            "Screenshot capture resolution"
        )
        
        # Screen Width
        self._create_entry_row(
            section,
            "Width (px)",
            "screen_width",
            "Screenshot width in pixels",
            self.BG_SCREEN
        )
        
        # Screen Height
        self._create_entry_row(
            section,
            "Height (px)",
            "screen_height",
            "Screenshot height in pixels",
            self.BG_SCREEN
        )
    
    def _create_camera_section(self) -> None:
        """
        Create the Camera settings section.
        
        Contains settings for:
            - Vertical field of view (radians)
        """
        section = self._create_section(
            "📷 Camera Settings",
            self.BG_CAMERA,
            "Camera and field of view parameters"
        )
        
        # Vertical FOV
        self._create_entry_row(
            section,
            "Vertical FOV (rad)",
            "vertical_fov",
            "Vertical field of view in radians (default: 0.8 ≈ 45.8°)",
            self.BG_CAMERA
        )
    
    def _create_layout_section(self) -> None:
        """
        Create the UI Layout settings section.
        
        Contains settings for:
            - Left sidebar width
            - Right sidebar width
        """
        section = self._create_section(
            "📐 UI Layout",
            self.BG_LAYOUT,
            "Sidebar and panel dimensions"
        )
        
        # Left Sidebar Width
        self._create_entry_row(
            section,
            "Left Sidebar (px)",
            "left_sidebar_width",
            "Width of the left configuration panel",
            self.BG_LAYOUT
        )
        
        # Right Sidebar Width
        self._create_entry_row(
            section,
            "Right Sidebar (px)",
            "right_sidebar_width",
            "Width of the right legend panel",
            self.BG_LAYOUT
        )
    
    def _create_plot_section(self) -> None:
        """
        Create the Plot settings section.
        
        Contains settings for:
            - Point size
            - Point transparency (alpha)
            - Runway transparency
            - Apex marker size
        """
        section = self._create_section(
            "📊 Plot Settings",
            self.BG_PLOT,
            "3D visualization parameters"
        )
        
        # Point Size
        self._create_entry_row(
            section,
            "Point Size",
            "point_size",
            "Size of scatter plot points",
            self.BG_PLOT
        )
        
        # Point Alpha
        self._create_entry_row(
            section,
            "Point Alpha (0-1)",
            "point_alpha",
            "Point transparency (0 = invisible, 1 = solid)",
            self.BG_PLOT
        )
        
        # Runway Alpha
        self._create_entry_row(
            section,
            "Runway Alpha (0-1)",
            "runway_alpha",
            "Runway polygon transparency",
            self.BG_PLOT
        )
        
        # Apex Point Size
        self._create_entry_row(
            section,
            "Apex Marker Size",
            "apex_point_size",
            "Size of the apex marker in the 3D plot",
            self.BG_PLOT
        )
    
    # =========================================================================
    # HELPER METHODS FOR UI CREATION
    # =========================================================================
    
    def _create_section(self, title: str, bg_color: str, description: str) -> tk.LabelFrame:
        """
        Create a labeled section frame with description.
        
        Args:
            title: Section title text
            bg_color: Background color for the section
            description: Brief description of the section
            
        Returns:
            tk.LabelFrame: The created section frame
        """
        frame = tk.LabelFrame(
            self.content_frame,
            text=title,
            font=("Helvetica", 10, "bold"),
            bg=bg_color,
            fg="#333",
            padx=10,
            pady=5
        )
        frame.pack(fill="x", padx=5, pady=5)
        
        # Description label
        tk.Label(
            frame,
            text=description,
            font=("Helvetica", 8, "italic"),
            bg=bg_color,
            fg="#666"
        ).pack(anchor="w", pady=(0, 5))
        
        return frame
    
    def _create_entry_row(
        self,
        parent: tk.Frame,
        label: str,
        key: str,
        tooltip: str,
        bg_color: str
    ) -> None:
        """
        Create a labeled entry field row.
        
        Args:
            parent: Parent frame to add the row to
            label: Label text for the entry
            key: Unique key for storing the entry widget
            tooltip: Tooltip text (shown as small help text)
            bg_color: Background color for the row
        """
        row = tk.Frame(parent, bg=bg_color)
        row.pack(fill="x", pady=2)
        
        # Label
        tk.Label(
            row,
            text=label,
            bg=bg_color,
            width=18,
            anchor="w",
            font=("Helvetica", 9)
        ).pack(side="left", padx=(0, 5))
        
        # Entry field
        entry = ttk.Entry(row, width=20)
        entry.pack(side="right", fill="x", expand=True)
        
        # Store entry reference
        self.entries[key] = entry
    
    def _create_path_entry(
        self,
        parent: tk.Frame,
        label: str,
        key: str,
        tooltip: str,
        bg_color: str
    ) -> None:
        """
        Create a path entry field with browse button.
        
        Args:
            parent: Parent frame to add the row to
            label: Label text for the entry
            key: Unique key for storing the entry widget
            tooltip: Tooltip text
            bg_color: Background color for the row
        """
        row = tk.Frame(parent, bg=bg_color)
        row.pack(fill="x", pady=2)
        
        # Label
        tk.Label(
            row,
            text=label,
            bg=bg_color,
            width=18,
            anchor="w",
            font=("Helvetica", 9)
        ).pack(side="left", padx=(0, 5))
        
        # Entry field
        entry = ttk.Entry(row, width=15)
        entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        # Browse button
        def browse():
            path = filedialog.askdirectory(title=f"Select {label}")
            if path:
                entry.delete(0, tk.END)
                entry.insert(0, path)
        
        ttk.Button(row, text="Browse", command=browse, width=8).pack(side="right")
        
        # Store entry reference
        self.entries[key] = entry
    
    def _create_footer(self) -> None:
        """
        Create the footer section with action buttons.
        
        Contains:
            - Save button: Save changes and close
            - Reset button: Reset to defaults
            - Cancel button: Discard changes and close
        """
        footer_frame = tk.Frame(self, bg=self.BG_MAIN)
        footer_frame.pack(fill="x", padx=10, pady=10)
        
        # Separator
        ttk.Separator(footer_frame, orient="horizontal").pack(fill="x", pady=(0, 10))
        
        # Button container
        button_frame = tk.Frame(footer_frame, bg=self.BG_MAIN)
        button_frame.pack(fill="x")
        
        # Reset button (left side)
        ttk.Button(
            button_frame,
            text="Reset to Defaults",
            command=self._reset_to_defaults
        ).pack(side="left", padx=5)
        
        # Cancel and Save buttons (right side)
        ttk.Button(
            button_frame,
            text="Cancel",
            command=self.destroy
        ).pack(side="right", padx=5)
        
        ttk.Button(
            button_frame,
            text="Save",
            command=self._save_and_close
        ).pack(side="right", padx=5)
    
    # =========================================================================
    # DATA LOADING AND SAVING
    # =========================================================================
    
    def _load_current_settings(self) -> None:
        """
        Load current settings from SettingsManager into the UI fields.
        
        This method populates all entry fields with the current values
        from the settings manager.
        """
        # Paths
        self.entries["screenshot_path"].insert(0, 
            self.settings_manager.get("paths", "screenshot_path"))
        
        # Screen
        self.entries["screen_width"].insert(0, 
            str(self.settings_manager.get("screen", "width")))
        self.entries["screen_height"].insert(0, 
            str(self.settings_manager.get("screen", "height")))
        
        # Camera
        self.entries["vertical_fov"].insert(0, 
            str(self.settings_manager.get("camera", "vertical_fov_radians")))
        
        # UI Layout
        self.entries["left_sidebar_width"].insert(0, 
            str(self.settings_manager.get("ui_layout", "left_sidebar_width")))
        self.entries["right_sidebar_width"].insert(0, 
            str(self.settings_manager.get("ui_layout", "right_sidebar_width")))
        
        # Plot
        self.entries["point_size"].insert(0, 
            str(self.settings_manager.get("plot", "point_size")))
        self.entries["point_alpha"].insert(0, 
            str(self.settings_manager.get("plot", "point_alpha")))
        self.entries["runway_alpha"].insert(0, 
            str(self.settings_manager.get("plot", "runway_alpha")))
        self.entries["apex_point_size"].insert(0, 
            str(self.settings_manager.get("plot", "apex_point_size")))
    
    def _validate_inputs(self) -> bool:
        """
        Validate all input values before saving.
        
        Returns:
            bool: True if all inputs are valid, False otherwise
        """
        errors = []
        
        # Validate numeric fields
        numeric_fields = {
            "screen_width": ("screen", "width", int),
            "screen_height": ("screen", "height", int),
            "vertical_fov": ("camera", "vertical_fov_radians", float),
            "left_sidebar_width": ("ui_layout", "left_sidebar_width", int),
            "right_sidebar_width": ("ui_layout", "right_sidebar_width", int),
            "point_size": ("plot", "point_size", int),
            "point_alpha": ("plot", "point_alpha", float),
            "runway_alpha": ("plot", "runway_alpha", float),
            "apex_point_size": ("plot", "apex_point_size", int)
        }
        
        for entry_key, (category, setting_key, type_func) in numeric_fields.items():
            value = self.entries[entry_key].get().strip()
            try:
                converted = type_func(value)
                
                # Additional validation for alpha values (must be 0-1)
                if "alpha" in entry_key and (converted < 0 or converted > 1):
                    errors.append(f"{entry_key}: Value must be between 0 and 1")
                
                # Validate positive values
                if "width" in entry_key or "height" in entry_key or "size" in entry_key:
                    if converted <= 0:
                        errors.append(f"{entry_key}: Value must be positive")
                        
            except ValueError:
                errors.append(f"{entry_key}: Invalid {type_func.__name__} value")
        
        # Validate path exists (optional - just warn)
        path = self.entries["screenshot_path"].get().strip()
        import os
        if path and not os.path.exists(path):
            # Just a warning, not an error
            print(f"[Settings] Warning: Screenshot path does not exist: {path}")
        
        # Show errors if any
        if errors:
            messagebox.showerror(
                "Validation Error",
                "Please fix the following errors:\n\n" + "\n".join(errors)
            )
            return False
        
        return True
    
    def _save_and_close(self) -> None:
        """
        Validate inputs, save settings, and close the dialog.
        
        This method:
            1. Validates all input values
            2. Updates the settings manager with new values
            3. Saves settings to file
            4. Calls the on_save_callback if provided
            5. Closes the dialog
        """
        if not self._validate_inputs():
            return
        
        # Update settings manager with new values
        # Paths
        self.settings_manager.set("paths", "screenshot_path", 
            self.entries["screenshot_path"].get().strip())
        
        # Screen
        self.settings_manager.set("screen", "width", 
            int(self.entries["screen_width"].get().strip()))
        self.settings_manager.set("screen", "height", 
            int(self.entries["screen_height"].get().strip()))
        
        # Camera
        self.settings_manager.set("camera", "vertical_fov_radians", 
            float(self.entries["vertical_fov"].get().strip()))
        
        # UI Layout
        self.settings_manager.set("ui_layout", "left_sidebar_width", 
            int(self.entries["left_sidebar_width"].get().strip()))
        self.settings_manager.set("ui_layout", "right_sidebar_width", 
            int(self.entries["right_sidebar_width"].get().strip()))
        
        # Plot
        self.settings_manager.set("plot", "point_size", 
            int(self.entries["point_size"].get().strip()))
        self.settings_manager.set("plot", "point_alpha", 
            float(self.entries["point_alpha"].get().strip()))
        self.settings_manager.set("plot", "runway_alpha", 
            float(self.entries["runway_alpha"].get().strip()))
        self.settings_manager.set("plot", "apex_point_size", 
            int(self.entries["apex_point_size"].get().strip()))
        
        # Save to file
        if self.settings_manager.save_settings():
            messagebox.showinfo("Success", "Settings saved successfully!")
            
            # Call the callback if provided
            if self.on_save_callback:
                self.on_save_callback()
            
            self.destroy()
        else:
            messagebox.showerror("Error", "Failed to save settings. Check console for details.")
    
    def _reset_to_defaults(self) -> None:
        """
        Reset all settings to their default values.
        
        Prompts the user for confirmation before resetting.
        """
        if messagebox.askyesno(
            "Reset Settings",
            "Are you sure you want to reset all settings to their default values?"
        ):
            # Reset settings manager
            self.settings_manager.reset_to_defaults()
            
            # Clear and reload UI
            for entry in self.entries.values():
                entry.delete(0, tk.END)
            
            self._load_current_settings()
            
            messagebox.showinfo("Reset", "Settings have been reset to defaults.")
