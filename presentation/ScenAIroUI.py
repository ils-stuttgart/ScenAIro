import os
import tkinter as tk
from tkinter import ttk, messagebox, PhotoImage, filedialog
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tools.MetadataFileReader import MetadataFileReader
from tools import SamplingPointGenerator, AircraftPositioningAgent
from tools.RunwayGeometryCalculator import RunwayGeometryCalculator

class JSONManager:
    @staticmethod
    def save_to_file(data, filetypes=("JSON files", "*.json")):
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

    def __init__(self, parent):
        super().__init__(parent)
        self.airport = None
        self.parent = parent
        self.jsonmanager = JSONManager()

        # main-Frames: left (1/3) and right (2/3)
        main_frame = tk.Frame(self.parent, bg="#f0f4f8")
        main_frame.pack(fill="both", expand=True)

        self.left_frame = tk.Frame(main_frame, bg="#f0f4f8", width=500)
        self.left_frame.pack(side="left", fill="y", padx=5, pady=5)

        self.right_frame = tk.Frame(main_frame, bg="#f0f4f8")
        self.right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        # Input fields left frame:
        self.metadataFileReader = self.__initializeMetadataSection(
            "Load Metadata Files for Image regeneration",
            parent=self.left_frame,
            load_command=self.loadMetadataFiles,
            generate_command=self.generateImageFromMetadata,  
        )

        self.airport_entries = self.__initializeInputSection(
            "Airport Parameters",
            ["Airport Name", "ICAO Code", "Runway Name", "Width", "Length", "Heading", "Latitude", "Longitude",
             "Altitude", "Start Height", "End Height"],
            "#e6f7ff",
            parent=self.left_frame,
            load_command=self.loadAirport,
            save_command=self.saveAirport
        )

        self.point_entries = self.__initializeInputSection(
            "Point Generation Parameters",
            ["Apex X", "Apex Y", "Apex Z", "Lateral Angle Left", "Lateral Angle Right",
             "Vertical Min Angle", "Vertical Max Angle", "Maximum Distance", "Number of Points"],
            "#ffede6",
            parent=self.left_frame,
            load_command=self.loadParameters,
            save_command=self.saveParameters
        )

        self.angle_entries = self.__initializeInputSection(
            "Angle Parameters (Pitch, Yaw, Bank)",
            ["Pitch Min", "Pitch Max", "Yaw Min", "Yaw Max", "Bank Min", "Bank Max"],
            "#e6ffe6",
            parent=self.left_frame,
            load_command=self.loadAngles,
            save_command=self.saveAngles
        )

        self.time_entries = self.__initializeInputSection(
            "Time of Day",
            ["Hours", "Minutes"],
            "#FFEDE0",
            parent=self.left_frame,
            load_command=self.loadTime,
            save_command=self.saveTime
        )

        ## Distribution-Frame
        self.dummy_frame = tk.LabelFrame(self.left_frame, text="Distribution Settings",
                                         font=("Helvetica", 12, "bold"), bg="#f3e6ff", fg="#333")
        self.dummy_frame.pack(fill="x", padx=5, pady=10)

        # Division in 2 lines
        row1 = tk.Frame(self.dummy_frame, bg="#f3e6ff")
        row1.pack(fill="x", padx=5, pady=2)
        row2 = tk.Frame(self.dummy_frame, bg="#f3e6ff")
        row2.pack(fill="x", padx=5, pady=2)

        # Line 1: Dropdown-Menü
        tk.Label(row1, text="Distribution:", bg="#f3e6ff").pack(side="left", padx=5)
        self.distribution_var = tk.StringVar(value="Normal Distribution")
        
        # Combobox mit direktem Trigger
        self.distribution_menu = ttk.Combobox(row1, textvariable=self.distribution_var,
                                         values=["Normal Distribution", "Parabel", "Exponentiell"],
                                         state="readonly")
        self.distribution_menu.pack(side="left", padx=5)
        
        # Line 2: Checkboxes
        tk.Label(row2, text="Apply to:", bg="#f3e6ff").pack(side="left", padx=5)
        self.apply_x = tk.BooleanVar(value=True)
        self.apply_y = tk.BooleanVar(value=True)
        
        tk.Checkbutton(row2, text="X-Axis", variable=self.apply_x, bg="#f3e6ff", 
                       command=self._trigger_update).pack(side="left", padx=5)
        tk.Checkbutton(row2, text="Y-Axis", variable=self.apply_y, bg="#f3e6ff", 
                       command=self._trigger_update).pack(side="left", padx=5)


        ## Rechter Haupt-Frame
        self.right_frame_top = tk.Frame(self.right_frame, bg="#f0f4f8")
        self.right_frame_top.pack(side="top", fill="both", expand=True, padx=5, pady=5)

        self.right_frame_bottom = tk.Frame(self.right_frame, bg="#f0f4f8")
        self.right_frame_bottom.pack(side="bottom", fill="x", padx=5, pady=5)

        ## Plot-Frame
        plot_frame = tk.Frame(self.right_frame_top, bg="#f0f4f8")
        plot_frame.pack(side="left", fill="both", expand=True, padx=10, pady=(10, 0))

        self.fig = plt.figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Zusatzbereich rechts
        right_plot_frame = tk.Frame(self.right_frame_top, bg="#f0f4f8", width=250)
        right_plot_frame.pack(side="right", fill="both", expand=False, padx=5, pady=5)
        right_plot_frame.pack_propagate(False)

        description_frame = tk.LabelFrame(right_plot_frame, text="Plot Legend", font=("Helvetica", 12, "bold"),
                                          bg="#e6f7ff", fg="#333", height=40)
        description_frame.pack(fill="both", expand=True, padx=5, pady=5)
        description_frame.pack_propagate(False)

        self.plot_description = tk.Text(description_frame, wrap="word", bg="#ffffff", fg="#333",
                                        font=("Helvetica", 10))
        self.plot_description.pack(fill="both", expand=True, padx=5, pady=5)
        self.plot_description.insert("1.0", "Plot Information is going to be displayed here")

        distribution_frame = tk.LabelFrame(right_plot_frame, text="2D Distribution", font=("Helvetica", 12, "bold"),
                                           bg="#ffede6", fg="#333", height=60)
        distribution_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.dist_fig, self.dist_ax = plt.subplots(figsize=(4, 2))
        self.dist_canvas = FigureCanvasTkAgg(self.dist_fig, master=distribution_frame)
        self.dist_canvas.get_tk_widget().pack(fill="both", expand=True)

        self.__setupButtons(self.right_frame_bottom)
        self.__setPlotHeight()
        
        self._bind_automatic_updates()


    def _trigger_update(self, event=None):
        """Wird aufgerufen, wenn sich ein Wert ändert."""
        if hasattr(self.parent, "generateSampleDataset"):
            # HIER silent=True übergeben!
            self.parent.generateSampleDataset(silent=True)

    def _bind_automatic_updates(self):
        """Bindet alle Eingabefelder an den Trigger."""
        # Alle Entry-Gruppen durchgehen
        entry_groups = [self.airport_entries, self.point_entries, self.angle_entries]
        
        for group in entry_groups:
            for entry in group.values():
                # Update bei Fokus-Verlust (Tab oder Klick woanders)
                entry.bind("<FocusOut>", self._trigger_update)
                # Update bei Enter-Taste
                entry.bind("<Return>", self._trigger_update)
        
        # Combobox (Distribution) binden
        self.distribution_menu.bind("<<ComboboxSelected>>", self._trigger_update)
    # ---------------------------------------------

    def __setPlotHeight(self):
        self.plot_description.config(height=10)

    def __initializeInputSection(self, title, fields, bg_color, parent, save_command=None, load_command=None):
        section_frame = tk.LabelFrame(parent, text=title, font=("Helvetica", 10, "bold"), bg=bg_color, fg="#333")
        section_frame.pack(fill="x", padx=3, pady=3)
        section_frame.pack_propagate(False)
        section_frame.config(width=160)

        section_entries = {}
        total_fields = len(fields)

        for idx, field in enumerate(fields):
            row = tk.Frame(section_frame, bg=bg_color)
            row.grid(row=idx // 2, column=idx % 2, padx=2, pady=2, sticky="w")

            tk.Label(row, text=field, bg=bg_color, anchor="w", width=10).pack(side="left")
            entry = ttk.Entry(row, width=8)
            entry.pack(side="right", fill="x")
            section_entries[field] = entry

        if save_command and load_command:
            button_frame = tk.Frame(section_frame, bg=bg_color)
            button_frame.grid(row=(total_fields // 2) + 1, columnspan=1, pady=5)
            ttk.Button(button_frame, text="Save", command=save_command).pack(side="left", padx=5)
            ttk.Button(button_frame, text="Load", command=load_command).pack(side="right", padx=5)

        return section_entries
    
    def __initializeMetadataSection(self, title, parent, load_command, generate_command):
        bg_color = "#e8e8ff"
        section_frame = tk.LabelFrame(parent, text=title, font=("Helvetica", 10, "bold"),
                                    bg=bg_color, fg="#333")
        section_frame.pack(fill="x", padx=3, pady=3)

        self.metadata_path_var = tk.StringVar()
        row = tk.Frame(section_frame, bg=bg_color)
        row.pack(fill="x", padx=2, pady=2)

        entry = ttk.Entry(row, textvariable=self.metadata_path_var, width=22)
        entry.pack(side="left", fill="x", expand=True)

        def browse_folder():
            folder = filedialog.askdirectory(title="Select Folder Containing JSON Files")
            if folder: self.metadata_path_var.set(folder)

        ttk.Button(row, text="Browse", command=browse_folder).pack(side="left", padx=4)

        button_row = tk.Frame(section_frame, bg=bg_color)
        button_row.pack(fill="x", pady=4)

        ttk.Button(button_row, text="Generate From Folder",
                command=lambda: self.generateImagesFromFolder(self.metadata_path_var.get())
                ).pack(side="left", padx=2)

        return section_frame

    def __setupButtons(self, frame):
        # Trennlinie horizontal
        separator = ttk.Separator(frame, orient="horizontal")
        separator.pack(fill="x", pady=10)

        main_section_frame = tk.Frame(frame, bg="#f0f4f8")
        main_section_frame.pack(fill="x", padx=10, pady=5)

        # --- ÄNDERUNG: Point Generation Button entfernt ---
        # (Hier war früher der point_frame mit dem Button)

        # Labeling & Data Creation - Rechts
        labeling_data_frame = tk.LabelFrame(main_section_frame, text="Data Creation",
                                            font=("Helvetica", 12, "bold"), bg="#ececec", fg="#333")
        # Pack-Optionen angepasst, da es jetzt alleine ist
        labeling_data_frame.pack(side="left", expand=True, fill="both", padx=5, pady=0)

        labeling_data_row = tk.Frame(labeling_data_frame, bg="#ececec")
        labeling_data_row.pack(anchor="w", padx=10, pady=5)
        self.labeling_var = tk.BooleanVar(value=False)
        self.labeling_exclImg =tk.BooleanVar(value=False)
        self.validation_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(labeling_data_row, text="Enable Labeling", variable=self.labeling_var).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(labeling_data_row, text="Enable visual overlay validation images", variable=self.validation_var).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(labeling_data_row, text="Exclude Image Data", variable=self.labeling_exclImg).pack(side="left", padx=(0, 10))
        ttk.Button(labeling_data_row, text="Create Data", command=self.parent.generateData).pack(side="left")

    def refreshPlot(self, points, airport, apex):
        self.ax.clear()
        legend_entries = []

        if apex is not None:
            self.ax.scatter(apex[0], apex[1], apex[2], color="red", s=50, label="Transformed Apex")
            legend_entries.append(f"🟢 Transformed Apex: ({apex[0]:.2f}, {apex[1]:.2f}, {apex[2]:.2f})")

        if points is not None:
            self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c="blue", alpha=0.5, label="Generated Points")
            legend_entries.append(f"🔵 Generated Points: {len(points)} points")

        if airport:
            corners = airport.calculateRunwayCorners()
            runway_points = [
                (corners["top_left"][0], corners["top_left"][1], 0),
                (corners["top_right"][0], corners["top_right"][1], 0),
                (corners["bottom_right"][0], corners["bottom_right"][1], 0),
                (corners["bottom_left"][0], corners["bottom_left"][1], 0)
            ]
            self.ax.scatter([p[0] for p in runway_points], [p[1] for p in runway_points],
                            [p[2] for p in runway_points], c="red", s=20, label="Runway Corners")
            legend_entries.append("🔴 Runway Corners")
            poly = Poly3DCollection([runway_points], color="gray", alpha=0.5, label="Runway Area")
            self.ax.add_collection3d(poly)
            legend_entries.append("⬜️ Runway Area")

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.canvas.draw()

        self.plot_description.config(state="normal", font=("Helvetica", 12))
        self.plot_description.delete("1.0", tk.END)
        self.plot_description.tag_configure("spacing", spacing3=10)
        for entry in legend_entries:
            self.plot_description.insert("end", entry + "\n", "spacing")
        self.plot_description.config(state="disabled")

        self.__plotSamplingPointDistribution()

    def __plotSamplingPointDistribution(self):
        """Aktualisiert den kleinen 2D-Plot basierend auf der Auswahl."""
        distribution = self.distribution_var.get()
        apply_y = self.apply_y.get()

        self.dist_ax.clear()

        # Standard X-Achse für schematische Darstellung
        x = np.linspace(-1, 1, 100)
        y = np.zeros_like(x)
        title = ""

        if distribution == "Normal Distribution":
            y = np.ones_like(x)
            title = "Uniform (Gleichmäßig)"

        elif distribution == "Parabel":
            y = x**2
            title = "Parabel (Mitte wenig, Ränder viel)"

        elif distribution == "Exponentiell":
            if apply_y:
                y = np.exp(-4 * x**2)
                title = "Exp. (Zentriert/Winkel)"
            else:
                x = np.linspace(0, 3, 100)
                y = np.exp(-x)
                title = "Exp. (Start viel, Ende wenig)"

        self.dist_ax.plot(x, y, color="#007acc", linewidth=2)
        self.dist_ax.fill_between(x, y, color="#007acc", alpha=0.3)
        self.dist_ax.set_title(title, fontsize=9)
        self.dist_ax.grid(True, linestyle=":", alpha=0.6)
        self.dist_ax.set_yticks([]) 
        if distribution == "Exponentiell" and not apply_y:
             self.dist_ax.set_xticks([])
        else:
             self.dist_ax.set_xticks([-1, 0, 1])
             self.dist_ax.set_xticklabels(["L", "0", "R"])

        self.dist_canvas.draw()
    
    # ... (Restliche Methoden wie saveAirport, loadAirport, loadMetadataFiles etc. bleiben unverändert) ...
    # Bitte hier die restlichen Methoden aus deiner Original-Datei beibehalten/kopieren,
    # sie wurden nicht geändert.
    def saveAirport(self):
        try:
            name = self.airport_entries["Airport Name"].get()
            icao = self.airport_entries["ICAO Code"].get()
            runway_name = self.airport_entries["Runway Name"].get()
            width = self.airport_entries["Width"].get()
            length = self.airport_entries["Length"].get()
            heading = self.airport_entries["Heading"].get()
            lat = self.airport_entries["Latitude"].get()
            lon = self.airport_entries["Longitude"].get()
            alt = self.airport_entries["Altitude"].get()
            start_height = self.airport_entries["Start Height"].get()
            end_height = self.airport_entries["End Height"].get()

            if not all([name, icao, runway_name, width, length, heading, lat, lon, alt]):
                raise ValueError("All fields must be filled out to save the airport!")

            width, length, heading = map(float, [width, length, heading])
            lat, lon, alt = map(float, [lat, lon, alt])
            start_height, end_height = map(float, [start_height, end_height])

            self.airport = RunwayGeometryCalculator(name, icao, runway_name, width, length, heading, lat, lon, alt, start_height, end_height, {})
            file = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
            if file:
                self.airport.saveAirport(file)
                messagebox.showinfo("Success", f"Airport data saved to: {file}")
        except ValueError as ve:
            messagebox.showerror("Input Error", str(ve))
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

    def loadAirport(self):
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
            # Trigger update after loading
            self._trigger_update()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load airport: {e}")

    def saveParameters(self):
        params = {key: self.point_entries[key].get() for key in self.point_entries}
        file = JSONManager.save_to_file(params)
        if file: messagebox.showinfo("Success", f"Parameters saved to: {file}")

    def loadParameters(self):
        params = JSONManager.load_from_file()
        if params: 
            self.__populateEntryFields(self.point_entries, params)
            self._trigger_update()

    def saveAngles(self):
        angles = {key: self.angle_entries[key].get() for key in self.angle_entries}
        file = JSONManager.save_to_file(angles)
        if file: messagebox.showinfo("Success", f"Angles saved to: {file}")

    def loadAngles(self):
        angles = JSONManager.load_from_file()
        if angles: 
            self.__populateEntryFields(self.angle_entries, angles)
            self._trigger_update()

    def saveTime(self):
        times = {key: self.time_entries[key].get() for key in self.time_entries}
        file = JSONManager.save_to_file(times)
        if file: messagebox.showinfo("Success", f"Times saved to: {file}")

    def loadTime(self):
        times = JSONManager.load_from_file()
        if times: self.__populateEntryFields(self.time_entries, times)

    def loadMetadataFiles(self, path=None):
        if not path:
            path = filedialog.askopenfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
            if not path: return
        try:
            metadata_reader = MetadataFileReader(path)
            metadata_reader.load_metadata()
            messagebox.showinfo("Success", f"Metadata loaded from: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load metadata: {e}")

    def generateImageFromMetadata(self, path=None):
        if not path: path = self.metadata_path_var.get()
        if not path:
            messagebox.showerror("Error", "Please select a metadata JSON file first.")
            return
        try:
            reader = MetadataFileReader(path)
            reader.load_metadata()
            out_path = reader.generate_image_from_metadata()
            messagebox.showinfo("Success", f"Image generated from metadata:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate image: {e}")

    def generateImagesFromFolder(self, folder_path):
        if not folder_path:
            folder_path = filedialog.askdirectory(title="Select folder with JSON files")
            if not folder_path: return
        try:
            reader = MetadataFileReader(file_path="", screenshot_dir=folder_path)
            out_paths = reader.process_folder(folder_path, use_sim=True)
            messagebox.showinfo("Success", f"{len(out_paths)} images generated.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {e}")
            
    def __populateEntryFields(self, entry_fields, values):
        for key, value in values.items():
            if key in entry_fields:
                entry_fields[key].delete(0, tk.END)
                entry_fields[key].insert(0, str(value))