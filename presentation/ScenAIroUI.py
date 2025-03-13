import tkinter as tk
from tkinter import ttk, messagebox, PhotoImage, filedialog
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .tools import PointCloudGenerator, ConeTransformer, CoordSetter
from .tools.RunwayCalc import RunwayCalc

class JSONManager:
    @staticmethod
    def save_to_file(data, filetypes=("JSON files", "*.json")):
        """Saves data to a JSON file."""
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
        """Loads data from a JSON file."""
        try:
            file = filedialog.askopenfilename(defaultextension=".json", filetypes=[filetypes])
            if file:
                with open(file, "r") as f:
                    return json.load(f)
            return None
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
            return None


class ScenAIroUI(tk.Tk):

    def __init__(self, parent):
        self.airport = None
        self.parent = parent
        self.jsonmanager = JSONManager()

        # Lade das Icon


        # Haupt-Frames: Links (1/3) und Rechts (2/3)
        main_frame = tk.Frame(self.parent, bg="#f0f4f8")
        main_frame.pack(fill="both", expand=True)

        self.left_frame = tk.Frame(main_frame, bg="#f0f4f8", width=400)
        self.left_frame.pack(side="left", fill="y", padx=10, pady=10)

        self.right_frame = tk.Frame(main_frame, bg="#f0f4f8")
        self.right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Eingabefelder im linken Bereich
        self.airport_entries = self.create_input_section(
            "Airport Parameters",
            ["Airport Name", "ICAO Code", "Runway Name", "Width", "Length", "Heading", "Latitude", "Longitude",
             "Altitude", "Start Height", "End Height"],
            "#e6f7ff",
            parent=self.left_frame,
            load_command=self.load_airport,
            save_command=self.save_airport
        )

        self.point_entries = self.create_input_section(
            "Point Generation Parameters",
            ["Apex X", "Apex Y", "Apex Z", "Lateral Angle Left", "Lateral Angle Right",
             "Vertical Min Angle", "Vertical Max Angle", "Maximum Distance", "Number of Points"],
            "#ffede6",
            parent=self.left_frame,
            load_command=self.load_parameters,
            save_command=self.save_parameters
        )

        self.angle_entries = self.create_input_section(
            "Angle Parameters (Pitch, Yaw, Bank)",
            ["Pitch Min", "Pitch Max", "Yaw Min", "Yaw Max", "Bank Min", "Bank Max"],
            "#e6ffe6",
            parent=self.left_frame,
            load_command=self.load_angles,
            save_command=self.save_angles
        )

        ## Dummy-Verteilungs-Frame hinzufügen
        self.dummy_frame = tk.LabelFrame(self.left_frame, text="Distribution Settings",
                                         font=("Helvetica", 12, "bold"), bg="#f3e6ff", fg="#333")
        self.dummy_frame.pack(fill="x", padx=5, pady=10)

        # Unterteilung in zwei Zeilen
        row1 = tk.Frame(self.dummy_frame, bg="#f3e6ff")
        row1.pack(fill="x", padx=5, pady=2)
        row2 = tk.Frame(self.dummy_frame, bg="#f3e6ff")
        row2.pack(fill="x", padx=5, pady=2)

        # Zeile 1: Dropdown-Menü
        tk.Label(row1, text="Distribution:", bg="#f3e6ff").pack(side="left", padx=5)
        self.distribution_var = tk.StringVar(value="Normal Distribution")
        distribution_menu = ttk.Combobox(row1, textvariable=self.distribution_var,
                                         values=["Normal Distribution", "Parabel", "Exponentiell"],
                                         state="readonly")
        distribution_menu.pack(side="left", padx=5)

        # Zeile 2: Checkboxen zur Achsenauswahl und Button
        tk.Label(row2, text="Apply to:", bg="#f3e6ff").pack(side="left", padx=5)
        self.apply_x = tk.BooleanVar(value=True)
        self.apply_y = tk.BooleanVar(value=True)
        tk.Checkbutton(row2, text="X-Axis", variable=self.apply_x, bg="#f3e6ff").pack(side="left", padx=5)
        tk.Checkbutton(row2, text="Y-Axis", variable=self.apply_y, bg="#f3e6ff").pack(side="left", padx=5)

        update_button = ttk.Button(row2, text="Update Distribution", command=self.plot_dummy_distribution)
        update_button.pack(side="right", padx=5)

        ## Segmentierungs-Frame hinzufügen
        self.segmentation_frame = tk.LabelFrame(self.left_frame, text="Point Cloud Segmentation",
                                                font=("Helvetica", 12, "bold"), bg="#F5E0D3", fg="#333")
        self.segmentation_frame.pack(fill="x", padx=5, pady=10)

        # Zeile 1: Dropdown-Menü zur Auswahl der Methode
        row1 = tk.Frame(self.segmentation_frame, bg="#F5E0D3")
        row1.pack(fill="x", padx=5, pady=2)

        tk.Label(row1, text="Method:", bg="#F5E0D3").pack(side="left", padx=5)
        self.segmentation_method = tk.StringVar(value="K-Means")
        segmentation_menu = ttk.Combobox(row1, textvariable=self.segmentation_method,
                                         values=["K-Means", "DBSCAN", "Threshold"], state="readonly")
        segmentation_menu.pack(side="left", padx=5)

        # Zeile 2: Checkboxen zur Achsenauswahl und Button
        row2 = tk.Frame(self.segmentation_frame, bg="#F5E0D3")
        row2.pack(fill="x", padx=5, pady=2)

        tk.Label(row2, text="Apply to:", bg="#F5E0D3").pack(side="left", padx=5)
        self.segment_x = tk.BooleanVar(value=True)
        self.segment_y = tk.BooleanVar(value=True)
        tk.Checkbutton(row2, text="X-Axis", variable=self.segment_x, bg="#F5E0D3").pack(side="left", padx=5)
        tk.Checkbutton(row2, text="Y-Axis", variable=self.segment_y, bg="#F5E0D3").pack(side="left", padx=5)

        segmentation_button = ttk.Button(row2, text="Start Segmentation", command=self.start_segmentation)
        segmentation_button.pack(side="right", padx=5)

        ## Rechter Haupt-Frame: Oberer und unterer Bereich
        self.right_frame_top = tk.Frame(self.right_frame, bg="#f0f4f8")
        self.right_frame_top.pack(side="top", fill="both", expand=True, padx=5, pady=5)

        self.right_frame_bottom = tk.Frame(self.right_frame, bg="#f0f4f8")
        self.right_frame_bottom.pack(side="bottom", fill="x", padx=5, pady=5)

        ## Plot-Frame erstellen
        plot_frame = tk.Frame(self.right_frame_top, bg="#f0f4f8")
        plot_frame.pack(side="left", fill="both", expand=True, padx=10, pady=(10, 0))

        # Erstelle den Plot
        self.fig = plt.figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Zusatzbereich für Beschreibung und Distribution
        right_plot_frame = tk.Frame(self.right_frame_top, bg="#f0f4f8", width=250)  # Breiterer Bereich
        right_plot_frame.pack(side="right", fill="both", expand=False, padx=5, pady=5)
        right_plot_frame.pack_propagate(False)  # Verhindert Größenanpassung an Inhalte

        # Feld zur Anzeige der Plot-Beschreibung
        description_frame = tk.LabelFrame(right_plot_frame, text="Plot Legend", font=("Helvetica", 12, "bold"),
                                          bg="#e6f7ff", fg="#333", height=40)
        description_frame.pack(fill="both", expand=True, padx=5, pady=5)
        description_frame.pack_propagate(False)

        self.plot_description = tk.Text(description_frame, wrap="word", bg="#ffffff", fg="#333",
                                        font=("Helvetica", 10))
        self.plot_description.pack(fill="both", expand=True, padx=5, pady=5)
        self.plot_description.insert("1.0", "Plot Information is going to be displayed here")

        # Feld zur Darstellung der 2D-Distribution
        distribution_frame = tk.LabelFrame(right_plot_frame, text="2D Distribution", font=("Helvetica", 12, "bold"),
                                           bg="#ffede6", fg="#333", height=60)
        distribution_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.dist_fig, self.dist_ax = plt.subplots(figsize=(4, 2))
        self.dist_canvas = FigureCanvasTkAgg(self.dist_fig, master=distribution_frame)
        self.dist_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Buttons unter dem Plot (sicherstellen, dass sie sichtbar sind)
        self.create_buttons(self.right_frame_bottom)

        # Höhe des Plots anpassen
        self.adjust_plot_height()

    def adjust_plot_height(self):
        """Passt die Höhe des Plots und der neuen Felder rechts an."""
        left_box_height = 650  # Höhe der linken Boxen

        reduced_plot_height = 350  # Kleinere Höhe für den Plot
        # self.canvas.get_tk_widget().config(height=reduced_plot_height)
        self.plot_description.config(height=10)  # Fixierte Höhe für die Textbeschreibung
        # self.dist_canvas.get_tk_widget().config(height=reduced_plot_height // 2)  # Distribution auf halber Höhe

    def create_input_section(self, title, fields, bg_color, parent, save_command=None, load_command=None):
        section_frame = tk.LabelFrame(parent, text=title, font=("Helvetica", 12, "bold"), bg=bg_color, fg="#333")
        section_frame.pack(fill="x", padx=5, pady=5)
        section_frame.pack_propagate(False)  # Verhindert Größenanpassung
        section_frame.config(width=160)  # Breite von 180 px

        section_entries = {}
        total_fields = len(fields)

        for idx, field in enumerate(fields):
            row = tk.Frame(section_frame, bg=bg_color)
            row.grid(row=idx // 2, column=idx % 2, padx=2, pady=2, sticky="w")

            tk.Label(row, text=field, bg=bg_color, anchor="w", width=10).pack(side="left")
            entry = ttk.Entry(row, width=8)  # Schmalere Eingabefelder
            entry.pack(side="right", fill="x")
            section_entries[field] = entry

        # Buttons für Save und Load unten platzieren
        if save_command and load_command:
            button_frame = tk.Frame(section_frame, bg=bg_color)
            button_frame.grid(row=(total_fields // 2) + 1, columnspan=1, pady=5)

            ttk.Button(button_frame, text="Save", command=save_command).pack(side="left", padx=5)
            ttk.Button(button_frame, text="Load", command=load_command).pack(side="right", padx=5)

        return section_entries

    def create_buttons(self, frame):
        # Gruppe 1: Save/Load Options
        """
        save_load_frame = tk.LabelFrame(frame, text="Save/Load Options", font=("Helvetica", 12, "bold"), bg="#f5f5f5",
                                        fg="#333")
        save_load_frame.pack(fill="x", padx=10, pady=5)
        buttons = [
            ("Save Airport", self.save_airport),
            ("Load Airport", self.load_airport),
            ("Save Parameters", self.save_parameters),
            ("Load Parameters", self.load_parameters),
            ("Save Angles", self.save_angles),
            ("Load Angles", self.load_angles),
        ]
        for text, command in buttons:
            ttk.Button(save_load_frame, text=text, command=command).pack(side="left", padx=5, pady=2)
        """

        # Trennlinie horizontal
        separator = ttk.Separator(frame, orient="horizontal")
        separator.pack(fill="x", pady=10)

        # Haupt-Frame für Point Generation und Labeling & Data Creation in einer Zeile
        main_section_frame = tk.Frame(frame, bg="#f0f4f8")
        main_section_frame.pack(fill="x", padx=10, pady=5)

        # Point Generation - Links
        point_frame = tk.LabelFrame(main_section_frame, text="Point Generation", font=("Helvetica", 12, "bold"),
                                    bg="#f9f3e6", fg="#333")
        point_frame.pack(side="left", expand=True, fill="both", padx=(0, 5), pady=0)
        ttk.Button(point_frame, text="Generate Points", command=self.parent.generate_and_transform_points).pack(anchor="w",
                                                                                                         padx=10,
                                                                                                         pady=5)

        # Senkrechte Trennlinie
        separator_vertical = ttk.Separator(main_section_frame, orient="vertical")
        separator_vertical.pack(side="left", fill="y", padx=5)

        # Labeling & Data Creation - Rechts
        labeling_data_frame = tk.LabelFrame(main_section_frame, text="Labeling & Data Creation",
                                            font=("Helvetica", 12, "bold"), bg="#ececec", fg="#333")
        labeling_data_frame.pack(side="left", expand=True, fill="both", padx=(5, 0), pady=0)

        # Labeling Checkbox und Create Data Button in einer Zeile
        labeling_data_row = tk.Frame(labeling_data_frame, bg="#ececec")
        labeling_data_row.pack(anchor="w", padx=10, pady=5)
        self.labeling_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(labeling_data_row, text="Enable Labeling", variable=self.labeling_var).pack(side="left",
                                                                                                    padx=(0, 10))
        ttk.Button(labeling_data_row, text="Create Data", command=self.parent.create_data).pack(side="left")

    def update_plot(self, points, airport, apex):
        """Updates the 3D plot and displays the legend in the description text field."""
        self.ax.clear()  # Lösche den aktuellen Plot
        legend_entries = []  # List to hold legend descriptions

        # Plot the transformed Apex
        if apex is not None:
            self.ax.scatter(apex[0], apex[1], apex[2], color="red", s=50, label="Transformed Apex")
            legend_entries.append(f"🟢 Transformed Apex: Position ({apex[0]:.2f}, {apex[1]:.2f}, {apex[2]:.2f})")

        # Plot the generated points
        if points is not None:
            self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c="blue", alpha=0.5,
                            label="Generated Points")
            legend_entries.append(f"🔵 Generated Points: {len(points)} points")

        # Plot the runway corners and area
        if airport:
            corners = airport.calculate_runway_corners()
            runway_points = [
                (corners["top_left"][0], corners["top_left"][1], 0),
                (corners["top_right"][0], corners["top_right"][1], 0),
                (corners["bottom_right"][0], corners["bottom_right"][1], 0),
                (corners["bottom_left"][0], corners["bottom_left"][1], 0)
            ]

            # Plot runway corners
            self.ax.scatter([p[0] for p in runway_points], [p[1] for p in runway_points],
                            [p[2] for p in runway_points], c="red", s=20, label="Runway Corners")
            legend_entries.append("🔴 Runway Corners")

            # Plot runway area
            poly = Poly3DCollection([runway_points], color="gray", alpha=0.5, label="Runway Area")
            self.ax.add_collection3d(poly)
            legend_entries.append("⬜️ Runway Area")

        # Update legend
        self.plot_description.config(state="normal")
        self.plot_description.delete("1.0", tk.END)
        for entry in legend_entries:
            self.plot_description.insert("end", entry + "\n")
        self.plot_description.config(state="disabled")

        # Update the plot
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.canvas.draw()

        # Update the legend in the Plot Description frame
        self.plot_description.config(state="normal", font=("Helvetica", 12))  # Larger font
        self.plot_description.delete("1.0", tk.END)  # Clear existing text
        self.plot_description.tag_configure("spacing", spacing3=10)  # Add more spacing between lines
        for entry in legend_entries:
            self.plot_description.insert("end", entry + "\n", "spacing")  # Apply spacing tag
        self.plot_description.config(state="disabled")

        self.plot_dummy_distribution()

    def plot_cone_boundaries(self):
        """
        Plottet die äußeren Grenzen des Kegels entlang der Heading-Richtung.
        """
        if not hasattr(self, 'apex') or self.apex is None:
            messagebox.showerror("Error", "Apex is not defined. Please generate points first.")
            return

        # Parameter aus den Eingabefeldern auslesen
        lateral_left_rad = np.radians(float(self.point_entries["Lateral Angle Left"].get()))
        lateral_right_rad = np.radians(float(self.point_entries["Lateral Angle Right"].get()))
        vertical_min_rad = np.radians(float(self.point_entries["Vertical Min Angle"].get()))
        vertical_max_rad = np.radians(float(self.point_entries["Vertical Max Angle"].get()))
        max_distance = float(self.point_entries["Maximum Distance"].get())
        heading_rad = np.radians(self.airport.runway_heading)

        # Apex-Position
        apex = np.array(PointCloudGenerator.transform_apex(self.apex, np.degrees(heading_rad)))

        # Liste der Kegelgrenzenpunkte berechnen
        boundary_points = []
        for lateral_angle in [lateral_left_rad, lateral_right_rad]:
            for vertical_angle in [vertical_min_rad, vertical_max_rad]:
                x = max_distance
                y = x * np.tan(lateral_angle)
                z = x * np.tan(vertical_angle)

                # Punkte rotieren um die Z-Achse gemäß Heading
                x_rot = x * np.cos(heading_rad) - y * np.sin(heading_rad)
                y_rot = x * np.sin(heading_rad) + y * np.cos(heading_rad)

                # Punkte um den Apex verschieben
                boundary_points.append([x_rot + apex[0], y_rot + apex[1], z + apex[2]])

        boundary_points = np.array(boundary_points)

        # Plot der Kegelgrenzen (Apex zu den Grenzen)
        for boundary in boundary_points:
            self.ax.plot([apex[0], boundary[0]],
                         [apex[1], boundary[1]],
                         [apex[2], boundary[2]],
                         color='green', linewidth=1.5)

        # Plot der Basis des Kegels
        self.ax.plot([boundary_points[0][0], boundary_points[1][0]],
                     [boundary_points[0][1], boundary_points[1][1]],
                     [boundary_points[0][2], boundary_points[1][2]], color='orange', linestyle="--")

        self.ax.plot([boundary_points[2][0], boundary_points[3][0]],
                     [boundary_points[2][1], boundary_points[3][1]],
                     [boundary_points[2][2], boundary_points[3][2]], color='orange', linestyle="--")

        # Apex plotten
        self.ax.scatter(apex[0], apex[1], apex[2], color='green', s=50, label="Apex")

    def plot_dummy_distribution(self):
        """Creates and updates a bar chart based on the selected distribution and axis."""
        distribution = self.distribution_var.get()  # Ausgewählte Verteilung
        apply_x = self.apply_x.get()
        apply_y = self.apply_y.get()

        # Daten für die Balkendiagramme
        x = np.linspace(-5, 5, 20)

        # Verteilung berechnen
        if distribution == "Normal Distribution":
            y = np.exp(-x ** 2)  # Gaußsche Glockenkurve
        elif distribution == "Parabel":
            y = x ** 2  # Parabel: U-Form
        elif distribution == "Exponentiell":
            y = np.exp(x) if apply_x else np.exp(-x)  # Exponentiell steigend oder fallend

        # Achsenanpassung durch Checkboxen
        if not apply_x:
            x = np.ones_like(y)  # X bleibt konstant
        if not apply_y:
            y = np.ones_like(x)  # Y bleibt konstant

        # Plot aktualisieren
        self.dist_ax.clear()
        self.dist_ax.bar(x, y, width=0.5, color="skyblue", edgecolor="black")
        self.dist_ax.grid(True, linestyle="--", alpha=0.7)

        # Canvas aktualisieren
        self.dist_canvas.draw()

    def start_segmentation(self):
        """Dummy function to simulate point cloud segmentation."""
        method = self.segmentation_method.get()
        segment_x = self.segment_x.get()
        segment_y = self.segment_y.get()

        # Zeige eine Dummy-Meldung
        messagebox.showinfo("Segmentation",
                            f"Segmentation started using '{method}' method.\n"
                            f"Segment X: {'Yes' if segment_x else 'No'}\n"
                            f"Segment Y: {'Yes' if segment_y else 'No'}")

    def fill_fields(self, entry_fields, values):
        """
        # Populates the provided entry fields with corresponding values.
        #
        # Args:
        #    entry_fields (dict): Dictionary mapping field names to their entry widgets.
        #    values (dict): Dictionary containing values to populate into the fields.
        """
        for key, value in values.items():
            if key in entry_fields:
                entry_fields[key].delete(0, tk.END)  # Clear the field
                entry_fields[key].insert(0, str(value))  # Populate new value

    def save_airport(self):
        
        try:
            # Werte aus den Eingabefeldern holen
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

            # Überprüfen auf leere Felder
            if not all([name, icao, runway_name, width, length, heading, lat, lon, alt]):
                raise ValueError("All fields must be filled out to save the airport!")

            # Konvertiere numerische Werte
            width = float(width)
            length = float(length)
            heading = float(heading)
            lat = float(lat)
            lon = float(lon)
            alt = float(alt)
            start_height = float(start_height)
            end_height = float(end_height)

            # Flughafen-Objekt erstellen
            self.airport = RunwayCalc(name, icao, runway_name, width, length, heading, lat, lon, alt, start_height, end_height, {})

            # Dateispeicher-Dialog
            file = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
            if file:
                self.airport.save_to_file(file)
                messagebox.showinfo("Success", f"Airport data saved to: {file}")
        except ValueError as ve:
            messagebox.showerror("Input Error", str(ve))
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

    def load_airport(self):
        try:
            file = filedialog.askopenfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
            if not file:
                return

            # Flughafen-Objekt direkt laden
            self.airport = RunwayCalc.load_from_file(file)

            # Eingabefelder aktualisieren
            self.fill_fields(self.airport_entries, {
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

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load airport: {e}")

    def save_parameters(self):
        params = {key: self.point_entries[key].get() for key in self.point_entries}
        file = JSONManager.save_to_file(params)
        if file:
            messagebox.showinfo("Success", f"Parameters saved to: {file}")

    def load_parameters(self):
        params = JSONManager.load_from_file()
        if params:
            self.fill_fields(self.point_entries, params)

    def save_angles(self):
        angles = {key: self.angle_entries[key].get() for key in self.angle_entries}
        file = JSONManager.save_to_file(angles)
        if file:
            messagebox.showinfo("Success", f"Angles saved to: {file}")

    def load_angles(self):
        angles = JSONManager.load_from_file()
        if angles:
            self.fill_fields(self.angle_entries, angles)
