"""
ScenAIro - Main Entry Point
===========================

This module serves as the entry point for the ScenAIro application.
ScenAIro is a tool for generating synthetic training data for computer vision
models in the context of Microsoft Flight Simulator (MSFS).

The application provides a GUI for configuring runway parameters, generating
sampling points in 3D space, and creating labeled image datasets for ML training.

Usage:
    Run this script directly: python main.py
    
The application will launch a tkinter-based GUI where users can:
    - Configure airport and runway parameters
    - Define trajectory/cone parameters for point generation
    - Set environmental conditions (time, weather)
    - Generate and export labeled datasets with screenshots

Author: ScenAIro Team
"""

from ScenAIro import ScenAIro

if __name__ == "__main__":
    # Create the main application window (ScenAIro class inherits from tk.Tk)
    app = ScenAIro()
    
    # Start the tkinter main event loop
    app.mainloop()
