import pygetwindow as gw

class SimFrameScout:

    def __init__(self, window_title="Microsoft Flight Simulator"):
        self.window_title = window_title
        self.window = None

    def find_window(self):
        # find window by title
        windows = gw.getWindowsWithTitle(self.window_title)
        if not windows:
            raise Exception(f"No Window of the '{self.window_title}' found.")
        self.window = windows[0]
        return self.window

    def get_aspect_ratio(self):
        # get window size
        if self.window is None:
            self.find_window()

        width = self.window.width
        height = self.window.height

        if height == 0:
            raise ValueError("The window height is 0, the aspect ratio cannot be calculated.")

        aspect_ratio = width / height
        return round(aspect_ratio, 2)  # aspect ratio rounded to 2 decimal places
    


    