import metashape_loader as ml

class Label:
    def __init__(self, name, color):
        self.name = name
        self.signature = color
        self.sample_points_ref = [] # reference to 


class SamplePoint:
    def __init__(self, position):
        self.position = position
        self.camera_refs = []
        self.projected_coords = []

        self.label = None  # Instance of Label or None

class LabelSet:
    def __init__(self):
        self.labels = {}
        self.sample_points = []

    def add_label(self, name, color):
        if name not in self.labels:
            self.labels[name] = Label(name, color)


global sample_points

sample_points = []




