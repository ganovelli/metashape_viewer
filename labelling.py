import metashape_loader as ml
import json
import glm

class Label:
    def __init__(self, name, color):
        self.name = name
        self.color = color
        self.sample_points_ref = [] # reference to 


class SamplePoint:
    def __init__(self, position):
        self.position = position
        self.camera_refs = []
        self.projected_coords = []

        self.label = None  # Instance of Label or None



def load_labels(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

        labels_array = data.get("Labels", [])

        for label in labels_array:
            name = label.get("name")
            fill = label.get("fill")
            labels.append(Label(name,fill))

def save_labelling(metashape_path, labels_path,output_path):
    global sample_points
    data = {
        "metashape_path": metashape_path,
        "labels_path"   : labels_path,
        "sample_points": []
    }

    for sp in sample_points:
        entry = {
            "position": [float(sp.position.x),
                         float(sp.position.y),
                         float(sp.position.z)],

            "camera_refs": [
                [int(a), int(b)] for a, b in sp.camera_refs
            ],

            "projected_coords": [
                [int(x), int(y)] for x, y in sp.projected_coords
            ],

            "label": int(sp.label) if sp.label is not None else -1
        }

        data["sample_points"].append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_labelling(input_path):
    """
    Loads labelling data from JSON and reconstructs SamplePoint objects.

    Returns:
        metashape_path (str)
        labels_path (str)
        sample_points (list[SamplePoint])
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metashape_path = data.get("metashape_path")
    labels_path = data.get("labels_path")


    for entry in data.get("sample_points", []):
        # Position
        pos = entry["position"]
        sp = SamplePoint(glm.vec3(pos[0], pos[1], pos[2]))

        # Camera refs
        sp.camera_refs = [
            (int(a), int(b)) for a, b in entry.get("camera_refs", [])
        ]

        # Projected coords
        sp.projected_coords = [
            (int(x), int(y)) for x, y in entry.get("projected_coords", [])
        ]

        # Label
        label = entry.get("label", -1)
        sp.label = None if label == -1 else int(label)

        sample_points.append(sp)

    return metashape_path, labels_path, sample_points


global sample_points
global labels

sample_points = []
labels = []




