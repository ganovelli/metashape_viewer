import metashape_loader as ml
import json
import glm

class Label:
    def __init__(self, name, color, group):
        self.color = color
        self.name = name
        self.group = group
        self.clicks = 0 # number of times this label has been assigned/removed
        self.sample_points_ref = [] # reference to 


class SamplePoint:
    def __init__(self, position,normal):
        self.position = position
        self.normal = normal
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
            group = label.get("group", "ungrouped")
            labels.append(Label(name,fill,group))

def save_labelling(metashape_path, images_path,labels_path,output_path):
    global labels
    global sample_points
    global sampling_radius
    data = {
        "metashape_path": metashape_path,
        "images_path": images_path,
        "labels_path"   : labels_path,
        "sampling_radius": sampling_radius,
        "sample_points": [],
        "label_occurrences": [label.clicks for label in labels]
        }
    for sp in sample_points:
        entry = {
            "position": [float(sp.position.x),
                         float(sp.position.y),
                         float(sp.position.z)],
            "normals": [float(sp.normal.x),
                         float(sp.normal.y),
                         float(sp.normal.z)],

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
    global sampling_radius

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metashape_path = data.get("metashape_path")
    images_path = data.get("images_path")
    labels_path = data.get("labels_path")
    sampling_radius = data.get("sampling_radius", 0.01)


    for entry in data.get("sample_points", []):
        # Position
        pos = entry["position"]
        # Normal
        nor = entry["normals"]

        sp = SamplePoint(glm.vec3(pos[0], pos[1], pos[2]), glm.vec3(nor[0], nor[1], nor[2]))

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

    labels_occurrences = data.get("label_occurrences", [])


    return metashape_path, images_path,labels_path, sample_points, labels_occurrences


global sample_points
global sampling_radius
global labels

sample_points = []
labels = []




