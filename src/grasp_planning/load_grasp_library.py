from pydrake.all import RigidTransform
from pydrake.math import RotationMatrix
import sys, os, json
import numpy as np
sys.path.append("/workspaces/CheckMate-6.4210-final-project/src")

def load_grasp_library():
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, "grasp_library.json")
    
    with open(path, "r") as f:
        data = json.load(f)

    # Convert dicts → RigidTransforms
    grasp_library = {}
    for piece_type, grasp_list in data.items():
        transforms = []
        for g in grasp_list:
            R = RotationMatrix(np.array(g["rotation"]))
            p = np.array(g["translation"])
            transforms.append(RigidTransform(R, p))
        grasp_library[piece_type] = transforms

    return grasp_library