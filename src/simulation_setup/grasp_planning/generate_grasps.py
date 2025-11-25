import json
import os
from pydrake.all import RigidTransform
from mesh_grasp_sampling import generate_mesh_grasps

# ---------------------------------------------------------
# Path Setup
# ---------------------------------------------------------

HERE = os.path.dirname(__file__)  # .../simulation_setup/grasp_planning

# HERE = "/OneDrive/desktop/robotic manipulation/CheckMate-6.4210-final-project/src/simulation_setup/grasp_planning"
print(HERE)
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))  # .../src
MESH_ROOT = os.path.join(REPO_ROOT, "models", "pieces")

OUTPUT_JSON = os.path.join(HERE, "grasp_library.json")


# pieces to process
PIECE_TYPES = ["pawn", "rook", "knight", "bishop", "queen", "king"]


def get_mesh_path(piece_type: str) -> str:
    """Return full filesystem path for a piece mesh."""
    return os.path.join(
        MESH_ROOT,
        piece_type + "s",
        f"{piece_type}_mesh.obj"
    )


def transform_to_dict(T: RigidTransform):
    """Convert a RigidTransform to JSON-serializable dict."""
    return {
        "rotation": T.rotation().matrix().tolist(),
        "translation": T.translation().tolist()
    }


def main():
    grasp_db = {}

    for piece_name in PIECE_TYPES:
        mesh_path = get_mesh_path(piece_name)
        print(f"Generating grasps for {piece_name}: {mesh_path}")

        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Missing mesh file: {mesh_path}")

        grasps = generate_mesh_grasps(mesh_path, n_candidates=8)

        grasp_db[piece_name] = [transform_to_dict(g) for g in grasps]

    # Write to JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(grasp_db, f, indent=4)

    print(f"Grasp library saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
