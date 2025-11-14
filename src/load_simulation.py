# chess_setup.py
import yaml
import os
from pydrake.all import RigidTransform, RollPitchYaw

def load_chessboard_from_yaml(plant, parser, yaml_file):
    """Load chessboard with pieces from a YAML configuration"""
    
    # Load the YAML with piece positions
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get the absolute path to your models directory
    models_dir = "/workspaces/CheckMate-6.4210-final-project/src/models"
    
    # Add chessboard - use absolute path
    board_path = os.path.join(models_dir, "chessboard.sdf")
    board_url = f"file://{board_path}"
    board_models = parser.AddModelsFromUrl(board_url)
    board_model = board_models[0]
    plant.WeldFrames(plant.world_frame(), 
                    plant.GetFrameByName("board_base", board_model),
                    RigidTransform([0, 0, 0]))
    
    # ADD IIWA ROBOT AND GRIPPER USING DRAKE_MODELS PACKAGE
    print("Loading IIWA robot and gripper from drake_models...")
    
    try:
        # Load IIWA arm from drake_models package
        iiwa_url = "package://drake_models/iiwa_description/sdf/iiwa14_no_collision.sdf"
        iiwa_models = parser.AddModelsFromUrl(iiwa_url)
        
        if iiwa_models:
            iiwa = iiwa_models[0]
            print("Successfully loaded IIWA robot from drake_models")
            
            # Weld robot base to world (behind white pieces)
            # Position: [0, -0.5, 0] with 180° rotation around Z
            X_iiwa = RigidTransform(
                RollPitchYaw(0, 0, 3.14159),  # 180 degrees in radians
                [0, -.8, 0]
            )
            plant.WeldFrames(
                plant.world_frame(),
                plant.GetFrameByName("iiwa_link_0", iiwa),
                X_iiwa
            )
            
            # Try to load WSG gripper from drake_models
            try:
                wsg_url = "package://manipulation/hydro/schunk_wsg_50_with_tip.sdf"
                wsg_models = parser.AddModelsFromUrl(wsg_url)
                
                if wsg_models:
                    wsg = wsg_models[0]
                    
                    # Weld gripper to end effector
                    X_gripper = RigidTransform(
                        RollPitchYaw(1.5708, 0, 1.5708),  # 90, 0, 90 degrees in radians
                        [0, 0, 0.09]
                    )
                    plant.WeldFrames(
                        plant.GetFrameByName("iiwa_link_7", iiwa),
                        plant.GetFrameByName("body", wsg),
                        X_gripper
                    )
                    
                    print("Successfully loaded WSG gripper")
                else:
                    print("WSG gripper not found in drake_models")
                    
            except Exception as e:
                print(f"Note: Could not load WSG gripper: {e}")
                
        else:
            print("IIWA robot not found in drake_models")
            
    except Exception as e:
        print(f"Error loading robot from drake_models: {e}")
        print("Falling back to custom robot...")
        
        # Fallback: Create a simple placeholder robot
        try:
            robot_sdf = """
            <?xml version="1.0"?>
            <sdf version="1.9">
              <model name="simple_robot">
                <link name="base">
                  <visual name="base_visual">
                    <geometry>
                      <cylinder>
                        <radius>0.05</radius>
                        <length>0.1</length>
                      </cylinder>
                    </geometry>
                    <material>
                      <diffuse>0.3 0.3 0.3 1.0</diffuse>
                    </material>
                  </visual>
                </link>
                <link name="arm1">
                  <visual name="arm1_visual">
                    <geometry>
                      <box>
                        <size>0.05 0.05 0.3</size>
                      </box>
                    </geometry>
                    <material>
                      <diffuse>0.8 0.2 0.2 1.0</diffuse>
                    </material>
                  </visual>
                  <pose>0 0 0.15 0 0 0</pose>
                </link>
                <link name="arm2">
                  <visual name="arm2_visual">
                    <geometry>
                      <box>
                        <size>0.04 0.04 0.2</size>
                      </box>
                    </geometry>
                    <material>
                      <diffuse>0.2 0.8 0.2 1.0</diffuse>
                    </material>
                  </visual>
                  <pose>0 0 0.1 0 0 0</pose>
                </link>
              </model>
            </sdf>
            """
            robot_models = parser.AddModelsFromString(robot_sdf, "simple_robot.sdf")
            if robot_models:
                plant.WeldFrames(
                    plant.world_frame(),
                    plant.GetFrameByName("base", robot_models[0]),
                    RigidTransform([0, -0.5, 0])
                )
                print("Created simple placeholder robot")
                
        except Exception as e2:
            print(f"Could not create placeholder robot: {e2}")
    
    piece_height = 0.05
    
    # Add pieces based on configuration
    for color, pieces in config['piece_positions'].items():
        for piece_type, positions in pieces.items():
            for i, pos in enumerate(positions):
                piece_name = f"{color}_{piece_type}_{i+1}"
                
                # Create a simple box SDF string for the piece
                piece_sdf = f"""
                <?xml version="1.0"?>
                <sdf version="1.9">
                  <model name="{piece_name}">
                    <link name="piece_base">
                      <visual name="piece_visual">
                        <geometry>
                          <cylinder>
                            <radius>0.02</radius>
                            <length>0.08</length>
                          </cylinder>
                        </geometry>
                        <material>
                          <diffuse>{'1 1 1' if color == 'white' else '0 0 0'} 1.0</diffuse>
                        </material>
                      </visual>
                      <collision name="piece_collision">
                        <geometry>
                          <cylinder>
                            <radius>0.02</radius>
                            <length>0.08</length>
                          </cylinder>
                        </geometry>
                      </collision>
                    </link>
                  </model>
                </sdf>
                """
                
                # Add the piece from string
                piece_models = parser.AddModelsFromString(piece_sdf, f"{piece_name}.sdf")
                
                # Weld piece to board
                plant.WeldFrames(
                    plant.world_frame(),
                    plant.GetFrameByName("piece_base", piece_models[0]),
                    RigidTransform([pos[0], pos[1], piece_height])
                )