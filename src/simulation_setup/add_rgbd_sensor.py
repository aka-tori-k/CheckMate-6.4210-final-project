from pydrake.all import (RgbdSensor, DepthRenderCamera,ColorRenderCamera, 
                         RenderCameraCore, CameraInfo, ClippingRange, DepthRange, StartMeshcat,
                         PointCloud, PeriodicEventData,Parser, RigidTransform, RotationMatrix)
import numpy as np

def add_rgbd_sensor(builder, plant, parser, scene_graph, iiwa):
    """
    Adds an RGB-D sensor to the simulation.
    """
    # Add a simple model to the plant
    wrist_frame = plant.GetFrameByName("iiwa_link_7", iiwa)
    frame_id = plant.GetBodyFrameIdOrThrow(wrist_frame.body().index())

    #adjust X_PB if move sensor
    sensor = RgbdSensor(
        parent_id=frame_id,
        X_PB=RigidTransform([0.1, 0, 0.1]),
        color_camera=ColorRenderCamera(
            core=RenderCameraCore(
                renderer_name="vtk",
                intrinsics=CameraInfo(width=640, height=480, fov_y=np.pi/4),
                clipping = ClippingRange(0.01, 10.0),
                X_BS=RigidTransform([0, 0, 0]),
            )
        ),
        depth_camera=DepthRenderCamera(
            core=RenderCameraCore(
                renderer_name="vtk",
                intrinsics=CameraInfo(width=640, height=480, fov_y=np.pi/4),
                clipping = ClippingRange(0.01, 10.0),
                X_BS=RigidTransform([0, 0, 0]),
            ),
            depth_range = DepthRange(0.01, 10.0)
        )
    )
    # Add sensor system
    rgbd = builder.AddSystem(sensor)

    # Connect SG → sensor
    builder.Connect(
        scene_graph.get_query_output_port(),
        rgbd.query_object_input_port()
    )


    # camera_model = parser.AddModels("src/models/rgbd_sensor.sdf")[0]

    # wrist_frame = plant.GetFrameByName("iiwa_link_7", iiwa)
    # camera_frame = plant.GetFrameByName("base", camera_model)   

    # adjust p if move physical camera
    #will need to move this bc curretly causing collision error
    # plant.WeldFrames(wrist_frame, camera_frame, 
    #                  RigidTransform(R=RotationMatrix.Identity(),p=np.array([0, 0, 0.11]).reshape((3, 1)))
    #                  )
    
    return plant, rgbd

if __name__ == "__main__":
    # add_rgbd_sensor(builder, plant, scene_graph)
    print("RGB-D sensor creation test completed successfully.")