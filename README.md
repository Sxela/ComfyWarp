# ComfyWarp
[WarpFusion](https://github.com/Sxela/WarpFusion) Custom Nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

# Installation
- Create a folder for ComfyWarp. Avoid whitespaces and non-latin alphanumeric characters. 
- Download install & run bat files and put them into your ComfyWarp folder
- Run install.bat. If the installation is successful, the server will be launched.
- Download and unzip the attached archive into your ComfyWarp/ComfyUI/custom_nodes folder 
- Close the server window and run run.bat to fetch the newly installed warp nodes :D
  
# Sample workflows
Sample workflows are located in ComfyWarp\sample_workflow\

# Video with the workflow:
- v0.1.0 : [WarpFusion: Warp and Consistency explanation in ComfyUI - YouTube](https://www.youtube.com/watch?v=ZuPBDRjwtu0&t=20s&ab_channel=S_X)
- v0.2.1 : [WarpFusion: ComfyWarp iteration 2.](https://www.youtube.com/watch?v=vRpmx5Iusdo&t=1s&ab_channel=S_X)

# Nodes 

## Make Frame Dataset
Input a path to extracted frames

## Load Frame From Dataset
Loads a frame from the frame folder

## Extract optical flow
Input 2 frames, get optical flow between them, and consistency masks

## WarpFrame
Applies optical flow to a frame

## LoadFrameFromFolder
Used to load rendered frames from the output folder for loopback.
Loads frame from folder. Updates frame list on each call. If there are no frames in the folder, returns init_image.

## LoadFramePairFromDataset
Returns 2 frames: current and previous.

## ResizeToFit
Resize an image to fit, keeping the aspect ratio.

## SaveFrame
Save frame to a folder, using current frame number.

## MixConsistencyMaps
Mix consistency maps, blur, dilate.

## RenderVideo 
Trigger output video render at a given frame 

## Schedulers: 
SchedulerString
SchedulerFloat
SchedulerInt
Provide disco-style schedules.
A list of values or a dictionary with keyframes and input frame number to index into the schedule.

- v0.4.2 : [WarpFusion: ComfyWarp v0.4.2 (schedulers, flow_blend)](https://www.youtube.com/watch?v=CdP8fus_vNg)
