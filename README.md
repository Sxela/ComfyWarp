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

**start_frame** - frame to begin with
**end_frame** - frame to end with
**nth_frame** - n-th frame to extract

If you provide a path to a video, only a range of n-th frames between start_frame and end_frame will be extracted.
If you provide a folder or a glob pattern, only a range of n-th frames between start_frame and end_frame will be stored in the dataset.

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

FlowBlend Deflickering pipeline from warp's video output cell\
This part smooths out the frames by blending the current stylized output frame and previously stylized and warped output frame, where consistent areas are blended linearly, and inconsistent areas are taken from the current stylized output frame only. This smooths non-moving parts and helps reduce trails on moving parts.
Renders a video from frames following the {frames_folder}{batch_name}%06d.png pattern.

#### Inputs

`output_dir`:
folder to put the rendered video to. Will be created automatically

`frames_input_dir`:
folder to get frames from, plug your SaveFrame output dir here

`batch_name`:
batch name you've set in your SaveFrame node, default - ComfyWarp

`first_frame`:
start video from that frame, default: 0

`last_frame`:
end video at that frame, default: -1, means use all available frames

`render_at_frame`:
frame at which to begin rendering video. Plug your FrameDataset total frame number here or specify manually if you want to render video before diffusing all the frames.

`current_frame`:
current frame being rendered, used to trigger video render. don't enter manually, Plug your current frame variable here.

`fps`:
output fps

`output_format`:
codec to use, h264_mp4, qtrle_mov, prores_mov

`use_deflicker`:
enable ffmpeg built-in deflicker

## Schedulers: 
- SchedulerString
- SchedulerFloat
- SchedulerInt

Added Scheduler nodes, by output variable type: `string`, `int`, `float`.
Can be used with any ComfyUI node inputs. For example, for automating parameter testing, scheduling controlnet weights, sampler settings, and prompts.
Require a current frame input to drive the scheduled sampling. Accepts any int as current frame input.


Input formats:
- single value: `value`
- list of consecutive values: `[frame1_value, frame_2value, ..., frameN_value]`
- dictionary of keyframe:value pairs: `{0: frame1_value, 1: frame2_value, ... N: frameN_Value}`
The dictionary format supports interpolation of values between frames, just like in WarpFusion.

## FixedQueue node
start - frame to begin with
end - frame to end with
current_frame - iterator, showing the current frame, which is being output as the current value. The current value should be plugged into downstream nodes as the source of the current frame number.

**Queue Many button**

Click to render a set of frames from start to end. Plug end output to your RenderVideo node's render_at_frame input to automatically render video after finishing the end frame.


## Flow_blend pipeline
Works like its WarpFusion counterpart.
blends previously stylized and warped frame (with cc mask applied) with the corresponding raw video frame. Acts like style opacity. 0 - no style, only raw frame, 1 - only stylized frame, intermediary values - linear interpolation between raw and stylized frame.

- v0.4.2 : [WarpFusion: ComfyWarp v0.4.2 (schedulers, flow_blend)](https://www.youtube.com/watch?v=CdP8fus_vNg)
  
