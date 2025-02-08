import glob
import os
from PIL import Image, ImageOps
import torch
import numpy as np
import folder_paths
from .frame_utils import FrameDataset, StylizedFrameDataset, get_scheduled_arg, get_size, save_video

class ApplyMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination": ("IMAGE",),
                "source": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"

    CATEGORY = "WarpFusion"

    def composite(self, destination, source, mask = None):
        
        mask = mask[..., None].repeat(1,1,1,destination.shape[-1])
        res = destination*(1-mask) + source*(mask)
        return (res,)
    
class ApplyMaskConditional:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination": ("IMAGE",),
                "source": ("IMAGE",),
                "current_frame_number": ("INT",),
                "apply_at_frames": ("STRING",),
                "don_not_apply_at_frames": ("BOOLEAN",),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"

    CATEGORY = "WarpFusion"

    def composite(self, destination, source, current_frame_number, apply_at_frames, don_not_apply_at_frames, mask = None):
        idx_list = [int(i) for i in apply_at_frames.split(',')]
        if (current_frame_number not in idx_list) if don_not_apply_at_frames else (current_frame_number in idx_list):
            # Convert mask to correct format for interpolation [b,c,h,w]
            mask = mask[None,...] 
            
            # Resize mask to destination size using explicit dimensions
            mask = torch.nn.functional.interpolate(mask, size=(destination.shape[1], destination.shape[2]), mode='bilinear')
            
            # Convert back to [b,h,w,1] format
            mask = mask[0,...,None].repeat(1,1,1,destination.shape[-1])
           
            source = source.permute(0,3,1,2)
            source = torch.nn.functional.interpolate(source, size=(destination.shape[1], destination.shape[2]), mode='bilinear')
            source = source.permute(0,2,3,1)
            
            res = destination*(1-mask) + source*(mask)
            return (res,)
        else:
            return (destination,)

class ApplyMaskLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination": ("LATENT",),
                "source": ("LATENT",),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "composite"

    CATEGORY = "WarpFusion"

    def composite(self, destination, source, mask = None):
        destination = destination['samples']
        source = source['samples']
        mask = mask[None, ...]
        mask = torch.nn.functional.interpolate(mask, size=(destination.shape[2], destination.shape[3]))
        res = destination*(1-mask) + source*(mask)
        return ({"samples":res}, )
    
class ApplyMaskLatentConditional:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination": ("LATENT",),
                "source": ("LATENT",),
                "current_frame_number": ("INT",),
                "apply_at_frames": ("STRING",),
                "don_not_apply_at_frames": ("BOOLEAN",),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "composite"

    CATEGORY = "WarpFusion"

    def composite(self, destination, source, current_frame_number, apply_at_frames, don_not_apply_at_frames, mask = None):
        destination = destination['samples']
        source = source['samples']
        idx_list = [int(i) for i in apply_at_frames.split(',')]
        if (current_frame_number not in idx_list) if don_not_apply_at_frames else (current_frame_number in idx_list):
            mask = mask[None, ...]
            mask = torch.nn.functional.interpolate(mask, size=(destination.shape[2], destination.shape[3]))
            res = destination*(1-mask) + source*(mask)
            return ({"samples":res}, )
        else:
            return ({"samples":destination}, )

class LoadFrameSequence:
    @classmethod
    def INPUT_TYPES(self):
        return {"required":
                    {
                        "file_path": ("STRING", {"multiline": True, 
                                                 "default":"C:\\code\\warp\\19_cn_venv\\images_out\\stable_warpfusion_0.20.0\\videoFrames\\650571deef_0_0_1"})
                    }
                }
    
    CATEGORY = "WarpFusion"
    OUTPUT_IS_LIST = (True,False)
    RETURN_TYPES = ("FRAMES", "INT")
    RETURN_NAMES = ("Frames", "Total_frames")
    FUNCTION = "get_frames"

    def get_frames(self, file_path):
        print(file_path)
        self.frames = glob.glob(file_path+'/**/*.*', recursive=True)
        self.max_frames = len(self.frames)
        print(f'Found {len(self.frames)} frames.')
        out = [{'image':frame, 'max_frames':self.max_frames}for frame in self.frames]
        return (out,self.max_frames)

    @classmethod
    def IS_CHANGED(self, file_path):
        self.get_frames(self, file_path)

    @classmethod
    def VALIDATE_INPUTS(self, file_path):
        self.get_frames(self, file_path)
        if len(self.frames)==0:
            return f"Found 0 frames in path {file_path}"

        return True
    
class LoadFrame:
    @classmethod
    def INPUT_TYPES(self):
        return {"required":
                    {
                        "file_paths": ("FRAMES",),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 9999999999}),
                        "total_frames":("INT", {"default": 0, "min": 0, "max": 9999999999})
                    }
                }
    
    CATEGORY = "WarpFusion"

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE","INT")
    RETURN_NAMES = ("Image","Frame number")
    FUNCTION = "load_frame"
    #validation fails here for some reason 

    def load_frame(self, file_paths, seed, total_frames):
        frame_number = seed
        print(file_paths[:10], frame_number, total_frames)
        frame_number = frame_number[0]
        total_frames = total_frames[0]
        frame_number = min(min(frame_number, total_frames), len(file_paths)-1)
        print(frame_number)
        image_path = file_paths[frame_number]['image']

        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        print(image.shape, frame_number, total_frames)
        
        return (image, frame_number)

class MakeFrameDataset:
    @classmethod
    def INPUT_TYPES(self):
        return {"required":
                    {
                        "file_path": ("STRING", {"multiline": True, 
                                                 "default":"C:\\code\\warp\\19_cn_venv\\images_out\\stable_warpfusion_0.20.0\\videoFrames\\650571deef_0_0_1"}),
                        "update_on_frame_load": ("BOOLEAN", {"default": True}),
                        "start_frame":("INT", {"default": 0, "min": 0, "max": 9999999999}),
                        "end_frame":("INT", {"default": -1, "min": -1, "max": 9999999999}),
                        "nth_frame":("INT", {"default": 1, "min": 1, "max": 9999999999}),         
                        "overwrite":("BOOLEAN", {"default": False})
                    }
                }
    
    CATEGORY = "WarpFusion"
    RETURN_TYPES = ("FRAME_DATASET", "INT")
    RETURN_NAMES = ("FRAME_DATASET", "Total_frames")
    FUNCTION = "get_frames"

    def get_frames(self, file_path, update_on_frame_load, start_frame, end_frame, nth_frame, overwrite):
        ds = FrameDataset(file_path, outdir_prefix='', videoframes_root=folder_paths.get_output_directory(), 
                          update_on_getitem=update_on_frame_load, start_frame=start_frame, end_frame=end_frame, nth_frame=nth_frame, overwrite=overwrite)
        if len(ds)==0:
            raise Exception(f"Found 0 frames in path {file_path}") #thanks to https://github.com/Aljnk 
        return (ds,len(ds))
    
class LoadFrameFromFolder:
    @classmethod
    def INPUT_TYPES(self):
        return {"required":
                    {
                        "file_path": ("STRING", {"multiline": True, 
                                                 "default":"C:\\code\\warp\\19_cn_venv\\images_out\\stable_warpfusion_0.20.0\\videoFrames\\650571deef_0_0_1"}),                
                        "init_image":("IMAGE",) ,
                        "frame_number":("INT", {"default": 0, "min": 0, "max": 9999999999}),
                        "fit_into": ("INT", {"default": 1280, "min": 0, "max": 8196*2}),

                    },
                }
    
    CATEGORY = "WarpFusion"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get_frames"

    def load_image(self, image_path, fit_into):
        
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        size = get_size(i.size, max_size=fit_into, divisible_by=8)
        image = i.convert("RGB").resize(size)
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return image

    def get_frames(self, file_path, init_image, frame_number, fit_into):
        if frame_number == -1: return  (init_image,)
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)
        ds = StylizedFrameDataset(file_path)
        frame_number = min(frame_number, len(ds)-1)
        frame_number = max(0, frame_number)
        if len(ds) == 0: return (init_image,)
        return (self.load_image(ds[frame_number], fit_into),)

    
class LoadFrameFromDataset:
    @classmethod
    def INPUT_TYPES(self):
        return {"required":
                    {
                        "frame_dataset": ("FRAME_DATASET",),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 9999999999}),
                        "total_frames":("INT", {"default": 0, "min": 0, "max": 9999999999})
                    }
                }
    
    CATEGORY = "WarpFusion"

    RETURN_TYPES = ("IMAGE","INT")
    RETURN_NAMES = ("Image","Frame number")
    FUNCTION = "load_frame"

    def load_frame(self, frame_dataset, seed, total_frames):
        frame_number = seed
        frame_number = min(min(frame_number, total_frames), len(frame_dataset)-1)
        frame_number = max(0, frame_number)
        image_path = frame_dataset[frame_number]

        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        
        return (image, frame_number)
    
class LoadFramePairFromDataset:
    @classmethod
    def INPUT_TYPES(self):
        return {"required":
                    {
                        "frame_dataset": ("FRAME_DATASET",),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 9999999999}),
                        "total_frames":("INT", {"default": 0, "min": 0, "max": 9999999999}),
                        "fit_into": ("INT", {"default": 1280, "min": 0, "max": 8196*2}),
                        
                    }
                }
    
    CATEGORY = "WarpFusion"

    RETURN_TYPES = ("IMAGE","IMAGE","INT")
    RETURN_NAMES = ("Current frame","Previous Frame","Frame number")
    FUNCTION = "load_frames"

    def load_frame(self, frame_dataset, seed, total_frames, fit_into):
        frame_number = seed
        frame_number = min(min(frame_number, total_frames), len(frame_dataset)-1)
        frame_number = max(0, frame_number)
        image_path = frame_dataset[frame_number]

        i = Image.open(image_path)
        size = get_size(i.size, fit_into, divisible_by=8)
        i = ImageOps.exif_transpose(i).resize(size)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        
        return image
    
    def load_frames(self, frame_dataset, seed, total_frames, fit_into):
        current_frame = self.load_frame(frame_dataset, seed, total_frames, fit_into)
        previous_frame = self.load_frame(frame_dataset, seed-1, total_frames, fit_into)
        return (current_frame, previous_frame, seed)
    
class ResizeToFit:
    @classmethod
    def INPUT_TYPES(self):
        return {"required":
                    {
                        "image": ("IMAGE",),
                        "max_size": ("INT", {"default": 1280, "min": 0, "max": 9999999999}),
                        "divisible_by": ("INT", {"default": 64, "min": 2, "max": 2048}),
                    }
                }
    
    CATEGORY = "WarpFusion"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Image",)
    FUNCTION = "resize"

    def resize(self, image, max_size, divisible_by):
        image = image.transpose(1,-1)
        size = image.shape[2:]
        size = get_size(size, max_size, divisible_by)

        image = torch.nn.functional.interpolate(image, size)
        image = image.transpose(1,-1)
        return (image, )
    
class SaveFrame:
    @classmethod
    def INPUT_TYPES(self):
        return {"required":
                    {
                        "image": ("IMAGE",),
                        "output_dir": ("STRING", {"multiline": True, 
                                                 "default":''}),
                        "batch_name": ("STRING",{"default": "ComfyWarp"}),
                        "frame_number":("INT",{"default": 0, "min": 0, "max": 9999999999}),
                    }
                }
    
    CATEGORY = "WarpFusion"

    FUNCTION = "save_img"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    def save_img(self, image, output_dir, batch_name, frame_number):
        os.makedirs(output_dir, exist_ok=True)
        fname = f'{batch_name}_{frame_number:06}.png'
        out_fname  = os.path.join(output_dir, fname)
        print('image.shape', image.shape, image.max(), image.min())
        image = (image[0].clip(0,1)*255.).cpu().numpy().astype('uint8')
        image = Image.fromarray(image)
        image.save(out_fname)
        print('fname', out_fname)
        return ()
    
class RenderVideo:
    @classmethod
    def INPUT_TYPES(self):
        return {"required":
                    {
                        "output_dir": ("STRING", {"multiline": True, 
                                                 "default":''}),
                        "frames_input_dir": ("STRING", {"multiline": True, 
                                                 "default":''}),
                        "batch_name": ("STRING", {"default":'ComfyWarp'}),
                        "first_frame":("INT",{"default": 0, "min": 0, "max": 9999999999}),
                        "last_frame":("INT",{"default": -1, "min": -1, "max": 9999999999}),
                        "render_at_frame":("INT",{"default": 0, "min": 0, "max": 9999999999}),
                        "current_frame":("INT",{"default": 0, "min": 0, "max": 9999999999}),
                        "fps":("FLOAT",{"default": 24, "min": 0, "max": 9999999999}),
                        "output_format":(["h264_mp4", "qtrle_mov", "prores_mov"],),
                        "use_deflicker": ("BOOLEAN", {"default": False})   
                    }
                }
    
    CATEGORY = "WarpFusion"

    RETURN_TYPES = ()
    FUNCTION = "export_video"
    OUTPUT_NODE = True

    def export_video(self, output_dir, frames_input_dir, batch_name, first_frame=1, last_frame=-1, 
                     render_at_frame=999999, current_frame=0, fps=30, output_format='h264_mp4', use_deflicker=False):
        if current_frame>=render_at_frame:
            print('Exporting video.')
            save_video(indir=frames_input_dir, video_out=output_dir, batch_name=batch_name, start_frame=first_frame, 
                       last_frame=last_frame, fps=fps, output_format=output_format, use_deflicker=use_deflicker)
            # raise Exception(f'Exported video successfully. This exception is raised to just stop the endless cycle :D.\n you can find your video at {output_dir}')
        return ()
    
class SchedulerInt:
    @classmethod
    def INPUT_TYPES(self):
        return {"required":
                    {
                        "schedule": ("STRING", {"multiline": True, 
                                                 "default":''}),
                        "frame_number":("INT",{"default": 0, "min": 0, "max": 9999999999}),
                        "blend_json": ("BOOLEAN", {"default": True})   
                    }
                }
    
    CATEGORY = "WarpFusion"

    RETURN_TYPES = ("INT", )
    FUNCTION = "get_value"

    def get_value(self, schedule, frame_number, blend_json=True):
        value = get_scheduled_arg(frame_num=frame_number, schedule=schedule, blend_json_schedules=blend_json)
        return (int(value),)
    
class SchedulerFloat:
    @classmethod
    def INPUT_TYPES(self):
        return {"required":
                    {
                        "schedule": ("STRING", {"multiline": True, 
                                                 "default":''}),
                        "frame_number":("INT",{"default": 0, "min": 0, "max": 9999999999}),
                        "blend_json": ("BOOLEAN", {"default": True})   
                    }
                }
    
    CATEGORY = "WarpFusion"

    RETURN_TYPES = ("FLOAT", )
    FUNCTION = "get_value"

    def get_value(self, schedule, frame_number, blend_json=True):
        value = get_scheduled_arg(frame_num=frame_number, schedule=schedule, blend_json_schedules=blend_json)
        return (float(value),)
    
class SchedulerString:
    @classmethod
    def INPUT_TYPES(self):
        return {"required":
                    {
                        "schedule": ("STRING", {"multiline": True, 
                                                 "default":''}),
                        "frame_number":("INT",{"default": 0, "min": 0, "max": 9999999999}),
                        
                    }
                }
    
    CATEGORY = "WarpFusion"

    RETURN_TYPES = ("STRING", )
    FUNCTION = "get_value"

    def get_value(self, schedule, frame_number):
        value = get_scheduled_arg(frame_num=frame_number, schedule=schedule, blend_json_schedules=False)
        return (str(value),)
    
"""
inspired by https://github.com/Fannovel16/ComfyUI-Loopchain
"""
class FixedQueue:
    @classmethod
    def INPUT_TYPES(self):
        return {"required":
                    {
                        "start": ("INT",{"default": 0, "min": 0, "max": 9999999999}),
                        "end":("INT",{"default": 1, "min": 0, "max": 9999999999}),
                        "current_number":("INT",{"default": 0, "min": 0, "max": 9999999999}),
                    }
                }
    
    CATEGORY = "WarpFusion"

    RETURN_TYPES = ("INT", "INT", "INT",)
    RETURN_NAMES = ("current", "start", "end",)
    FUNCTION = "get_value"

    def get_value(self, start, end, current_number):
        return (current_number, start, end)

class MakePaths:
    @classmethod
    def INPUT_TYPES(self):
        return {"required": {
            "root_path": ("STRING", {"multiline": True, "default": "./"}),
            "experiment": ("STRING", {"default": "experiment"}),
            "video": ("STRING", {"default": "video"}),
            "frames": ("STRING", {"default": "frames"}),
            "smoothed": ("STRING", {"default": "smoothed"}),
        }}
    
    CATEGORY = "WarpFusion"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_path", "frames_path", "smoothed_frames_path")
    FUNCTION = "build_paths"

    def build_paths(self, root_path, experiment, video, frames, smoothed):
        base_path = os.path.join(root_path, experiment)
        video_path = os.path.join(base_path, video)
        frames_path = os.path.join(base_path, frames)
        smoothed_frames_path = os.path.join(base_path, smoothed)
        
        return (video_path, frames_path, smoothed_frames_path)

NODE_CLASS_MAPPINGS = {
    "LoadFrameSequence": LoadFrameSequence,
    "LoadFrame": LoadFrame,
    "LoadFrameFromDataset":LoadFrameFromDataset,
    "MakeFrameDataset":MakeFrameDataset,
    "LoadFramePairFromDataset":LoadFramePairFromDataset,
    "LoadFrameFromFolder":LoadFrameFromFolder,
    "ResizeToFit":ResizeToFit,
    "SaveFrame":SaveFrame,
    "RenderVideo": RenderVideo,
    "SchedulerString":SchedulerString,
    "SchedulerFloat":SchedulerFloat,
    "SchedulerInt":SchedulerInt,
    "FixedQueue":FixedQueue,
    "ApplyMask":ApplyMask,
    "ApplyMaskConditional":ApplyMaskConditional,
    "ApplyMaskLatent":ApplyMaskLatent,
    "ApplyMaskLatentConditional":ApplyMaskLatentConditional,
    "MakePaths": MakePaths,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadFrameSequence": "Load Frame Sequence",
    "LoadFrame":"Load Frame",
    "LoadFrameFromDataset":"Load Frame From Dataset",
    "MakeFrameDataset":"Make Frame Dataset",
    "LoadFramePairFromDataset":"Load Frame Pair From Dataset",
    "LoadFrameFromFolder": "Maybe Load Frame From Folder",
    "ResizeToFit":"Resize To Fit",
    "SaveFrame":"SaveFrame",
    "RenderVideo": "RenderVideo",
    "SchedulerString":"SchedulerString",
    "SchedulerFloat":"SchedulerFloat",
    "SchedulerInt":"SchedulerInt",
    "FixedQueue":"FixedQueue",
    "ApplyMask":"ApplyMask",
    "ApplyMaskConditional":"ApplyMaskConditional",
    "ApplyMaskLatent":"ApplyMaskLatent",
    "ApplyMaskLatentConditional":"ApplyMaskLatentConditional",
    "MakePaths": "Make Paths",
}
