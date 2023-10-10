from PIL import Image, ImageOps 
import torch
import numpy as np
import glob, os, hashlib, pathlib, sys, subprocess
from .frame_utils import FrameDataset
import folder_paths

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
                                                 "default":"C:\\code\\warp\\19_cn_venv\\images_out\\stable_warpfusion_0.20.0\\videoFrames\\650571deef_0_0_1"})
                    }
                }
    
    CATEGORY = "WarpFusion"
    RETURN_TYPES = ("FRAME_DATASET", "INT")
    RETURN_NAMES = ("FRAME_DATASET", "Total_frames")
    FUNCTION = "get_frames"

    def get_frames(self, file_path):
        ds = FrameDataset(file_path, outdir_prefix='', videoframes_root=folder_paths.get_output_directory())
        return (ds,len(ds))

    @classmethod
    def VALIDATE_INPUTS(self, file_path):
        _, n_frames = self.get_frames(self, file_path)
        if n_frames==0:
            return f"Found 0 frames in path {file_path}"

        return True

    
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
        print(frame_dataset[-1], frame_number, total_frames)
        frame_number = min(min(frame_number, total_frames), len(frame_dataset)-1)
        print(frame_number)
        image_path = frame_dataset[frame_number]

        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        print('load_frame', image.shape, frame_number, total_frames)
        
        return (image, frame_number)

NODE_CLASS_MAPPINGS = {
    "LoadFrameSequence": LoadFrameSequence,
    "LoadFrame": LoadFrame,
    "LoadFrameFromDataset":LoadFrameFromDataset,
    "MakeFrameDataset":MakeFrameDataset
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadFrameSequence": "Load Frame Sequence",
    "LoadFrame":"Load Frame",
    "LoadFrameFromDataset":"LoadFrame From Dataset",
    "MakeFrameDataset":"Make Frame Dataset"
}
