import glob
import os
from PIL import Image, ImageOps
import torch
import numpy as np
import folder_paths
from .frame_utils import FrameDataset, StylizedFrameDataset, get_size, save_video

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
                        "update_on_frame_load": ("BOOLEAN", {"default": True})             
                    },
                }
    
    CATEGORY = "WarpFusion"
    RETURN_TYPES = ("FRAME_DATASET", "INT")
    RETURN_NAMES = ("FRAME_DATASET", "Total_frames")
    FUNCTION = "get_frames"

    def get_frames(self, file_path, update_on_frame_load):
        ds = FrameDataset(file_path, outdir_prefix='', videoframes_root=folder_paths.get_output_directory(), 
                          update_on_getitem=update_on_frame_load)
        return (ds,len(ds))

    @classmethod
    def VALIDATE_INPUTS(self, file_path, update_on_frame_load):
        _, n_frames = self.get_frames(self, file_path, update_on_frame_load)
        if n_frames==0:
            return f"Found 0 frames in path {file_path}"

        return True
    
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
                    }
                }
    
    CATEGORY = "WarpFusion"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Image",)
    FUNCTION = "resize"

    def resize(self, image, max_size):
        image = image.transpose(1,-1)
        size = image.shape[2:]
        size = get_size(size, max_size)

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

    RETURN_TYPES = ()
    FUNCTION = "save_img"
    OUTPUT_NODE = True
    type = 'output'

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
        return ()


NODE_CLASS_MAPPINGS = {
    "LoadFrameSequence": LoadFrameSequence,
    "LoadFrame": LoadFrame,
    "LoadFrameFromDataset":LoadFrameFromDataset,
    "MakeFrameDataset":MakeFrameDataset,
    "LoadFramePairFromDataset":LoadFramePairFromDataset,
    "LoadFrameFromFolder":LoadFrameFromFolder,
    "ResizeToFit":ResizeToFit,
    "SaveFrame":SaveFrame,
    "RenderVideo": RenderVideo
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
    "RenderVideo": "RenderVideo"
}
