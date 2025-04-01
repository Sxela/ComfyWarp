from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights
from torchvision.models.optical_flow import raft_large, raft_small
import torch
from .flow_utils import apply_warp, get_flow_and_mask, mix_cc

raft_weights = Raft_Large_Weights.C_T_SKHT_V1
raft_device = "cuda" if torch.cuda.is_available() else "cpu"


class ExtractOpticalFlow:
    @classmethod
    def INPUT_TYPES(self):
        return {"required":
                    {
                        "current_frame": ("IMAGE",), 
                        "previous_frame": ("IMAGE",), 
                        "num_flow_updates": ("INT", {"default": 20, "min": 5, "max": 100})
                    }
                }
    
    CATEGORY = "WarpFusion"
    RETURN_TYPES = ("BACKWARD_FLOW", "MASK", "MASK", "MASK", "IMAGE")
    RETURN_NAMES = ("Flow", "Motion edge mask", "Occlusion mask", "Border mask", "Flow preview")
    FUNCTION = "get_flow"

    raft_model = raft_large(weights=raft_weights, progress=False).to(raft_device).half()

    def get_flow(self, current_frame, previous_frame, num_flow_updates):
        flow, flow_imgs, edge_mask, occlusion_mask, border_mask = get_flow_and_mask(previous_frame, current_frame, num_flow_updates=num_flow_updates, raft_model=self.raft_model, edge_width=11, dilation=2)
        return (flow, edge_mask, occlusion_mask, border_mask, flow_imgs, )
    

class WarpFrame:
    @classmethod
    def INPUT_TYPES(self):
        return {"required":
                    {
                        "previous_frame": ("IMAGE",), 
                        "flow": ("BACKWARD_FLOW",), 
                    }
                }
    
    CATEGORY = "WarpFusion"
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "warp"

    def warp(self, previous_frame, flow):
        warped_frame = apply_warp(previous_frame, flow, padding=0.2)
        
        return (warped_frame, )
    
class MixConsistencyMaps:
    @classmethod
    def INPUT_TYPES(self):
        return {"required":
                    {
                        "missed_consistency": ("MASK",), 
                        "overshoot_consistency": ("MASK",), 
                        "edge_consistency": ("MASK",), 
                        "blur": ("INT", {"default": 1, "min": 0, "max": 100}),
                        "dilate":("INT", {"default": 2, "min": 0, "max": 100}),
                        "force_binary":("BOOLEAN", {"default": True}),
                        "missed_consistency_weight":("FLOAT", {"default": 1.0, "min": 0.0, "max": 1, "step": 0.01}),
                        "overshoot_consistency_weight":("FLOAT", {"default": 1.0, "min": 0.0, "max": 1, "step": 0.01}),
                        "edges_consistency_weight":("FLOAT", {"default": 1.0, "min": 0.0, "max": 1, "step": 0.01}),

                    }
                }
    
    CATEGORY = "WarpFusion"
    RETURN_TYPES = ("MASK", )
    FUNCTION = "get_mixed_cc"

    def get_mixed_cc(self, missed_consistency, overshoot_consistency, edge_consistency, blur, dilate, 
                     force_binary, missed_consistency_weight, overshoot_consistency_weight, edges_consistency_weight):
        mixed = mix_cc(missed_consistency, overshoot_consistency, edge_consistency, blur=blur, dilate=dilate, missed_consistency_weight=missed_consistency_weight, 
           overshoot_consistency_weight=overshoot_consistency_weight, edges_consistency_weight=edges_consistency_weight, force_binary=force_binary)
        
        return (mixed, )
    

class ExtractFlowAndMixConsistencyMaps:
    @classmethod
    def INPUT_TYPES(self):
        return {"required":
                    {
                        "current_frame": ("IMAGE",), 
                        "previous_frame": ("IMAGE",), 
                        "num_flow_updates": ("INT", {"default": 20, "min": 5, "max": 100}),
                        "blur": ("INT", {"default": 1, "min": 0, "max": 100}),
                        "dilate":("INT", {"default": 2, "min": 0, "max": 100}),
                        "force_binary":("BOOLEAN", {"default": True}),
                        "missed_consistency_weight":("FLOAT", {"default": 1.0, "min": 0.0, "max": 1, "step": 0.01}),
                        "overshoot_consistency_weight":("FLOAT", {"default": 1.0, "min": 0.0, "max": 1, "step": 0.01}),
                        "edges_consistency_weight":("FLOAT", {"default": 1.0, "min": 0.0, "max": 1, "step": 0.01}),
                    }
                }
    
    CATEGORY = "WarpFusion"
    RETURN_TYPES = ("BACKWARD_FLOW", "MASK", "MASK", "MASK", "IMAGE", "MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("Flow", "Motion edge mask", "Occlusion mask", "Border mask", "Flow preview", "Mixed consistency map", "Current frame", "Previous frame")
    FUNCTION = "get_flow_and_mixed_cc"

    raft_model = raft_large(weights=raft_weights, progress=False).to(raft_device).half()

    def get_flow_and_mixed_cc(self, current_frame, previous_frame, num_flow_updates, blur, dilate, force_binary, missed_consistency_weight, overshoot_consistency_weight, edges_consistency_weight):
        flow, flow_imgs, edge_mask, occlusion_mask, border_mask = get_flow_and_mask(previous_frame, current_frame, num_flow_updates=num_flow_updates, raft_model=self.raft_model, edge_width=11, dilation=2)
        mixed = mix_cc(missed_cc=occlusion_mask, overshoot_cc=border_mask, edge_cc=edge_mask, blur=blur, dilate=dilate, missed_consistency_weight=missed_consistency_weight, 
           overshoot_consistency_weight=overshoot_consistency_weight, edges_consistency_weight=edges_consistency_weight, force_binary=force_binary)
        
        return (flow, edge_mask, occlusion_mask, border_mask, flow_imgs, mixed, current_frame, previous_frame)

class KeyframedFlowApplication:
    @classmethod
    def INPUT_TYPES(self):
        return {"required": {
            "motion_source_frames": ("IMAGE",),
            "frames_to_warp": ("IMAGE",),
            "keyframe_weights": ("STRING", {
                "multiline": True,
                "default": '{"0": 1.0, "10": 2.0}'
            }),
            "keyframe_repeats": ("STRING", {
                "multiline": True,
                "default": '{"0": 1, "10": 3}'
            }),
            "num_flow_updates": ("INT", {"default": 20, "min": 5, "max": 100})
        }}
    
    CATEGORY = "WarpFusion"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Processed Frame",)
    FUNCTION = "process_frames"

    raft_model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V1, progress=False).to("cuda" if torch.cuda.is_available() else "cpu").half()

    def process_frames(self, motion_source_frames, frames_to_warp, keyframe_weights, keyframe_repeats, num_flow_updates):
        # Parse the keyframe dictionaries
        print('keyframe_weights', keyframe_weights)
        print('keyframe_repeats', keyframe_repeats)
        weights = eval(keyframe_weights)
        repeats = eval(keyframe_repeats)

        if type(weights) == list:
            weights = {str(i): v for i, v in enumerate(weights) if v > 1}
        if type(repeats) == list:
            repeats = {str(i): v for i, v in enumerate(repeats) if v > 1}

        weights = {str(k): v for k, v in weights.items()}
        repeats = {str(k): int(v) for k, v in repeats.items()}

        print('parsed weights', weights)
        print('parsed repeats', repeats)
        
        # Convert frames to list if they're not already
        processed_frames = []
        num_frames = len(motion_source_frames)

        flow_dict = {}

        # Sort keyframes to process them in order

        weights = [weights.get(str(frame_number), 1.0) for frame_number in range(num_frames-1)]
        repeats = [repeats.get(str(frame_number), 1) for frame_number in range(num_frames-1)]

        flow_map = {}
        frame_number = 0
        while frame_number < num_frames-1:
            repeat_count = repeats[frame_number]
            if repeat_count <= 1: 
                frame_number += 1
                continue
            if repeat_count > 1:
                for i in range(repeat_count):
                    flow_map[frame_number + i] = frame_number
                frame_number += repeat_count - 1

        # print('flow_map', flow_map)
        # print('weights', weights)
        # print('repeats', repeats)

        from tqdm import trange
        for frame_number in trange(num_frames-1):
            # Find the active keyframe
            weight = weights[frame_number]
            if (frame_number == 0) or (frame_number == num_frames-1) or ((frame_number not in flow_map) and(weight <= 1.0)):
                processed_frames.append(frames_to_warp[frame_number:frame_number+1])
                continue

            # print('\napplying flow for frame', frame_number)

            # Extract flow between source frames
            flow_frame = flow_map.get(frame_number, frame_number)
            if flow_frame != frame_number:
                print('applying flow for frame', frame_number, 'from frame', flow_frame)
 
            if flow_frame not in flow_dict:
                    flow, _, _, _, _ = get_flow_and_mask(
                        motion_source_frames[flow_frame:flow_frame+1],
                        motion_source_frames[flow_frame+1:flow_frame+2],
                        num_flow_updates=num_flow_updates,
                        raft_model=self.raft_model
                    )
                    flow_dict[frame_number] = flow
            else: 
                    flow = flow_dict[flow_frame]
            # print('flow', flow.max(), flow.min())
            
            if flow_frame != frame_number:
                warped_frame = processed_frames[-1]
            else:
                warped_frame = frames_to_warp[frame_number:frame_number+1]

            warped_frame = apply_warp(
                        warped_frame,
                        flow * weight,
                        padding=0.2
                    )
            processed_frames.append(warped_frame)

        # Concatenate all processed frames
        output_frames = torch.cat(processed_frames, dim=0)
        return (output_frames,)

NODE_CLASS_MAPPINGS = {
    "ExtractOpticalFlow": ExtractOpticalFlow,
    "WarpFrame":WarpFrame,
    "MixConsistencyMaps":MixConsistencyMaps,
    "ExtractFlowAndMixConsistencyMaps":ExtractFlowAndMixConsistencyMaps,
    "KeyframedFlowApplication": KeyframedFlowApplication
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExtractOpticalFlow": "ExtractOpticalFlow",
    "WarpFrame":"WarpFrame",
    "MixConsistencyMaps":"MixConsistencyMaps",
    "ExtractFlowAndMixConsistencyMaps":"ExtractFlowAndMixConsistencyMaps",
    "KeyframedFlowApplication": "KeyframedFlowApplication"
}
        
