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
            "source_frames": ("IMAGE",),
            "stylized_frames": ("IMAGE",),
            "keyframe_weights": ("STRING", {
                "multiline": True,
                "default": '{"0": 1.0, "10": 2.0}'
            }),
            "keyframe_repeats": ("STRING", {
                "multiline": True,
                "default": '{"0": 1, "10": 3}'
            }),
            "frame_number": ("INT", {"default": 0, "min": 0}),
            "num_flow_updates": ("INT", {"default": 20, "min": 5, "max": 100})
        }}
    
    CATEGORY = "WarpFusion"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("Processed Frame", "Flow Mask")
    FUNCTION = "process_frames"

    raft_model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V1, progress=False).to("cuda" if torch.cuda.is_available() else "cpu").half()

    def process_frames(self, source_frames, stylized_frames, keyframe_weights, keyframe_repeats, frame_number, num_flow_updates):
        # Parse the keyframe dictionaries
        weights = eval(keyframe_weights)
        repeats = eval(keyframe_repeats)
        
        # Find the active keyframe
        active_keyframe = None
        repeat_count = 1
        weight = 1.0
        
        # Sort keyframes to process them in order
        keyframes = sorted(set(list(weights.keys()) + list(repeats.keys())))
        
        for kf in keyframes:
            if frame_number >= kf:
                active_keyframe = kf
                weight = weights.get(kf, 1.0)
                repeat_count = repeats.get(kf, 1)
            else:
                break
                
        if active_keyframe is None:
            return (stylized_frames[frame_number:frame_number+1], torch.ones((1, stylized_frames.shape[1], stylized_frames.shape[2], 1)))
            
        # If we're within repeat range and repeat > 1
        if repeat_count > 1 and frame_number < active_keyframe + repeat_count:
            # Extract flow between source frames
            flow, _, _, _, _ = get_flow_and_mask(
                source_frames[active_keyframe:active_keyframe+1],
                source_frames[frame_number:frame_number+1],
                num_flow_updates=num_flow_updates,
                raft_model=self.raft_model
            )
            
            # Apply flow to keyframe's stylized frame with weight multiplier
            warped_frame = apply_warp(
                stylized_frames[active_keyframe:active_keyframe+1],
                flow * weight,
                padding=0.2
            )
            
            # Generate a simple mask for visualization
            flow_mask = torch.ones((1, stylized_frames.shape[1], stylized_frames.shape[2], 1))
            
            return (warped_frame, flow_mask)
            
        # If weight > 1, apply weighted flow
        elif weight > 1.0:
            # Extract flow between source frames
            flow, _, _, _, _ = get_flow_and_mask(
                source_frames[frame_number-1:frame_number],
                source_frames[frame_number:frame_number+1],
                num_flow_updates=num_flow_updates,
                raft_model=self.raft_model
            )
            
            # Apply weighted flow to stylized frames
            warped_frame = apply_warp(
                stylized_frames[frame_number-1:frame_number],
                flow * weight,
                padding=0.2
            )
            
            # Generate a simple mask for visualization
            flow_mask = torch.ones((1, stylized_frames.shape[1], stylized_frames.shape[2], 1))
            
            return (warped_frame, flow_mask)
            
        # Otherwise return the original stylized frame
        return (stylized_frames[frame_number:frame_number+1], torch.ones((1, stylized_frames.shape[1], stylized_frames.shape[2], 1)))

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
    "KeyframedFlowApplication": "Keyframed Flow Application"
}
        
