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
        print(flow_imgs.shape, flow_imgs.max(), type(flow_imgs))
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
    

NODE_CLASS_MAPPINGS = {
    "ExtractOpticalFlow": ExtractOpticalFlow,
    "WarpFrame":WarpFrame,
    "MixConsistencyMaps":MixConsistencyMaps

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExtractOpticalFlow": "ExtractOpticalFlow",
    "WarpFrame":"WarpFrame",
    "MixConsistencyMaps":"MixConsistencyMaps"
}
        
