from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights
from torchvision.models.optical_flow import raft_large, raft_small
import torch
from .flow_utils import apply_warp, get_flow_and_mask

raft_weights = Raft_Large_Weights.C_T_SKHT_V1
raft_device = "cuda" if torch.cuda.is_available() else "cpu"


class ExtractOpticalFlow:
    @classmethod
    def INPUT_TYPES(self):
        return {"required":
                    {
                        "current_frame": ("IMAGE",), 
                        "previous_frame": ("IMAGE",), 
                    }
                }
    
    CATEGORY = "WarpFusion"
    RETURN_TYPES = ("BACKWARD_FLOW", "MASK", "MASK", "MASK", "IMAGE")
    RETURN_NAMES = ("Flow", "Motion edge mask", "Occlusion mask", "Border mask", "Flow preview")
    FUNCTION = "get_flow"

    raft_model = raft_large(weights=raft_weights, progress=False).to(raft_device).half()

    def get_flow(self, current_frame, previous_frame):
        flow, flow_imgs, edge_mask, occlusion_mask, border_mask = get_flow_and_mask(previous_frame, current_frame, num_flow_updates=20, raft_model=self.raft_model, edge_width=11, dilation=2)
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
    
    

NODE_CLASS_MAPPINGS = {
    "ExtractOpticalFlow": ExtractOpticalFlow,
    "WarpFrame":WarpFrame

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExtractOpticalFlow": "ExtractOpticalFlow",
    "WarpFrame":"WarpFrame"
}
        
