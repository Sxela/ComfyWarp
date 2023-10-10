class OffsetNumber:
    @classmethod
    def INPUT_TYPES(self):
        return {"required":
                    {
                        "number": ("INT",{"default": 0, "min": -999999999999, "max": 9999999999}),
                        "offset": ("INT", {"default": 0, "min": -999999999999, "max": 9999999999}),
                    }
                }
    
    CATEGORY = "WarpFusion"

    RETURN_TYPES = ("INT",)
    FUNCTION = "offset_number"

    def offset_number(self, number, offset):
        return (number+offset,)

NODE_CLASS_MAPPINGS = {
    "OffsetNumber": OffsetNumber
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OffsetNumber": "Offset Number"
}
