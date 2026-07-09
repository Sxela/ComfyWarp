import torch


class WanBoilNoise:
    """Reintroduce hand-drawn 'boil' into Wan video by injecting per-frame-independent
    noise into the post-CFG x0 estimate during the late denoising steps only.

    Early steps set structure/motion; late steps set high-frequency detail. Adding fresh
    i.i.d. (temporally uncorrelated) noise that the remaining steps can't fully reconcile
    makes edges and fills wobble frame-to-frame while shapes/motion stay locked. Keep the
    amplitude small and start late, otherwise it drifts into full Deforum-style flicker.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "strength": ("FLOAT", {
                "default": 0.04, "min": 0.0, "max": 1.0, "step": 0.005,
                "tooltip": "Perturbation amplitude as a fraction of the latent std. Keep small: "
                           "~0.03-0.07 is the boil sweet spot, >0.15 drifts into full flicker. "
                           "0 = bypass (identical to the un-patched workflow).",
            }),
            "start_percent": ("FLOAT", {
                "default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "Begin injecting at this fraction through sampling. Higher = later = "
                           "pure detail boil; lower (e.g. 0.4) starts disturbing structure/motion.",
            }),
            "end_percent": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "Stop injecting at this fraction through sampling (1.0 = through the "
                           "final step, keeping the variation alive to the last frame).",
            }),
            "independence": ("FLOAT", {
                "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "Flicker<->boil knob. 1.0 = fully per-frame-independent noise (max "
                           "boil). Lower blends in noise shared across the temporal axis (adds no "
                           "flicker), softening toward temporal consistency.",
            }),
            "high_pass": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "Optional: remove low spatial frequencies from the injected noise so "
                           "variation concentrates in edges/detail and avoids low-freq color "
                           "drift across the frame. 0 = off.",
            }),
            "seed": ("INT", {
                "default": 0, "min": 0, "max": 0xffffffffffffffff,
                "control_after_generate": True,
                "tooltip": "Seed for the injected noise (CPU generator) — reproducible runs.",
            }),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "sampling/custom"

    def patch(self, model, strength, start_percent, end_percent, independence, high_pass, seed):
        m = model.clone()
        # model_sampling.percent_to_sigma: percent 0->high sigma (early), 1->sigma 0 (final step).
        # Monotonically decreasing, so start (earlier) gives the higher sigma bound.
        ms = m.get_model_object("model_sampling")
        sigma_hi = float(ms.percent_to_sigma(start_percent))  # earlier step -> higher sigma
        sigma_lo = float(ms.percent_to_sigma(end_percent))    # later step   -> lower sigma
        gen = torch.Generator(device="cpu").manual_seed(int(seed))

        def post_cfg(args):
            denoised = args["denoised"]
            sigma = float(args["sigma"].max())
            if strength <= 0.0 or sigma > sigma_hi or sigma < sigma_lo:
                return denoised

            # i.i.d. across every element (incl. temporal dim 2 for 5D latents) -> free
            # per-latent-frame independence. Generate on CPU for reproducibility.
            eps = torch.randn(denoised.shape, generator=gen, device="cpu").to(denoised)

            # consistency<->boil blend: mix in noise shared across temporal axis (adds no flicker)
            if independence < 1.0 and eps.dim() == 5:
                shared = eps[:, :, :1].repeat(1, 1, eps.shape[2], 1, 1)
                eps = independence * eps + (1.0 - independence) * shared

            # optional spatial high-pass so variation sits in edges/detail, not low-freq color
            if high_pass > 0.0:
                if eps.dim() == 5:
                    b, c, t, h, w = eps.shape
                    flat = eps.reshape(b * c * t, 1, h, w)
                    blur = torch.nn.functional.avg_pool2d(flat, 5, stride=1, padding=2)
                    eps = eps - high_pass * blur.reshape(b, c, t, h, w)
                elif eps.dim() == 4:
                    b, c, h, w = eps.shape
                    flat = eps.reshape(b * c, 1, h, w)
                    blur = torch.nn.functional.avg_pool2d(flat, 5, stride=1, padding=2)
                    eps = eps - high_pass * blur.reshape(b, c, h, w)

            scale = strength * denoised.std()
            return denoised + scale * eps

        # disable_cfg1_optimization=True so the hook still fires when cfg==1
        # (Wan distill / guidance-embed workflows often run cfg 1).
        m.set_model_sampler_post_cfg_function(post_cfg, disable_cfg1_optimization=True)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "WanBoilNoise": WanBoilNoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanBoilNoise": "Wan Boil Noise (late-step per-frame)",
}
