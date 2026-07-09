# TripoSplat add-ons for ComfyWarp:
#   1. VAEDecodeTripoSplatHighCount - lift the 1M gaussian cap and tile the gaussian
#      decoder over point chunks so very high counts fit in VRAM.
#   2. TripoSplatExpandLatent - grow the sampling latent past the trained 8192 tokens
#      (e.g. 2x -> 16384) so the final shape-code carries more detail.
#   3. TripoSplatTokenMultiscale - run that larger latent coarse-to-fine: the first part of
#      sampling at the trained 8192 tokens (locks global structure, in-distribution), then
#      the late steps split the larger token set into overlapping 8192-token tiles, each of
#      which is still in-distribution, to add detail (MODEL -> MODEL patch, sampler-agnostic).
#
# Heavy imports from comfy.ldm.triposplat are done lazily inside the node bodies so this
# module still registers if a given ComfyUI build does not ship TripoSplat.

import torch


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Octree point ordering is hierarchical (children of a node are contiguous), so a
# contiguous slice of the point sequence is a spatially-clustered group. That makes the
# per-chunk self-attention in the gaussian decoder a good approximation of the full pass:
# spatially-near points stay in the same chunk.

_NUM_GAUSSIANS_MIN = 32768
# Original comfy-core node caps at 1048576; we lift it. This ceiling is just a UI guard.
_NUM_GAUSSIANS_MAX_HIGH = 1 << 25  # 33,554,432


def _sobol_positions(n, device="cpu"):
    """Reconstruct the latent tokens' fixed 3D anchor positions, matching
    LatentSeqMMFlowModel.__init__ (SobolEngine dim=3, scramble=True, seed=123)."""
    return torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=123).draw(n).to(device)


# ---------------------------------------------------------------------------
# 1. High-count tiled decode
# ---------------------------------------------------------------------------

class VAEDecodeTripoSplatHighCount:
    """Decode a sampled TripoSplat latent into a 3D gaussian splat, allowing far more
    than the stock 1,048,576-gaussian cap by tiling the gaussian decoder over point
    chunks (bounded VRAM).

    The octree sampler picks `num_gaussians // 32` points in one cheap pass, then the
    elastic-gaussian decoder (16 self+cross-attention blocks - the VRAM bottleneck) runs
    over those points. We split the points into contiguous chunks and decode each chunk
    separately, then concatenate the resulting gaussians. Because the point order is
    octree-hierarchical, a contiguous chunk is a spatial cluster, so breaking cross-chunk
    self-attention is a mild approximation. Set gaussians_per_chunk=0 for an exact
    (un-tiled) decode.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",),
            "vae": ("VAE",),
            "num_gaussians": ("INT", {
                "default": 1048576, "min": _NUM_GAUSSIANS_MIN, "max": _NUM_GAUSSIANS_MAX_HIGH, "step": 32,
                "tooltip": "Number of gaussians to produce (rounded to a multiple of 32). The stock node "
                           "caps at 1048576; this node lifts that. 262144 matches the octree's point density; "
                           "higher oversamples the same points (denser, no new detail) and costs proportionally "
                           "more VRAM/time.",
            }),
            "gaussians_per_chunk": ("INT", {
                "default": 1048576, "min": 0, "max": _NUM_GAUSSIANS_MAX_HIGH, "step": 32,
                "tooltip": "Tile the gaussian decoder into chunks of at most this many gaussians to bound peak "
                           "VRAM (rounded to a multiple of 32). Lower = less VRAM, more passes. 0 = no tiling "
                           "(exact, but needs enough VRAM for the full count at once).",
            }),
            "seed": ("INT", {
                "default": 0, "min": 0, "max": 0xffffffffffffffff,
                "control_after_generate": True,
                "tooltip": "Seeds the octree point sampler (CPU generator) for deterministic decodes.",
            }),
        }}

    RETURN_TYPES = ("SPLAT",)
    FUNCTION = "decode"
    CATEGORY = "3d/latent"

    def decode(self, samples, vae, num_gaussians, gaussians_per_chunk, seed):
        import comfy.model_management
        from comfy.ldm.triposplat.vae import OctreeProbabilityFixedlenDecoder
        from comfy.ldm.triposplat.gaussian import build_gaussian_models
        from comfy_api.latest import Types

        s = samples["samples"]
        latent = s.unbind()[0] if getattr(s, "is_nested", False) else s  # drop the camera stream

        decoder = vae.first_stage_model
        gpp = decoder.gaussians_per_point
        n = max(_NUM_GAUSSIANS_MIN, min(_NUM_GAUSSIANS_MAX_HIGH, int(num_gaussians)))
        if n % gpp != 0:
            n = round(n / gpp) * gpp
        num_points = max(1, n // gpp)

        # chunk size in points; 0 (or >= total) means a single un-tiled pass
        if gaussians_per_chunk and gaussians_per_chunk > 0:
            points_per_chunk = max(1, int(gaussians_per_chunk) // gpp)
        else:
            points_per_chunk = num_points
        points_per_chunk = min(points_per_chunk, num_points)

        # Peak VRAM is driven by the largest single decode pass, i.e. one chunk.
        dtype_size = comfy.model_management.dtype_size(vae.vae_dtype)
        hidden = decoder.gs.model_channels
        cond_tokens = latent.shape[1]
        memory_required = (cond_tokens * 4 + points_per_chunk * 10) * hidden * dtype_size
        comfy.model_management.load_models_gpu([vae.patcher], memory_required=memory_required)

        latent = latent.to(device=vae.device, dtype=vae.vae_dtype)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

        # Octree sampling is cheap (coords + small per-level logits); do all points at once.
        points_pred = OctreeProbabilityFixedlenDecoder.sample(
            decoder.octree, latent, num_points=num_points,
            level=decoder._MAX_VOXEL_LEVEL, temperature=1.0, generator=generator,
        )
        pts = points_pred["points"]            # (B, num_points, 3)
        log_probs = points_pred["log_probs"]   # (B, num_points)
        batch = pts.shape[0]

        # Decode point chunks through the gaussian decoder, render, and concatenate.
        per_batch_chunks = [[] for _ in range(batch)]
        for start in range(0, num_points, points_per_chunk):
            stop = min(start + points_per_chunk, num_points)
            sub = {"points": pts[:, start:stop], "log_probs": log_probs[:, start:stop]}
            pred = decoder.gs(x=sub, cond=latent)
            for bi, g in enumerate(build_gaussian_models(decoder.gs, sub, pred)):
                per_batch_chunks[bi].append(g.render_tensors())

        merged = []
        for chunks in per_batch_chunks:
            merged.append(tuple(torch.cat([c[k] for c in chunks], dim=0) for k in range(5)))
        positions, scales, rotations, opacities, sh = (torch.stack(t) for t in zip(*merged))
        return (Types.SPLAT(positions, scales, rotations, opacities, sh),)


# ---------------------------------------------------------------------------
# 2. Experimental token-count multiscale sampler patch
# ---------------------------------------------------------------------------

class TripoSplatExpandLatent:
    """Grow a TripoSplat sampling latent past the model's trained 8192 tokens so the final
    shape-code carries more detail.

    TripoSplatConditioning emits a fixed (B, 8192, 16) noise-target plus a camera token. The
    sampler derives noise and all shapes from that latent, and the decoder cross-attends to
    however many tokens it has - so more tokens means a finer shape code. This node replaces
    the token axis with `token_multiplier x 8192` tokens (a longer prefix of the SAME Sobol
    anchor sequence, so the first 8192 stay identical), filling the extra slots by
    nearest-anchor copy. Pair it with TripoSplatTokenMultiscale so the diffusion model only
    ever sees 8192 tokens at a time (in-distribution).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",),
            "token_multiplier": ("INT", {
                "default": 2, "min": 1, "max": 8, "step": 1,
                "tooltip": "Multiply the latent token count by this factor (1 = unchanged, 2 = 16384 tokens). "
                           "More tokens = denser shape code = finer splat, at higher sampling cost.",
            }),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "expand"
    CATEGORY = "3d/latent"

    def expand(self, samples, token_multiplier):
        from comfy.nested_tensor import NestedTensor

        s = samples["samples"]
        mult = int(token_multiplier)
        if mult <= 1 or not getattr(s, "is_nested", False):
            return (samples,)

        latent_seq, camera = s.unbind()  # (B, L, C), (B, 1, cam)
        B, L, C = latent_seq.shape
        N = L * mult

        # Map each of the N new anchors to the nearest of the original L anchors and copy.
        # For a zero noise-target this is a no-op on values; it also stays correct if a
        # non-zero (partially sampled) latent is fed in.
        pos = _sobol_positions(N)                              # (N, 3)
        idx = torch.cdist(pos, pos[:L]).argmin(dim=1)          # (N,) nearest original anchor
        new_seq = latent_seq[:, idx.to(latent_seq.device)].contiguous()

        out = dict(samples)
        out["samples"] = NestedTensor((new_seq, camera))
        return (out,)


class TripoSplatTokenMultiscale:
    """Coarse-to-fine, in-distribution token tiling for the TripoSplat KSampler
    (MODEL -> MODEL patch, sampler-agnostic). Use with TripoSplatExpandLatent.

    The model is trained on 8192 tokens. When the carried latent has more (e.g. 16384 from a
    2x expand), feeding all of them at once is off-distribution. Instead:

      * Early steps (frac < coarse_until_percent): run only the trained 8192-token prefix
        (identical to vanilla sampling - Sobol(N)[:8192] == Sobol(8192)) and densify the
        result to all N tokens by nearest anchor. This locks global structure.
      * Late steps: split the N tokens into overlapping 8192-token tiles (each a balanced
        Sobol block, so each tile is in-distribution), run the model per tile, and average
        the overlaps back into the full N-token prediction. This adds detail without letting
        structure drift.

    Off-distribution effects at higher token density are still possible; treat as an
    experiment. A no-op when the latent is <= 8192 tokens (no expand) - pass-through.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "coarse_until_percent": ("FLOAT", {
                "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "Run the trained 8192-token prefix until this fraction through sampling (structure "
                           "phase), then switch to tiled full-token detail. 0.5 = first half coarse.",
            }),
            "overlap": ("FLOAT", {
                "default": 0.25, "min": 0.0, "max": 0.75, "step": 0.05,
                "tooltip": "Fraction of overlap between adjacent 8192-token tiles in the detail phase. Overlap "
                           "tiles are averaged, smoothing the boundary between independently-denoised tiles.",
            }),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "3d/latent"

    def patch(self, model, coarse_until_percent, overlap):
        import comfy.patcher_extension
        from comfy.ldm.triposplat.model import PcdAbsolutePositionEmbedder

        cache = {}  # N -> dict(pos_emb=(1,N,C) float, dense_idx=(N,) long)

        def _assets(N, base, model_channels):
            a = cache.get(N)
            if a is None:
                pos = _sobol_positions(N)                                  # (N, 3) cpu
                pos_emb = PcdAbsolutePositionEmbedder(model_channels)(pos.unsqueeze(0))  # (1, N, C)
                dense_idx = torch.cdist(pos, pos[:base]).argmin(dim=1)     # (N,) nearest prefix anchor
                a = {"pos_emb": pos_emb, "dense_idx": dense_idx}
                cache[N] = a
            return a

        def _progress_frac(transformer_options):
            ss = transformer_options.get("sample_sigmas", None)
            cur = transformer_options.get("sigmas", None)
            if ss is None or cur is None or len(ss) < 2:
                return None
            sig = float(cur.flatten()[0])
            steps = ss[:-1].to(torch.float32).flatten()  # starting sigma of each step (descending)
            i = int((steps - sig).abs().argmin())
            return i / max(1, steps.numel() - 1)

        def _windows(N, base):
            if N <= base:
                return [(0, N)]
            stride = max(1, int(round(base * (1.0 - float(overlap)))))
            starts = list(range(0, N - base + 1, stride))
            if starts[-1] != N - base:
                starts.append(N - base)
            return [(st, st + base) for st in starts]

        def diffusion_wrapper(executor, x, t, context=None, ref_latents=None, transformer_options={}, **kwargs):
            z, camera = x[0], x[1]
            B, N, C = z.shape
            dm = executor.class_obj  # LatentSeqMMFlowModel
            base = int(getattr(dm, "q_token_length", N))
            if N <= base:
                return executor(x, t, context, ref_latents, transformer_options, **kwargs)

            assets = _assets(N, base, int(dm.model_channels))
            pos_emb_full = assets["pos_emb"]

            def run_subset(start, stop):
                old_pos = dm.pos_emb
                dm.pos_emb = pos_emb_full[:, start:stop]
                try:
                    return executor([z[:, start:stop].contiguous(), camera], t, context,
                                    ref_latents, transformer_options, **kwargs)
                finally:
                    dm.pos_emb = old_pos

            frac = _progress_frac(transformer_options)
            if frac is None or frac < coarse_until_percent:
                # Structure phase: trained 8192-token prefix, densified to N by nearest anchor.
                out_z, out_cam = run_subset(0, base)
                full = out_z[:, assets["dense_idx"].to(out_z.device)].contiguous()
                return full, out_cam

            # Detail phase: overlapping in-distribution tiles, averaged over overlaps.
            accum = torch.zeros(B, N, C, dtype=torch.float32, device=z.device)
            wcount = torch.zeros(N, dtype=torch.float32, device=z.device)
            out_cam = None
            for st, sp in _windows(N, base):
                tile_z, tile_cam = run_subset(st, sp)
                accum[:, st:sp] += tile_z.float()
                wcount[st:sp] += 1.0
                if out_cam is None:
                    out_cam = tile_cam
            full = (accum / wcount.clamp_min(1.0).view(1, N, 1)).to(z.dtype)
            return full, out_cam

        m = model.clone()
        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "triposplat_token_multiscale",
            diffusion_wrapper,
        )
        return (m,)


NODE_CLASS_MAPPINGS = {
    "VAEDecodeTripoSplatHighCount": VAEDecodeTripoSplatHighCount,
    "TripoSplatExpandLatent": TripoSplatExpandLatent,
    "TripoSplatTokenMultiscale": TripoSplatTokenMultiscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VAEDecodeTripoSplatHighCount": "TripoSplat Decode (high-count, tiled)",
    "TripoSplatExpandLatent": "TripoSplat Expand Latent (>8192 tokens)",
    "TripoSplatTokenMultiscale": "TripoSplat Token Multiscale (tiled)",
}
