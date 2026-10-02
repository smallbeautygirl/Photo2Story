# Use fal.ai-hosted FLUX.1-dev as the primary Stage 2 backend, not local SD 1.5

Status: accepted

Local hardware does not have enough VRAM to run a larger, higher-quality image model; the local Stable Diffusion 1.5 path that fits on the available GPU produced visible facial artifacts (e.g. the nose blurring out or disappearing) in generated illustrations. Stage 2 therefore defaults demo/experiment runs to `stage2.backend = "fal"`, which calls FLUX.1-dev through fal.ai's hosted API, giving access to a stronger model without needing to own the GPU it requires. The local SD 1.5 path (`stage2.backend = "local"`) is kept rather than removed, since it still runs fully offline and remains useful where avoiding a third-party API call matters more than output quality (e.g. large-volume ablation runs under cost or connectivity constraints).
