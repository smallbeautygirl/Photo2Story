# Style reference images

Drop one reference image per registered style here, named `<style_key>.png`:

| File | Style | Dedicated checkpoint? |
| --- | --- | --- |
| `ghibli.png` | Studio Ghibli / Miyazaki | ✅ `nitrosocke/Ghibli-Diffusion` |
| `pixar.png` | 3D Pixar | ✅ `nitrosocke/mo-di-diffusion` |
| `disney.png` | Classic Disney | — (text + IP-Adapter) |
| `crayon.png` | Crayon drawing | — (text + IP-Adapter) |

Stage 2 feeds the matching image into **IP-Adapter** so every page keeps one
master's look. A missing file degrades gracefully to a text-only style prompt
(logged as a warning) — it does not crash the run.

Styles with a **dedicated checkpoint** (or `lora`) in
[`src/utils/styles.py`](../../src/utils/styles.py) stylise from the prompt alone
and do **not** need a reference image here. For the others, a reference image is
the best way to strengthen the look until a LoRA is assigned.

An explicit `stage2.style_image_path` in the config overrides the preset image.

Style keys are defined in [`src/utils/styles.py`](../../src/utils/styles.py).
