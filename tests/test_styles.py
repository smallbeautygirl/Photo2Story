from src.utils.styles import StyleKey, available_styles, resolve_style


def test_resolve_style_returns_registered_preset():
    preset = resolve_style("ghibli")
    assert preset.key == StyleKey.GHIBLI
    assert "Ghibli" in preset.label
    assert preset.sd_prompt


def test_resolve_style_is_case_and_separator_insensitive():
    assert resolve_style("Ink-Wash").key == StyleKey.INK_WASH
    assert resolve_style("FLAT PASTEL").key == StyleKey.FLAT_PASTEL


def test_resolve_style_falls_back_to_passthrough_for_unknown():
    preset = resolve_style("steampunk")
    assert preset.label == "steampunk"
    assert "steampunk" in preset.sd_prompt
    assert preset.style_image is None
    assert preset.model is None


def test_ghibli_preset_binds_a_stylised_checkpoint():
    preset = resolve_style("ghibli")
    assert preset.model == "nitrosocke/Ghibli-Diffusion"
    # The checkpoint's trigger token must be present for the style to activate.
    assert "ghibli style" in preset.sd_prompt


def test_available_styles_lists_all_registered_keys():
    assert set(available_styles()) == {s.value for s in StyleKey}


def test_ghibli_preset_binds_a_flux_lora():
    preset = resolve_style("ghibli")
    assert preset.flux_lora == "openfree/flux-chatgpt-ghibli-lora"
    assert preset.flux_prompt


def test_pixar_preset_binds_a_flux_lora():
    preset = resolve_style("pixar")
    assert preset.flux_lora == "prithivMLmods/Canopus-Pixar-3D-Flux-LoRA"
    # The LoRA's trigger token must lead the FLUX prompt.
    assert preset.flux_prompt.startswith("Pixar 3D")


def test_disney_preset_binds_a_flux_lora_with_suffix():
    preset = resolve_style("disney")
    assert preset.flux_lora == "tubbymeatball/DisneyStyleLora"
    # This LoRA needs trigger text both before and after the scene.
    assert preset.flux_prompt
    assert preset.flux_prompt_suffix


def test_crayon_has_a_flux_prompt_but_no_lora():
    preset = resolve_style("crayon")
    assert preset.flux_prompt
    assert preset.flux_lora is None


def test_passthrough_preset_has_no_flux_fields():
    preset = resolve_style("steampunk")
    assert preset.flux_prompt is None
    assert preset.flux_prompt_suffix is None
    assert preset.flux_lora is None
