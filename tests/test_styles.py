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
