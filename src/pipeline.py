from __future__ import annotations
from pathlib import Path
from src.config import PipelineConfig
from src.stages.stage0_select import run_stage0
from src.stages.stage1_story import run_stage1
from src.stages.stage2_illustrate import run_stage2
from src.stages.stage3_assemble import run_stage3


class StoryPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(
        self,
        image_paths: list[str],
        context: str,
        style: str,
        output_dir: str,
    ) -> dict:
        """
        Run the full pipeline. Returns:
            {
                "pdf_path": str,
                "pages": list[str],
                "narrative": str,
                "selected_paths": list[str],
            }
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        stage0 = run_stage0(image_paths, context, self.config["stage0"])
        stage1_context = stage0.get("effective_context") or context
        stage1 = run_stage1(stage0, stage1_context, style, self.config["stage1"])
        stage2 = run_stage2(stage1, style, self.config["stage2"], str(out / "illustrations"))
        stage3 = run_stage3(stage1, stage2, self.config["stage3"], str(out / "storybook.pdf"))

        return {
            "pdf_path": stage3["pdf_path"],
            "pages": stage1["pages"],
            "narrative": stage1["narrative"],
            "selected_paths": stage0["selected_paths"],
            "ordered_paths": stage0["ordered_paths"],
            "descriptions": stage0["descriptions"],
            "effective_context": stage0.get("effective_context", context),
            "illustration_paths": stage2["illustration_paths"],
        }
