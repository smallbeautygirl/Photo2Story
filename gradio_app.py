from __future__ import annotations
import shutil
import tempfile
from pathlib import Path

import gradio as gr

from src.config import load_config
from src.pipeline import StoryPipeline

STYLES = ["watercolor", "anime", "western cartoon", "pencil sketch"]
CONFIG_PATH = "configs/demo.yaml"


def run_pipeline(images: list, context: str, style: str, k: int) -> tuple[str | None, str]:
    if not images or len(images) < 3:
        return None, "請上傳至少 3 張照片"

    image_paths = [img.name for img in images]
    config = load_config(CONFIG_PATH)
    config["stage0"]["k"] = int(k)

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = StoryPipeline(config)
        result = pipeline.run(
            image_paths=image_paths,
            context=context,
            style=style,
            output_dir=tmpdir,
        )
        out_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        shutil.copy(result["pdf_path"], out_pdf.name)

    story_preview = "\n\n".join(
        f"**第 {i + 1} 頁：** {p}" for i, p in enumerate(result["pages"])
    )
    return out_pdf.name, story_preview


with gr.Blocks(title="Photo2Story 繪本生成器") as demo:
    gr.Markdown("# 📖 Photo2Story\n上傳照片，自動生成個人化繪本")

    with gr.Row():
        with gr.Column(scale=1):
            photos = gr.File(
                label="上傳照片（3–10 張）",
                file_count="multiple",
                file_types=["image"],
            )
            context_box = gr.Textbox(
                label="旅遊情境描述（可選）",
                placeholder="例如：北海道六天五夜，男方爸媽和兩個兒子",
                lines=2,
            )
            style_box = gr.Dropdown(
                choices=STYLES,
                value="watercolor",
                label="繪本風格",
            )
            k_slider = gr.Slider(
                minimum=3,
                maximum=6,
                value=4,
                step=1,
                label="繪本頁數",
            )
            run_btn = gr.Button("✨ 生成繪本", variant="primary")

        with gr.Column(scale=1):
            pdf_output = gr.File(label="下載 PDF 繪本")
            story_output = gr.Markdown(label="故事預覽")

    run_btn.click(
        fn=run_pipeline,
        inputs=[photos, context_box, style_box, k_slider],
        outputs=[pdf_output, story_output],
    )

if __name__ == "__main__":
    demo.launch(share=False)
