"""Exp21 BF16 图文检索推理 Demo
=============================
启动方式:
    python app_inference.py

访问: http://localhost:7861
"""
import logging

import gradio as gr
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("demo")

from siglip.inference.config import SERVER_NAME, SERVER_PORT
from siglip.inference.model_loader import InferenceEngine

print("=" * 60)
print("  Exp21 图文检索推理 Demo — 正在初始化...")
print("=" * 60)

ENGINE = InferenceEngine()

precision_tag = "BF16" if ENGINE.use_bf16 else "FP32"
print("=" * 60)
print(f"  模型: Exp21 ({precision_tag}) | 体积: {ENGINE.model_size_mb:.1f} MB")
print(f"  数据库: {len(ENGINE.flickr_data)} 张图像")
print(f"  访问 http://localhost:{SERVER_PORT}")
print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
#  Callbacks
# ═══════════════════════════════════════════════════════════════════════════════

def cb_text_to_image(query_text, top_k):
    if not query_text.strip():
        return [], "请输入文本描述"
    top_k = int(top_k)
    results = ENGINE.search_text_to_image(query_text, top_k)
    gallery = [(Image.open(r[0]).convert("RGB"), f"Score: {r[2]:.4f} | {r[1][:60]}")
               for r in results]
    info = f"查询: \"{query_text}\" | Top-{top_k} | {precision_tag} 推理 | 模型 {ENGINE.model_size_mb:.1f} MB"
    return gallery, info


def cb_image_to_text(query_image, top_k):
    if query_image is None:
        return [], "请上传图片"
    top_k = int(top_k)
    results = ENGINE.search_image_to_text(query_image, top_k)
    rows = [(i+1, r[0], f"{r[1]:.4f}") for i, r in enumerate(results)]
    info = f"Image → Text Top-{top_k} | {precision_tag} 推理"
    return rows, info


# ═══════════════════════════════════════════════════════════════════════════════
#  CSS & Banner
# ═══════════════════════════════════════════════════════════════════════════════

_CSS = """
.gradio-container { background: #1a1f2e !important; min-height: 100vh; }
.gradio-container, .gradio-container p, .gradio-container span,
.gradio-container div, .gradio-container label,
.gradio-container h1, .gradio-container h2, .gradio-container h3 {
    color: #e8ecf0 !important;
}
#banner {
    background: linear-gradient(90deg, #1a2744 0%, #1e3a5f 50%, #1a2744 100%);
    border: 1px solid #2d4a7a; border-radius: 12px;
    padding: 28px 36px; text-align: center; margin-bottom: 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,.4);
}
#banner h1 {
    font-size: 2em !important; color: #79b8ff !important;
    margin: 0 0 8px !important; text-shadow: 0 0 20px rgba(121,184,255,.4);
}
#banner p  { color: #b0bec5 !important; margin: 0 !important; font-size: 14px !important; }
#banner .badge {
    display: inline-block; background: #263650; border: 1px solid #3a5278;
    border-radius: 20px; padding: 3px 12px;
    font-size: 12px; color: #90caf9 !important; margin: 6px 3px 0;
}
button.primary, button[variant="primary"] {
    background: linear-gradient(135deg, #1e5f9a, #2980b9) !important;
    border: none !important; color: #ffffff !important; font-weight: 600 !important;
    border-radius: 8px !important; box-shadow: 0 3px 12px rgba(41,128,185,.4) !important;
}
.gr-panel, .gr-box, .gr-form, .gr-block {
    background: #1e2536 !important; border: 1px solid #2d3d5e !important;
    border-radius: 10px !important;
}
"""

_BANNER = f"""
<div id="banner">
    <h1>Exp21 图文检索推理 Demo</h1>
    <p>基于 SigLIP ViT-B/16 + DistilBERT 部分解冻微调 &nbsp;|&nbsp; Flickr8k</p>
    <div style="margin-top:10px">
        <span class="badge">{precision_tag}: {ENGINE.model_size_mb:.1f} MB</span>
        <span class="badge">t2i R@1=58.03%</span>
        <span class="badge">预计算全库 embedding</span>
        <span class="badge">检索 &lt;50ms</span>
    </div>
</div>
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  Build UI
# ═══════════════════════════════════════════════════════════════════════════════

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
    text_size=gr.themes.sizes.text_md,
).set(
    body_background_fill="#1a1f2e",
    body_text_color="#e8ecf0",
    block_background_fill="#1e2536",
    block_label_text_color="#90caf9",
    block_title_text_color="#79b8ff",
    input_background_fill="#263650",
    input_border_color="#3a5278",
    button_primary_background_fill="linear-gradient(135deg,#1e5f9a,#2980b9)",
    button_primary_text_color="#ffffff",
    slider_color="#4a90d9",
)

with gr.Blocks(theme=theme, css=_CSS, title="Exp21 图文检索推理 Demo") as demo:

    gr.HTML(_BANNER)

    # ── Tab 1: Text → Image ──────────────────────────────────────────────
    with gr.Tab("文本搜图 (Text → Image)"):
        gr.Markdown("输入任意文本描述，从 Flickr8k 中检索最相关的图像。")
        with gr.Row():
            query_text = gr.Textbox(
                label="查询文本", placeholder="e.g., a dog running on a beach",
                lines=1,
            )
            top_k1 = gr.Slider(1, 20, value=5, step=1, label="Top-K")
        search_btn1 = gr.Button("检索", variant="primary")
        info1 = gr.Markdown()
        gallery1 = gr.Gallery(columns=5, label="检索结果")

        search_btn1.click(cb_text_to_image, inputs=[query_text, top_k1],
                          outputs=[gallery1, info1])

    # ── Tab 2: Image → Text ──────────────────────────────────────────────
    with gr.Tab("图片搜文 (Image → Text)"):
        gr.Markdown("上传图片，检索最匹配的文本描述。")
        with gr.Row():
            query_image = gr.Image(type="pil", label="查询图片")
            top_k2 = gr.Slider(1, 20, value=5, step=1, label="Top-K")
        search_btn2 = gr.Button("检索", variant="primary")
        info2 = gr.Markdown()
        table2 = gr.Dataframe(
            headers=["Rank", "Caption", "Cosine Score"],
            label="检索结果",
        )

        search_btn2.click(cb_image_to_text, inputs=[query_image, top_k2],
                          outputs=[table2, info2])

    gr.HTML("""
    <div style="background:#1c2333;border-radius:8px;padding:14px 18px;
                border-left:4px solid #2ecc71;font-size:13px;color:#c9d1d9;margin-top:8px">
        <b>部署说明</b>：模型使用 BF16 半精度推理（体积 305 MB，vs FP32 610 MB），
        启动时预计算全库 embedding，检索仅需一次 dot product（< 50ms）。
        BF16 编码延迟 2.01 ms/sample（vs FP32 7.68 ms/sample），加速 3.82×，
        t2i R@1 仅降 0.02pp（58.03% → 58.01%）。
    </div>
    """)


if __name__ == "__main__":
    demo.queue(max_size=4).launch(
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
        share=False,
        show_error=True,
        inbrowser=True,
    )