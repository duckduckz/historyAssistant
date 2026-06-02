from __future__ import annotations

import gradio as gr

from historyAssistant.app.export import EXPORT_DIR
from historyAssistant.app.gradio_ui import build_demo


demo = build_demo()


def main() -> None:
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        theme=gr.themes.Soft(),
        allowed_paths=[str(EXPORT_DIR)],
    )


if __name__ == "__main__":
    main()
