import gradio as gr
from inference_v6 import RIFE

model_dir = "train_log"
output_base_path = "Z:\\AI_workspace\\video_interpolation_backup"
rife = RIFE(model_dir, output_base_path)

def process(video, extension, output_fps, scale):
    return rife.run(video, extension, output_fps, scale)

with gr.Blocks() as demo:
    gr.HTML("<h1 style='margin: 0; text-align: center;'>RIFE Demo</h1>")

    video = gr.Video(label="Video File", sources=["upload"], visible=True)
    extension = gr.Dropdown(choices=["mxf", "mp4"], label="Extension")
    output_fps = gr.Number(label="Output FPS", precision=2)
    scale = gr.Radio(choices=[1.0, 0.5, 0.25], label="Scale")
    save_img = gr.Checkbox(label="Save Image")

    submit = gr.Button("Run")
    output_text = gr.Textbox(label="Output")
    submit.click(process, inputs=[video, extension, output_fps, scale], outputs=output_text)

demo.launch()
