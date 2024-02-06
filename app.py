import gradio as gr
from utils import AppUtils
from app_inference import AppInference

inference = AppInference()

def process_image(input_id, image, label):
    return inference.inference(int(input_id), AppUtils.get_examples()[int(input_id)][1], image["mask"], label)

def preview(input_id, image, label):
    return inference.preview(int(input_id), AppUtils.get_examples()[int(input_id)][1], image["mask"], label)

def update_label_dropdown(input_id):
    choices = AppUtils.get_labels(int(input_id))
    return gr.Dropdown.update(choices=choices, value=choices[0])

with gr.Blocks() as demo:
    gr.Markdown(
    """
    <h1 align="center">Diverse Semantic Image Editing with Style Codes</h1>
    <center>
    <div> <a href="https://www.cs.bilkent.edu.tr/~adundar/projects/DivSem/">Website</a></div> 
   </center>
    <center> In this work, we propose a novel framework that can encode visible and partially visible objects with a novel mechanism to achieve consistency in the style encoding. 
            Here, we show our results for different images and label editing options. </center>
    <center> <h2> How to Try </h2> </center>
    <center> 1. Select an image from the example list at the bottom. </center>
    <center> 2. Draw a mask on the input image. </center>
    <center> 3. Select a label from the dropdown on the "Choose Label" section. This label is used </center>
    <center> for editing masked area. If you don't want to change label of the masked area, you can choose None. </center>
    <center> 4. Click to Preview button to see the edited instance map. </center>
    <center> 5. Click to Submit button to see the inference result. </center>
    <center> Note: Our demo currently does not support to get inference from an uploaded image. Please use example images. </center>
    """)
    with gr.Row():
        image_input = gr.Image(type="pil", shape=(256,256), label='Input', tool="sketch", value=AppUtils.get_examples()[0][1], scale=5).style(height=256)
        inst_map_output = gr.Image(type="pil", shape=(256,256), label='Instance Map', value=AppUtils.get_examples()[0][1].replace("images", "colored"), scale=4).style(height=256)
        image_output = gr.Image(type="pil", shape=(256,256), label='Output Image',scale=4).style(height=256)

    with gr.Row():
        input_id = gr.Textbox(label="Image ID", value=AppUtils.get_examples()[0][0], interactive=False, visible=False)
        with gr.Column(scale=1, min_width=50):
            label_dropdown = gr.Dropdown(AppUtils.get_labels(0), label="Choose Label", value=AppUtils.get_labels(0)[0])
        with gr.Column(scale=2, min_width=50):
            with gr.Row():
                preview_button = gr.Button(value="Preview")
            with gr.Row():
                submit_button = gr.Button(value="Submit")

    gr.Examples(
        examples=AppUtils.get_examples(),
        inputs=[input_id, image_input, inst_map_output],
        outputs=[image_output],
        fn=process_image,
    )
    input_id.change(update_label_dropdown, inputs=input_id, outputs=label_dropdown )
    submit_button.click(process_image, inputs=[input_id, image_input, label_dropdown], outputs=image_output)
    preview_button.click(preview, inputs=[input_id, image_input, label_dropdown], outputs=[inst_map_output])

demo.launch()