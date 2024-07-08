import gradio as gr
from huggingface_hub import hf_hub_download
import yolov9


# Load the model
model_path = r'./model/V2_best.pt'
model = yolov9.load(model_path)

def yolov9_inference(img_path, conf_threshold=0.4, iou_threshold=0.5):
    """   
    :param conf_threshold: Confidence threshold for NMS.
    :param iou_threshold: IoU threshold for NMS.
    :param img_path: Path to the image file.
    :param size: Optional, input size for inference.
    :return: A tuple containing the detections (boxes, scores, categories) and the results object for further actions like displaying.
    """
    global model
    # Set model parameters
    model.conf = conf_threshold
    model.iou = iou_threshold
    
    # Perform inference
    results = model(img_path, size=640)

    # Optionally, show detection bounding boxes on image
    output = results.render()
    
    return output[0]


def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                img_path = gr.Image(type="filepath", label="Image")
                # conf_threshold = gr.Slider(
                #     label="Confidence Threshold",
                #     minimum=0.1,
                #     maximum=1.0,
                #     step=0.1,
                #     value=0.4,
                # )
                # iou_threshold = gr.Slider(
                #     label="IoU Threshold",
                #     minimum=0.1,
                #     maximum=1.0,
                #     step=0.1,
                #     value=0.5,
                # )
                yolov9_infer = gr.Button(value="Prediction")

            with gr.Column():
                output_numpy = gr.Image(type="numpy",label="Output")

        yolov9_infer.click(
            fn=yolov9_inference,
            inputs=[
                img_path,
            ],
            outputs=[output_numpy],
        )
        

gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    Traffic Signs Detection - Case Study
    </h1>
    """)
    with gr.Row():
        with gr.Column():
            app()

# gradio_app.launch(debug=True)

# gradio_app.launch(server_name="0.0.0.0", server_port=8080, debug=True)


gradio_app.launch(
    server_name="0.0.0.0",
    server_port=8080,
    # ssl_keyfile="./key.pem",
    # ssl_certfile="./cert.pem",
    debug=True
)
