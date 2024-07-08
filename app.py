import os
import requests
import gradio as gr
import yolov9

# Define the URL and model path
model_url = "https://sadakatcdn.cyparta.com/V2_best.pt"
model_path = "./model/V2_best.pt"

# Function to download the model if it doesn't exist
def download_model(url, save_path):
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Downloading model from {url}...")
        response = requests.get(url)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully.")
    else:
        print("Model already exists.")

# Download the model
download_model(model_url, model_path)

# Load the model
model = yolov9.load(model_path)

def yolov9_inference(img_path, conf_threshold=0.4, iou_threshold=0.5):
    """
    :param conf_threshold: Confidence threshold for NMS.
    :param iou_threshold: IoU threshold for NMS.
    :param img_path: Path to the image file.
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
    with gr.Row():
        with gr.Column():
            img_path = gr.Image(type="filepath", label="Image")
            yolov9_infer = gr.Button(value="Prediction")
        with gr.Column():
            output_numpy = gr.Image(type="numpy", label="Output")

        yolov9_infer.click(
            fn=yolov9_inference,
            inputs=[img_path],
            outputs=[output_numpy],
        )

gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
        <h1 style='text-align: center'>
        Traffic Signs Detection - Case Study
        </h1>
        """
    )
    with gr.Row():
        with gr.Column():
            app()

gradio_app.launch(
    server_name="0.0.0.0",
    server_port=8080,
    ssl_keyfile="./key.pem",
    ssl_certfile="./cert.pem",
    debug=True
)
