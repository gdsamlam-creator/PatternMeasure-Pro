import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from core.processing import process_pattern_image, draw_measurements
from core.measurement import calculate_area
from core.segmentation import Segmentor
from core.measurement import calibrate_scale, measure_edges
import tempfile
import os

segmentor = None

def get_segmentor():
    global segmentor
    if segmentor is None:
        segmentor = Segmentor()
    return segmentor

def process_image(image, scale_value, scale_unit, calibration_line_length):
    if image is None:
        return None, "Please upload an image"

    image = np.array(image)
    segmentor = get_segmentor()
    mask = segmentor.segment_pattern(image)
    processed_image, edge_points = process_pattern_image(image, mask)

    if len(edge_points) < 3:
        return processed_image, "Could not detect pattern edges"

    real_world_scale = (float(scale_value) / float(calibration_line_length), scale_unit)
    annotated_image = draw_measurements(processed_image.copy(), edge_points, real_world_scale)

    area, area_unit = calculate_area(edge_points, real_world_scale)
    area_text = f"Area: {area:.2f} {area_unit}²"

    return annotated_image, area_text

def main():
    with gr.Blocks(title="PatternMeasure Pro") as app:
        gr.Markdown("# PatternMeasure Pro")
        gr.Markdown("Capture and measure paper patterns with computer vision")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Upload Pattern Image")

                with gr.Accordion("Calibration", open=True):
                    scale_value = gr.Number(value=10.0, label="Real-world distance (e.g., 10)", precision=2)
                    scale_unit = gr.Dropdown(["cm", "inches", "mm"], value="cm", label="Unit")
                    calibration_line_length = gr.Number(value=100.0, label="Line length in pixels (measure in image)", precision=1)

                process_btn = gr.Button("Analyze Pattern", variant="primary")

            with gr.Column():
                output_image = gr.Image(type="pil", label="Analyzed Pattern")
                area_output = gr.Textbox(label="Calculated Area", interactive=False)

        process_btn.click(
            process_image,
            inputs=[input_image, scale_value, scale_unit, calibration_line_length],
            outputs=[output_image, area_output]
        )

    app.launch()

if __name__ == "__main__":
    main()
