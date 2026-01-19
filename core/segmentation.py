import torch
import numpy as np
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

class Segmentor:
    def __init__(self, model_name="nvidia/segformer-b5-finetuned-ade-640-640"):
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.eval()

    def segment_pattern(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)

        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            logits = torch.nn.functional.interpolate(
                logits,
                size=image.shape[:2],
                mode="bilinear",
                align_corners=False
            )
            segmentation = logits.argmax(dim=1).squeeze().cpu().numpy()

        mask = np.zeros(segmentation.shape, dtype=np.uint8)
        unique_labels = np.unique(segmentation)
        if len(unique_labels) > 1:
            mask = (segmentation == unique_labels[1]).astype(np.uint8) * 255

        if np.sum(mask) < 100:
            mask = np.ones(segmentation.shape, dtype=np.uint8) * 255

        return mask
