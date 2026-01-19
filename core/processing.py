import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def process_pattern_image(image, mask):
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    masked = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

    gray = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image_rgb, []

    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    edge_points = []
    for point in approx:
        edge_points.append(tuple(point[0]))

    result = image_rgb.copy()
    cv2.drawContours(result, [approx], -1, (0, 255, 0), 3)

    for point in edge_points:
        cv2.circle(result, point, 8, (255, 0, 0), -1)

    return result, edge_points

def draw_measurements(image, edge_points, scale):
    if len(image.shape) == 3:
        result = image.copy()
    else:
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    scale_factor, unit = scale
    font_scale = max(0.5, min(result.shape[0], result.shape[1]) / 800)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", int(16 * font_scale))
    except:
        font = ImageFont.load_default()

    img_pil = Image.fromarray(result)
    draw = ImageDraw.Draw(img_pil)

    for i in range(len(edge_points)):
        p1 = edge_points[i]
        p2 = edge_points[(i + 1) % len(edge_points)]

        pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        real_distance = pixel_distance * scale_factor

        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2

        text = f"{real_distance:.1f} {unit}"
        draw.text((mid_x, mid_y), text, fill=(255, 255, 0), font=font)

    return np.array(img_pil)
