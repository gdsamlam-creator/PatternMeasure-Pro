import numpy as np

def calibrate_scale(pixel_distance, real_distance, unit="cm"):
    return pixel_distance / real_distance, unit

def measure_edges(edge_points):
    if len(edge_points) < 2:
        return []

    edges = []
    for i in range(len(edge_points)):
        p1 = edge_points[i]
        p2 = edge_points[(i + 1) % len(edge_points)]
        distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        edges.append({
            "start": p1,
            "end": p2,
            "pixel_length": distance
        })

    return edges

def calculate_area(edge_points, scale):
    if len(edge_points) < 3:
        return 0.0, ""

    scale_factor, unit = scale

    n = len(edge_points)
    area = 0.0
    for i in range(n):
        x1, y1 = edge_points[i]
        x2, y2 = edge_points[(i + 1) % n]
        area += x1 * y2 - x2 * y1

    area = abs(area) / 2.0
    real_area = area * (scale_factor ** 2)
    area_unit = f"{unit}²"

    return real_area, area_unit
