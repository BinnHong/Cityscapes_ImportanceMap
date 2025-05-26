import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from matplotlib.patches import Polygon

CITYSCAPES_LABEL_MAP = {
    0: 'unlabeled', 1: 'ego vehicle', 2: 'rectification border', 3: 'out of roi',
    4: 'static', 5: 'dynamic', 6: 'ground', 7: 'road', 8: 'sidewalk',
    9: 'parking', 10: 'rail track', 11: 'building', 12: 'wall', 13: 'fence',
    14: 'guard rail', 15: 'bridge', 16: 'tunnel', 17: 'pole', 18: 'pole group',
    19: 'traffic light', 20: 'traffic sign', 21: 'vegetation', 22: 'terrain',
    23: 'sky', 24: 'person', 25: 'rider', 26: 'car', 27: 'truck', 28: 'bus',
    29: 'caravan', 30: 'trailer', 31: 'train', 32: 'motorcycle', 33: 'bicycle'
}

def compute_importance_map(label_img_path, disparity_path, speed, output_path, image_id, fx=2262.52, baseline=0.209313):
    t_react = 2.5
    a = 3.4
    d_safe = max(1e-6, speed * t_react + (speed ** 2) / (2 * a))
    beta = np.log(2) / d_safe

    fixed_low_classes = {
        'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence',
        'guard rail', 'bridge', 'tunnel', 'pole', 'pole group',
        'vegetation', 'terrain', 'sky'
    }

    dynamic_classes = {
        'person', 'rider', 'car', 'truck', 'bus', 'on rails', 'motorcycle', 'bicycle',
        'caravan', 'trailer', 'traffic sign', 'traffic light',
        'ground', 'dynamic', 'static'
    }

    label_img = np.array(Image.open(label_img_path))
    raw_disparity = np.array(Image.open(disparity_path)).astype(np.float32)
    valid_mask = raw_disparity > 0
    disparity = np.zeros_like(raw_disparity, dtype=np.float32)
    disparity[valid_mask] = (raw_disparity[valid_mask] - 1.0) / 256.0

    importance_map = np.ones_like(disparity, dtype=np.float32)

    for label_id in np.unique(label_img):
        label_name = CITYSCAPES_LABEL_MAP.get(label_id, 'unlabeled')
        mask = (label_img == label_id)

        if label_name in fixed_low_classes:
            importance_map[mask] = 0.1
        elif label_name in dynamic_classes:
            disp_values = disparity[mask]
            valid_disp = disp_values[(disp_values > 0) & np.isfinite(disp_values)]
            if valid_disp.size == 0:
                continue
            distance = (fx * baseline) / np.mean(valid_disp)
            if distance <= d_safe:
                importance_map[mask] = 1.0
            else:
                decay = np.exp(-beta * (distance - d_safe))
                decay = float(np.clip(decay, 0.0, 1.0))
                importance_map[mask] = decay
        else:
            importance_map[mask] = 0.0

    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, f"{image_id}_importance.npy"), importance_map)

    norm_map = (importance_map * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(norm_map).save(os.path.join(output_path, f"{image_id}_importance.png"))

    plt.imshow(norm_map, cmap='hot')
    plt.axis('off')
    plt.title(f'Importance: {image_id}')
    plt.savefig(os.path.join(output_path, f"{image_id}_importance_vis.png"))
    plt.close()

def batch_generate_importance_maps_auto_speed_with_progress(
    label_root, disparity_root, vehicle_root, output_root,
    selected_cities=None, image_ext=".png", visualize_every=100
):
    for city in os.listdir(label_root):
        if selected_cities and city not in selected_cities:
            continue

        label_city_path = os.path.join(label_root, city)
        disparity_city_path = os.path.join(disparity_root, city)
        vehicle_city_path = os.path.join(vehicle_root, city)
        output_city_path = os.path.join(output_root, city)
        # json_city_path 제거 (폴더 이동 없음)
        os.makedirs(output_city_path, exist_ok=True)

        label_files = [f for f in os.listdir(label_city_path) if f.endswith("_labelIds" + image_ext)]
        for idx, filename in enumerate(tqdm(label_files, desc=f"Processing {city}")):
            image_id = filename.replace("_gtFine_labelIds" + image_ext, "")
            label_path = os.path.join(label_city_path, filename)
            disparity_path = os.path.join(disparity_city_path, image_id + "_disparity.png")
            vehicle_path = os.path.join(vehicle_city_path, image_id + "_vehicle.json")
            json_path = os.path.join(label_city_path, image_id + "_gtFine_polygons.json")

            if not os.path.exists(disparity_path) or not os.path.exists(vehicle_path):
                print(f"Skipping {image_id}: missing disparity or vehicle JSON")
                continue

            with open(vehicle_path, 'r') as f:
                speed_data = json.load(f)

            # camera json 경로
            camera_path = os.path.join(
                os.path.dirname(os.path.dirname(label_city_path)).replace("gtFine", "camera"),
                city,
                image_id + "_camera.json"
            )
            if os.path.exists(camera_path):
                with open(camera_path, 'r') as f:
                    camera_data = json.load(f)
                    fx = camera_data['intrinsic']['fx']
                    baseline = camera_data['extrinsic']['baseline']
            else:
                fx = 2262.52
                baseline = 0.209313
                speed = speed_data.get("speed", 10.0)

            compute_importance_map(label_path, disparity_path, speed, output_city_path, image_id, fx=fx, baseline=baseline)

            if idx % visualize_every == 0 and os.path.exists(json_path):
                try:
                    label_img = np.array(Image.open(label_path))
                    original_path = label_path.replace("gtFine_labelIds.png", "leftImg8bit.png").replace("gtFine", "leftImg8bit")
                    original_img = np.array(Image.open(original_path)) if os.path.exists(original_path) else np.zeros_like(label_img)
                    raw_disp = np.array(Image.open(disparity_path)).astype(np.float32)
                    disparity = np.zeros_like(raw_disp, dtype=np.float32)
                    valid_mask = raw_disp > 0
                    disparity[valid_mask] = (raw_disp[valid_mask] - 1.0) / 256.0

                    with open(json_path, 'r') as f:
                        polygon_data = json.load(f)

                    t_react = 2.5
                    a = 3.4
                    d_safe = max(1e-6, speed * t_react + (speed ** 2) / (2 * a))

                    fig, ax = plt.subplots(figsize=(14, 7))
                    ax.imshow(original_img)
                    ax.set_title(f"Cityscapes Disparity-Corrected Distance (d_safe = {d_safe:.1f}m)")
                    ax.axis('off')

                    for obj in polygon_data['objects']:
                        if 'polygon' not in obj or not obj['polygon']:
                            continue
                        label = obj['label']
                        polygon = np.array(obj['polygon'])
                        mask = np.zeros(disparity.shape, dtype=np.uint8)
                        cv2 = __import__('cv2')
                        cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
                        valid = (mask == 1) & np.isfinite(disparity) & (disparity > 0)
                        if np.count_nonzero(valid) < 5:
                            continue
                        median_disp = np.median(disparity[valid])
                        distance = (fx * baseline) / median_disp
                        color = 'red' if distance <= d_safe else 'blue'
                        ax.add_patch(Polygon(polygon, closed=True, edgecolor=color, fill=False, linewidth=1))
                        cx, cy = np.mean(polygon, axis=0)
                        ax.text(cx, cy, f"{label}: {distance:.1f}m", color=color, fontsize=6)

                    plt.tight_layout()
                    plt.savefig(os.path.join(output_city_path, f"{image_id}_distance_annotation.png"))
                    plt.close()

                except Exception as e:
                    print(f"Distance annotation failed for {image_id}: {e}")

# 사용 예시 경로 설정
label_root = r"F:\학교관련\비즈니스어낼리틱스\cityscapes_trainval\gtFine\val"
disparity_root = r"F:\학교관련\비즈니스어낼리틱스\cityscapes_trainval\disparity\val"
vehicle_root = r"F:\학교관련\비즈니스어낼리틱스\cityscapes_trainval\vehicle\val"
output_root = r"F:\학교관련\비즈니스어낼리틱스\cityscapes_trainval\importance_map\val"

selected_cities = ["frankfurt", "lindau", "munster"]

for sc in selected_cities:
    batch_generate_importance_maps_auto_speed_with_progress(
        label_root, disparity_root, vehicle_root, output_root,
        selected_cities=sc,
        visualize_every=30
    )
