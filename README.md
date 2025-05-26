# Cityscapes Importance Label Generator

이 프로젝트는 Cityscapes 데이터셋에서 주어진 **disparity 맵**과 **semantic segmentation label**, 그리고 **ego-vehicle speed, camera calibration 정보**를 바탕으로 \*\*중요도 맵 (importance map)\*\*을 생성합니다.

중요도 맵은 픽셀 단위로 0\~1 사이의 soft label을 갖고, 가까운 차량/보행자와 같은 중요 객체는 1에 가까운 값을, 배경이나 멀리 있는 객체는 낮은 값을 갖습니다.

---

## 📥 입력 데이터 구조

```
cityscapes_trainval/
├── gtFine/val/<city>/*_gtFine_labelIds.png         # semantic 라벨
├── disparity/val/<city>/*_disparity.png            # stereo disparity (16-bit)
├── vehicle/val/<city>/*_vehicle.json               # 속도 정보 ("speed")
├── camera/val/<city>/*_camera.json                 # fx, baseline 등 camera 파라미터
├── gtFine/val/<city>/*_gtFine_polygons.json        # 폴리곤 객체 위치 정보 (시각화용)
```

---

## ⚙️ 처리 과정 요약

1. **속도 & 카메라 파라미터 로딩**

   * `vehicle.json` → speed
   * `camera.json` → `fx`, `baseline`
   * 안전 거리 계산:

```
d_safe = v * t_react + v^2 / (2 * a)
기본값: t_react = 2.5 sec, a = 3.4 m/s^2
```

2. **Disparity → 거리 변환**

```
distance = (fx * baseline) / disparity
```

3. **클래스 중요도 설정**

* 고정 중요도 클래스 (`road`, `building`, ...): 0.1 부여
* 동적 객체 (`car`, `person`, `bicycle`, ...): 거리 기반 감쇠 함수 적용

```
I(d) =
  1.0                        if d <= d_safe
  exp(-β * (d - d_safe))    if d > d_safe

  where β = ln(2) / d_safe  # 거리 2배가 되면 중요도 반감
```

4. **중요도 맵 저장**

* `.npy`: 중요도 값 (float32)
* `.png`: 정규화된 grayscale
* `_vis.png`: 컬러맵 시각화
* `_distance_annotation.png`: 거리 주석 + 안전거리 색상 시각화

---

## 📤 출력 예시

| 파일명                         | 설명                         |
| --------------------------- | -------------------------- |
| `*_importance.npy`          | 픽셀별 중요도 float32 배열         |
| `*_importance.png`          | 정규화 grayscale 이미지 (0\~255) |
| `*_importance_vis.png`      | 컬러맵 시각화 (hot colormap)     |
| `*_distance_annotation.png` | 각 객체의 거리 + 색상 주석 시각화       |

---

## 🏃 실행 예시

```python
label_root = "cityscapes_trainval/gtFine/val"
disparity_root = "cityscapes_trainval/disparity/val"
vehicle_root = "cityscapes_trainval/vehicle/val"
camera_root = "cityscapes_trainval/camera/val"
output_root = "cityscapes_trainval/importance_map/val"

selected_cities = ["frankfurt", "lindau", "munster"]

for city in selected_cities:
    batch_generate_importance_maps_auto_speed_with_progress(
        label_root, disparity_root, vehicle_root, output_root,
        selected_cities=city,
        visualize_every=30
    )
```

---

## 📚 참고

* [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
* [Camera Calibration 공식 문서 (PDF)](https://github.com/mcordts/cityscapesScripts/blob/master/docs/csCalibration.pdf)
* Disparity 변환 공식: `(p - 1) / 256.0` (p는 uint16 픽셀 값)

---

## 🎯 활용 가능성

* 중요도 기반 비디오 압축 (중요 영역 고품질 유지)
* 중요도 기반 가중 손실 학습 (e.g., detection, segmentation)
* 중요 영역 ROI 지정 및 후처리 우선순위 설정

---

