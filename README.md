# Cityscapes Importance Label Generator

이 프로젝트는 Cityscapes 데이터셋에서 주어진 **disparity 맵**과 **semantic segmentation label**, 그리고 **ego-vehicle speed, camera calibration 정보**를 바탕으로 **중요도 맵 (importance map)**을 생성합니다. 이 중요도 맵은 픽셀 단위로 0~1 사이의 soft label로 출력되며, 중요한 객체(예: 가까운 차량, 보행자 등)는 1에 가까운 값을, 중요하지 않거나 멀리 있는 영역은 낮은 값을 갖습니다.

---

## 📥 입력 데이터

| 타입 | 설명 |
|------|------|
| `gtFine/val/<city>/*_gtFine_labelIds.png` | 시맨틱 클래스 라벨 ID 이미지 |
| `disparity/val/<city>/*_disparity.png` | stereo disparity 맵 (16bit png) |
| `vehicle/val/<city>/*_vehicle.json` | 차량 속도 정보 (`speed` key 사용) |
| `camera/val/<city>/*_camera.json` | 카메라 보정 파라미터 (fx, baseline 포함) |
| `gtFine/val/<city>/*_gtFine_polygons.json` | 시각화용 객체 폴리곤 정보 |

---

## ⚙️ 처리 과정 (Pipeline)

1. **속도와 카메라 파라미터 로드**
   - 각 프레임마다 `speed`, `fx`, `baseline`을 JSON에서 추출
   - 안전거리 계산:  
     \[
     d_{\text{safe}} = v \cdot t_{\text{react}} + \frac{v^2}{2a}
     \]
     (기본 반응시간 = 2.5초, 감속도 = 3.4m/s²)

2. **Disparity → 거리 변환**
   - 거리:  
     \[
     d = \frac{fx \cdot B}{\text{disparity}}
     \]

3. **클래스 기반 중요도 지정**
   - 고정 중요도 클래스: `road`, `building` 등 → 0.1
   - 동적 중요도 클래스: `car`, `person`, `bicycle` 등
     - 거리에 따라 중요도 감쇠 적용:
       \[
       I(d) =
       \begin{cases}
       1, & d \leq d_{\text{safe}} \\\\
       \exp(-\beta (d - d_{\text{safe}})), & d > d_{\text{safe}}
       \end{cases}
       \]
     - \(\beta = \frac{\ln 2}{d_{\text{safe}}}\): 반감 기준

4. **결과 저장**
   - `.npy`: float32 중요도 맵
   - `.png`: 0~255 회색조 정규화
   - `importance_vis.png`: 컬러맵 시각화
   - `distance_annotation.png`: 거리 및 폴리곤 시각화

---

## 📤 출력

| 파일명 | 설명 |
|--------|------|
| `*_importance.npy` | 픽셀별 중요도 값 (float32) |
| `*_importance.png` | 중요도 정규화 이미지 (grayscale) |
| `*_importance_vis.png` | 컬러맵 시각화 (hot cmap) |
| `*_distance_annotation.png` | 거리 + 객체명 주석 시각화 (red = 위험 범위, blue = 안전 범위) |

---

## 🏃 실행 예시

```python
label_root = "cityscapes_trainval/gtFine/val"
disparity_root = "cityscapes_trainval/disparity/val"
vehicle_root = "cityscapes_trainval/vehicle/val"
output_root = "cityscapes_trainval/importance_map/val"

selected_cities = ["frankfurt", "lindau", "munster"]

for city in selected_cities:
    batch_generate_importance_maps_auto_speed_with_progress(
        label_root, disparity_root, vehicle_root, output_root,
        selected_cities=city,
        visualize_every=30
    )
