import numpy as np
import subprocess
import os

def generate_qp_zones_string(importance_map_path, block_size=16, 
                              frame_width=2048, frame_height=1024,
                              base_qp=32, delta=3):
    imp = np.load(importance_map_path)
    H, W = imp.shape
    assert H == frame_height and W == frame_width, "프레임 크기와 맞지 않습니다."

    h_blocks = H // block_size
    w_blocks = W // block_size

    block_importance = np.zeros((h_blocks, w_blocks))
    for i in range(h_blocks):
        for j in range(w_blocks):
            block = imp[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            block_importance[i, j] = np.mean(block)

    max_qp = base_qp + delta
    min_qp = base_qp - delta
    norm_imp = (block_importance - block_importance.min()) / (block_importance.max() - block_importance.min() + 1e-8)
    qp_map = max_qp - norm_imp * (max_qp - min_qp)
    qp_map = np.round(qp_map).astype(int)

    zone_str_list = []
    for i in range(h_blocks):
        for j in range(w_blocks):
            x = j * block_size
            y = i * block_size
            w = block_size
            h = block_size
            qp = qp_map[i, j]
            zone_str_list.append(f"{x},{y},{w},{h},{qp}")
    zone_string = "/".join(zone_str_list)
    return zone_string

def png_to_compressed_png(importance_map_path, input_png, output_png,
                           tmp_video='tmp_output.mp4',
                           frame_width=2048, frame_height=1024, block_size=16):
    # Step 1: QP zones 생성
    zone_string = generate_qp_zones_string(importance_map_path, block_size, frame_width, frame_height)

    # Step 2: PNG → H.264 압축
    cmd_encode = [
        'ffmpeg',
        '-y',
        '-loop', '1',
        '-i', input_png,
        '-t', '1',
        '-vf', f'scale={frame_width}:{frame_height}',
        '-c:v', 'libx264',
        '-x264-params', f"zones={zone_string}",
        '-frames:v', '1',
        tmp_video
    ]
    #print("압축 중 (H.264):", " ".join(cmd_encode))
    subprocess.run(cmd_encode, check=True)

    # Step 3: 압축된 mp4 → PNG로 복원
    cmd_decode = [
        'ffmpeg',
        '-y',
        '-i', tmp_video,
        '-frames:v', '1',
        output_png
    ]
    print("복원 중 (PNG):", " ".join(cmd_decode))
    subprocess.run(cmd_decode, check=True)

    # 선택적으로 임시 비디오 삭제
    os.remove(tmp_video)
    print(f"압축 후 복원된 PNG 저장 완료: {output_png}")

# 실행 예시
if __name__ == '__main__':
    importance_map = 'C:/Users/ghdql/Desktop/tmp/새 폴더 (4)/cologne_000090_000019_importance.npy'
    input_png = 'C:/Users/ghdql/Desktop/tmp/새 폴더 (4)/cologne_000090_000019_leftImg8bit.png'
    output_png = 'compressed_output.png'

    png_to_compressed_png(importance_map, input_png, output_png)
