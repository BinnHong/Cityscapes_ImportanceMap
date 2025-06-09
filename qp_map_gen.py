import numpy as np
import pandas as pd
import os
'''
def generate_qp_map_csv(importance_map_path, output_csv_path='qp_map.csv',
                         block_size=16, base_qp=32, delta=3):
    """
    importance_map.npy → QP map 생성 후 CSV 저장
    """
    # 1. importance map 로드
    imp = np.load(importance_map_path)
    H, W = imp.shape

    # 2. 블록 수 계산
    h_blocks = H // block_size
    w_blocks = W // block_size

    # 3. 블록 단위 importance 평균 계산
    block_importance = np.zeros((h_blocks, w_blocks))
    for i in range(h_blocks):
        for j in range(w_blocks):
            block = imp[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            block_importance[i, j] = np.mean(block)

    # 4. importance → QP (정규화 후 반비례 매핑)
    max_qp = base_qp + delta
    min_qp = base_qp - delta
    norm_imp = (block_importance - block_importance.min()) / (block_importance.max() - block_importance.min() + 1e-8)
    qp_map = max_qp - norm_imp * (max_qp - min_qp)
    qp_map = np.round(qp_map).astype(int)

    # 5. CSV로 저장
    df = pd.DataFrame(qp_map)
    df.to_csv(output_csv_path, index=False, header=False)
    print(f"QP 맵이 CSV로 저장되었습니다: {output_csv_path}")

    return qp_map
'''
import matplotlib.pyplot as plt

def generate_qp_map_csv_viz(importance_map_path, output_csv_path='qp_map.csv',
                         block_size=16, base_qp=30, delta=15, show_plot=True):
    """
    importance_map.npy → QP map 생성 후 CSV 저장 + 시각화
    """
    # 1. importance map 로드
    imp = np.load(importance_map_path)
    H, W = imp.shape

    # 2. 블록 수 계산
    h_blocks = H // block_size
    w_blocks = W // block_size

    # 3. 블록 단위 importance 평균 계산
    block_importance = np.zeros((h_blocks, w_blocks))
    for i in range(h_blocks):
        for j in range(w_blocks):
            block = imp[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            block_importance[i, j] = np.mean(block)

    # 4. importance → QP (정규화 후 반비례 매핑)
    max_qp = base_qp + delta
    min_qp = base_qp - delta
    norm_imp = (block_importance - block_importance.min()) / (block_importance.max() - block_importance.min() + 1e-8)
    qp_map = max_qp - norm_imp * (max_qp - min_qp)
    qp_map = np.round(qp_map).astype(int)

    # 5. CSV로 저장
    df = pd.DataFrame(qp_map)
    df.to_csv(output_csv_path, index=False, header=False)
    print(f"QP 맵이 CSV로 저장되었습니다: {output_csv_path}")

    # 6. 시각화
    if show_plot:
        plt.figure(figsize=(8, 6))
        plt.imshow(qp_map, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Quantization Parameter (QP)')
        plt.title('QP Map from Importance Map')
        plt.xlabel('Block Index (x)')
        plt.ylabel('Block Index (y)')
        plt.tight_layout()
        plt.show()

    return qp_map

# 실행 예시
if __name__ == '__main__':
    foler_path = 'C:/Users/ghdql/Desktop/tmp/새 폴더 (7)/'
    file_name = 'frankfurt_000000_000294'
    generate_qp_map_csv_viz(foler_path + file_name + '_importance.npy', foler_path + file_name + '_qp_map.csv')

