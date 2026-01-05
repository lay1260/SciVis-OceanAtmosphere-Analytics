"""
将 GEOS cubed-sphere 6 个 face 展开为 3x4“立方体展开图”，并绘制指定变量的单层场。

默认示例：近地面温度 T，time_step=0，level=0，quality=-6。
用法示例：
    python geos_cube_plot.py            # 使用默认参数
    python geos_cube_plot.py T 0 0 -6   # variable time_step level quality

保存/显示：
    - 默认直接显示 plt.show()
    - 如需保存，可在 main 中改为 plt.savefig("out.png", dpi=200)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import OpenVisus as ov


BASE_URL = (
    "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/"
    "nasa/nsdf/climate3/dyamond/GEOS"
)


def get_face_url(variable: str, face: int) -> str:
    return (
        f"{BASE_URL}/GEOS_{variable.upper()}/"
        f"{variable.lower()}_face_{face}_depth_52_time_0_10269.idx"
    )


def load_face_layer(variable: str, face: int, time_step: int, level: int, quality: int):
    url = get_face_url(variable, face)
    db = ov.LoadDataset(url)
    data3d = db.read(time=time_step, quality=quality)
    # 取指定垂直层
    layer = data3d[level, :, :]
    return layer


def rotate_faces(face_layers):
    """
    按给定规则旋转 face，以便展开拼接：
    - face5: 逆时针 180°
    - face4: 逆时针 90°
    - face2: 逆时针 90°
    - face3: 逆时针 90°
    """
    face_layers = list(face_layers)
    face_layers[5] = np.rot90(face_layers[5], k=2)
    face_layers[4] = np.rot90(face_layers[4], k=1)
    face_layers[2] = np.rot90(face_layers[2], k=1)
    face_layers[3] = np.rot90(face_layers[3], k=1)
    return face_layers


def stitch_faces(face_layers):
    """
    将旋转后的 6 个 face 拼成 3x4 展开图：
      中行：face0, face1, face3, face4
      上行中：face2
      下行中：face5
    返回拼接后的 2D 数组。
    """
    H, W = face_layers[0].shape
    stitched = np.full((3 * H, 4 * W), np.nan)
    # 中行
    stitched[H:2 * H, 0:W] = face_layers[0]
    stitched[H:2 * H, W:2 * W] = face_layers[1]
    stitched[H:2 * H, 2 * W:3 * W] = face_layers[3]
    stitched[H:2 * H, 3 * W:4 * W] = face_layers[4]
    # 上行中
    stitched[2 * H:3 * H, 2 * W:3 * W] = face_layers[2]
    # 下行中
    stitched[0:H, 2 * W:3 * W] = face_layers[5]
    return stitched


def main():
    # 参数：variable time_step level quality
    variable = sys.argv[1] if len(sys.argv) > 1 else "T"
    time_step = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    level = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    quality = int(sys.argv[4]) if len(sys.argv) > 4 else -6

    faces = range(6)
    face_layers = []
    for face in faces:
        layer = load_face_layer(variable, face, time_step, level, quality)
        face_layers.append(layer)
        print(f"face {face} shape: {layer.shape}")

    face_layers = rotate_faces(face_layers)
    stitched = stitch_faces(face_layers)
    print("Stitched shape:", stitched.shape)

    plt.figure(figsize=(18, 10))
    im = plt.imshow(stitched, origin="lower", cmap="coolwarm")
    plt.colorbar(im, label=f"{variable} (arb.)")
    plt.title(
        f"GEOS DYAMOND {variable} (level {level}, time {time_step}, quality {quality}) – Stitched 6 Faces"
    )
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()

