#六面体展开图
import numpy as np
import matplotlib.pyplot as plt
import OpenVisus as ov

# -----------------------------
# 1️⃣ 参数设置
# -----------------------------
variable = "T"
faces = range(6)
time_step = 3000
vertical_level = 0
data_resolution = -9

face_layers = []

# -----------------------------
# 2️⃣ 读取 6 个 face 的第 0 层
# -----------------------------
for face in faces:
    idx_url = (
        f"https://nsdf-climate3-origin.nationalresearchplatform.org:50098/"
        f"nasa/nsdf/climate3/dyamond/GEOS/"
        f"GEOS_{variable}/{variable.lower()}_face_{face}_depth_52_time_0_10269.idx"
    )

    db = ov.LoadDataset(idx_url)
    data3D = db.read(time=time_step, quality=data_resolution)

    # 取近地面（第 0 层）
    layer0 = data3D[vertical_level,:, : ]
    face_layers.append(layer0)
    print(layer0.shape)
# -----------------------------
#  对指定 face 做旋转修正——逆时针
# -----------------------------
# face5：逆时针旋转 180°
face_layers[5] = np.rot90(face_layers[5], k=2)
# face4：逆时针旋转 90°
face_layers[4] = np.rot90(face_layers[4], k=1)
# face2：逆时针旋转 90°
face_layers[2] = np.rot90(face_layers[2], k=1)
# face3：逆时针旋转 90°
face_layers[3] = np.rot90(face_layers[3], k=1)
# -----------------------------
# 3️⃣ 数值拼接（3 行立方体展开图）
# -----------------------------
H, W = face_layers[0].shape

# 创建画布（3 行 × 4 列 face）
stitched = np.full((3 * H, 4 * W), np.nan)

# 中间一行：face0 - face1 - face3 - face4
stitched[H:2*H, 0:W]     = face_layers[3]
stitched[H:2*H, W:2*W]   = face_layers[4]
stitched[H:2*H, 2*W:3*W] = face_layers[0]
stitched[H:2*H, 3*W:4*W] = face_layers[1]

# 上一行中间：face2
stitched[2*H:3*H, 0:W] = face_layers[2]

# 下一行中间：face5
stitched[0:H, 0:W] = face_layers[5]

print("Stitched shape:", stitched.shape)

# -----------------------------
# 4️⃣ 绘制拼接结果
# -----------------------------
plt.figure(figsize=(18, 10))
im = plt.imshow(
    stitched,
    origin="lower",
    cmap="coolwarm"
)

plt.colorbar(im, label="Temperature (K)")
plt.title("GEOS DYAMOND Near-Surface Temperature (T, Level 0) – Stitched 6 Faces")
plt.axis("off")
plt.show()
