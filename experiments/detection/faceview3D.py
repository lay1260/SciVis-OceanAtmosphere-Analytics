#顺序不对
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import OpenVisus as ov

# =====================================================
# 1️⃣ 参数设置
# =====================================================
variable = "T"
time_step = 0
vertical_level = 0
quality = -12

# =====================================================
# 2️⃣ 读取 6 个 face 的第 0 层
# =====================================================
face_layers = []

for face in range(6):
    idx_url = (
        f"https://nsdf-climate3-origin.nationalresearchplatform.org:50098/"
        f"nasa/nsdf/climate3/dyamond/GEOS/"
        f"GEOS_{variable}/{variable.lower()}_face_{face}_depth_52_time_0_10269.idx"
    )
    db = ov.LoadDataset(idx_url)
    data3D = db.read(time=time_step, quality=quality)
    face_layers.append(data3D[vertical_level,:, :])

print("Loaded face shape:", face_layers[0].shape)

# =====================================================
# 3️⃣ ✅ 按你的规则做旋转修正（非常关键）
# =====================================================
# face5：逆时针 180°
#face_layers[5] = np.rot90(face_layers[5], k=2)

# face4：逆时针 90°
face_layers[4] = np.rot90(face_layers[4], k=1)

# face2：逆时针 90°
#face_layers[2] = np.rot90(face_layers[2], k=1)

# face3：逆时针 90°
face_layers[3] = np.rot90(face_layers[3], k=1)

# =====================================================
# 4️⃣ 工具函数：归一化 + 画单个立方体面
# =====================================================
def normalize(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

def plot_face(ax, face, plane="z", value=1.0):
    """
    plane: 'x', 'y', 'z'
    value: 固定坐标值（±1）
    """
    H, W = face.shape
    u = np.linspace(-1, 1, W)
    v = np.linspace(-1, 1, H)
    U, V = np.meshgrid(u, v)

    face_n = normalize(face)
    colors = cm.coolwarm(face_n)

    if plane == "x":
        X = np.full_like(U, value)
        Y = U
        Z = V
    elif plane == "y":
        X = U
        Y = np.full_like(U, value)
        Z = V
    elif plane == "z":
        X = U
        Y = V
        Z = np.full_like(U, value)
    else:
        raise ValueError("plane must be 'x', 'y', or 'z'")

    ax.plot_surface(
        X, Y, Z,
        facecolors=colors,
        rstride=1, cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False
    )

# =====================================================
# 5️⃣ 绘制可旋转 3D 立方体
# =====================================================
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# ✅ 按你展开逻辑绑定到立方体 6 个面
plot_face(ax, face_layers[0], plane="y", value=+1)  # front
plot_face(ax, face_layers[3], plane="y", value=-1)  # back
plot_face(ax, face_layers[4], plane="x", value=-1)  # left
plot_face(ax, face_layers[1], plane="x", value=+1)  # right
plot_face(ax, face_layers[2], plane="z", value=+1)  # top
plot_face(ax, face_layers[5], plane="z", value=-1)  # bottom

# 立方体外观设置
ax.set_box_aspect([1, 1, 1])
ax.set_axis_off()

# 初始视角（鼠标可自由旋转）
ax.view_init(elev=25, azim=45)

plt.title("GEOS Cubed-Sphere Faces as a Rotatable 3D Cube (T, Level 0)")
plt.show()
