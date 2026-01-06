import numpy as np
import pyvista as pv
import panel as pn

pn.extension("vtk")

# -------------------------------
# 1. 模拟数据
# -------------------------------
nx, ny, nz = 50, 50, 8
num_time = 6

x = np.linspace(105, 135, nx)
y = np.linspace(0, 30, ny)
z = np.linspace(0, 1000, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

data_time = [
    np.sin(X/10 + t) * np.cos(Y/10) * np.exp(-Z/200)
    for t in range(num_time)
]

grid = pv.StructuredGrid(X, Y, Z)
grid["U"] = data_time[0].flatten(order="F")

# -------------------------------
# 2. 主视图 plotter
# -------------------------------
main_plotter = pv.Plotter(off_screen=True)
main_plotter.add_mesh(
    grid.slice_orthogonal(),
    scalars="U",
    cmap="viridis"
)

# 关键：强制创建渲染窗口（不弹窗）
main_plotter.show(auto_close=False)

main_view = pn.pane.VTK(
    main_plotter.ren_win,
    sizing_mode="stretch_both"
)

# -------------------------------
# 3. 截面 plotter
# -------------------------------
slice_plotter = pv.Plotter(off_screen=True)
slice_plotter.show(auto_close=False)

slice_view = pn.pane.VTK(
    slice_plotter.ren_win,
    sizing_mode="stretch_both"
)

# -------------------------------
# 4. 点选截面
# -------------------------------
clicked = []

def pick_callback(point):
    clicked.append(point)

    if len(clicked) == 2:
        p1, p2 = clicked
        line = pv.Line(p1, p2, resolution=200)
        sampled = grid.sample_along_line(line)

        slice_plotter.clear()
        slice_plotter.add_mesh(sampled, scalars="U", cmap="viridis")
        slice_view.update(slice_plotter.ren_win)

        clicked.clear()

main_plotter.enable_point_picking(
    callback=pick_callback,
    show_message=True,
    color="red",
    point_size=12
)

# -------------------------------
# 5. 时间滑块
# -------------------------------
def update_time(t):
    grid["U"] = data_time[t].flatten(order="F")
    main_plotter.update_scalars(grid["U"])
    main_view.update(main_plotter.ren_win)

time_slider = pn.widgets.IntSlider(
    name="Time", start=0, end=num_time - 1, value=0
)
time_slider.param.watch(lambda e: update_time(e.new), "value")

# -------------------------------
# 6. 布局
# -------------------------------
ui = pn.Column(
    "## 3D 海洋可视化（解决 Panel 空白问题版）",
    time_slider,
    pn.Row(main_view, slice_view)
)

ui.servable()
