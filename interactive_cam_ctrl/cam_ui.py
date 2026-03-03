import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================
# SE(3) 工具函数
# =============================

def skew(w):
    wx, wy, wz = w
    return torch.tensor([
        [0, -wz, wy],
        [wz, 0, -wx],
        [-wy, wx, 0]
    ], dtype=torch.float32)

def se3_exp(xi):
    omega = xi[:3]
    v = xi[3:]

    theta = torch.norm(omega) + 1e-8

    if theta < 1e-6:
        R = torch.eye(3)
    else:
        omega_hat = skew(omega / theta)
        I = torch.eye(3)
        R = I + torch.sin(theta) * omega_hat + \
            (1 - torch.cos(theta)) * (omega_hat @ omega_hat)

    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = v

    return T

def mouse_to_twist(delta_m, mode,
                   k_rot=0.01,
                   k_trans=0.05):

    dx, dy = delta_m
    xi = torch.zeros(6)

    if mode == "orbit":
        xi[0] = k_rot * dy   # pitch
        xi[1] = k_rot * dx   # yaw

    elif mode == "zoom":
        xi[5] = k_trans * dy

    elif mode == "pan":
        xi[3] = k_trans * dx
        xi[4] = k_trans * dy

    return xi


# =============================
# 交互界面
# =============================

class CameraUI:

    def __init__(self):

        self.mode = "orbit"
        self.mouse_path = []
        self.trajectory = []
        self.T = torch.eye(4)

        self.fig = plt.figure(figsize=(10, 5))
        plt.rcParams['toolbar'] = 'None'
    
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)

        # 左侧 2D 画布
        self.ax2d = self.fig.add_subplot(121)
        self.ax2d.set_title("2D Mouse Input")
        self.ax2d.set_xlim(0, 1)
        self.ax2d.set_ylim(0, 1)

        # 右侧 3D 轨迹
        self.ax3d = self.fig.add_subplot(122, projection='3d')
        self.ax3d.set_title("3D Camera Trajectory")

        self.ax3d.set_xlim(-1, 1)
        self.ax3d.set_ylim(-1, 1)
        self.ax3d.set_zlim(-1, 1)

        self.dragging = False

        self.connect_events()

    def connect_events(self):
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_press(self, event):
        if event.inaxes == self.ax2d and event.button == 1:
            self.dragging = True
            self.mouse_path = [(event.xdata, event.ydata)]

    def on_release(self, event):
        self.dragging = False

    def on_motion(self, event):
        if self.dragging and event.inaxes == self.ax2d:
            self.mouse_path.append((event.xdata, event.ydata))
            self.update_trajectory()
            self.draw()

    def on_key(self, event):
        if event.key == 'o':
            self.mode = "orbit"
        elif event.key == 'z':
            self.mode = "zoom"
        elif event.key == 'p':
            self.mode = "pan"
        elif event.key == 'r':
            self.reset()

        print("Current mode:", self.mode)

    def reset(self):
        self.mouse_path = []
        self.trajectory = []
        self.T = torch.eye(4)
        self.draw()

    def update_trajectory(self):
        if len(self.mouse_path) < 2:
            return

        p1 = self.mouse_path[-2]
        p2 = self.mouse_path[-1]

        delta = np.array(p2) - np.array(p1)

        xi = mouse_to_twist(delta, self.mode)
        delta_T = se3_exp(xi)

        self.T = self.T @ delta_T
        self.trajectory.append(self.T[:3, 3].numpy())

    def draw(self):

        # 2D 画布
        self.ax2d.clear()
        self.ax2d.set_title(f"2D Mouse Input (Mode: {self.mode})")
        self.ax2d.set_xlim(0, 1)
        self.ax2d.set_ylim(0, 1)

        if len(self.mouse_path) > 1:
            xs, ys = zip(*self.mouse_path)
            self.ax2d.plot(xs, ys)

        # 3D 轨迹
        self.ax3d.clear()
        self.ax3d.set_title("3D Camera Trajectory")
        self.ax3d.set_xlim(-1, 1)
        self.ax3d.set_ylim(-1, 1)
        self.ax3d.set_zlim(-1, 1)

        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            self.ax3d.plot(traj[:, 0],
                           traj[:, 1],
                           traj[:, 2])

        plt.draw()


# =============================
# 运行
# =============================

if __name__ == "__main__":
    ui = CameraUI()
    plt.show()