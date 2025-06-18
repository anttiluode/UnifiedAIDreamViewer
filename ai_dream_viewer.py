# ai_dream_viewer_v2.py
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import threading
import time
import queue
from scipy.ndimage import laplace
from collections import deque
from dataclasses import dataclass
from typing import List
from PIL import Image, ImageTk

# PyTorch imports for the AI model
import torch
import torch.nn as nn
import torch.optim as optim

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# PART 1: AI DECODER MODEL (U-Net Architecture)
# ==============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

# --- FIX 1: Corrected the U-Net Up-sampling Module ---
# The original 'Up' class had a channel mismatch during concatenation.
# This version correctly defines the input channels for the convolution
# after concatenating the skip connection and the upsampled path.
class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels) # The input to conv is the full concatenated channels
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        # x1 is from the up path, x2 is the skip connection from the down path
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
# --- END FIX 1 ---

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x): return self.conv(x)

class DecoderUNet(nn.Module):
    def __init__(self, n_channels_in=4, n_channels_out=1, bilinear=True):
        super(DecoderUNet, self).__init__()
        self.inc = DoubleConv(n_channels_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        # The 'in_channels' for Up must account for the concatenated skip connection
        self.up1 = Up(512 + 256, 256, bilinear)
        self.up2 = Up(256 + 128, 128, bilinear)
        self.up3 = Up(128 + 64, 64, bilinear)
        self.outc = OutConv(64, n_channels_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return torch.sigmoid(self.outc(x))

# ==============================================================================
# PART 2: BRAIN SIMULATION (Core classes from previous script)
# ==============================================================================
# (This part is unchanged, so it's collapsed for brevity. It's the same as the previous script)
@dataclass
class DiscreteNeuron:
    x: float; y: float; membrane_potential: float = -70.0; spike_threshold: float = -55.0
    refractory_period: int = 0; neuron_type: str = "excitatory"; ephaptic_sensitivity: float = 0.1
    def update(self, local_field_value, dt=1.0, noise_level=0.5):
        if self.refractory_period > 0:
            self.refractory_period -= 1; self.membrane_potential = -80.0; return False
        ephaptic_current = self.ephaptic_sensitivity * local_field_value
        leak_current = (self.membrane_potential + 70.0) * 0.1
        self.membrane_potential += dt * (-leak_current + ephaptic_current + np.random.normal(0, noise_level))
        if self.membrane_potential > self.spike_threshold:
            self.refractory_period = 3; self.membrane_potential = -80.0; return True
        return False
class VisionProcessor:
    def __init__(self, target_size=(16, 16)):
        self.target_size = target_size; self.cap = None; self.current_frame = None
        self.processed_vision = np.zeros(target_size); self.camera_active = False
        self.available_cameras = self.detect_cameras()
    def detect_cameras(self):
        cams = [i for i in range(3) if cv2.VideoCapture(i).isOpened()]; return cams if cams else [0]
    def start_camera(self, camera_index=0):
        try:
            if self.cap: self.cap.release()
            self.cap = cv2.VideoCapture(camera_index)
            if self.cap.isOpened(): self.camera_active = True; return True
        except Exception: pass
        return False
    def stop_camera(self):
        self.camera_active = False
        if self.cap: self.cap.release()
    def process_frame(self):
        if not self.camera_active or not self.cap or not self.cap.isOpened(): return self.processed_vision, None
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, self.target_size)
            self.processed_vision = resized.astype(np.float32) / 255.0
            return self.processed_vision, gray
        return self.processed_vision, None
class HybridEphapticField:
    def __init__(self, grid_size=32, area_name="Unknown"):
        self.grid_size = grid_size; self.area_name = area_name; self.neurons: List[DiscreteNeuron] = []
        self.tactical_field = self.generate_fractal_noise(grid_size, 1.2)
        self.positional_field = self.generate_fractal_noise(grid_size, 0.75)
        self.strategic_field = self.generate_fractal_noise(grid_size, 0.3)
        self.combined_field = np.zeros((grid_size, grid_size), dtype=complex)
        self.spike_history = deque(maxlen=100); self.is_vision_area = False; self.vision_coupling = 0.2
        self.update_combined_field()
    def generate_fractal_noise(self, s, beta):
        kx, ky = np.fft.fftfreq(s).reshape(1,-1), np.fft.fftfreq(s).reshape(-1,1)
        k = np.sqrt(kx*kx + ky*ky); k[0,0] = 1e-6
        spec = (k**(-beta/2))*np.exp(2j*np.pi*np.random.rand(s,s))
        noise = np.real(np.fft.ifft2(spec)); return (noise - noise.min())/(noise.max()-noise.min())
    def set_neuron_count(self, count):
        count = min(count, 500)
        if count > len(self.neurons):
            for _ in range(count - len(self.neurons)):
                ntype = "excitatory" if np.random.random() > 0.2 else "inhibitory"
                self.neurons.append(DiscreteNeuron(x=np.random.uniform(1, self.grid_size-2),
                                                   y=np.random.uniform(1, self.grid_size-2),
                                                   neuron_type=ntype,
                                                   ephaptic_sensitivity=0.1 if ntype=="excitatory" else -0.05))
        elif count < len(self.neurons): self.neurons = self.neurons[:count]
    def apply_vision_input(self, vision_data):
        if not self.is_vision_area or len(self.neurons) == 0 or vision_data is None: return
        vision_resized = cv2.resize(vision_data, (self.grid_size, self.grid_size))
        for n in self.neurons:
            gx, gy = int(n.x), int(n.y)
            n.membrane_potential += self.vision_coupling * vision_resized[gy,gx] * 10.0
    def update_neurons(self):
        spikes = []
        for n in self.neurons:
            gx, gy = int(n.x), int(n.y)
            local_field = np.abs(self.combined_field[gy,gx]) * 0.1
            if n.update(local_field):
                spikes.append(n); self.apply_neuron_spike_to_field(n)
        self.spike_history.append(len(spikes)); return spikes
    def apply_neuron_spike_to_field(self, n):
        gx, gy = int(n.x), int(n.y)
        r, strength = 3, 0.1 if n.neuron_type == "excitatory" else -0.05
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                fx,fy = gx+dx, gy+dy
                if 0 <= fx < self.grid_size and 0 <= fy < self.grid_size:
                    dist = np.sqrt(dx*dx+dy*dy)
                    if dist <= r:
                        influence = strength * np.exp(-(dist**2) / (2*1.5**2))
                        self.positional_field[fy,fx] += influence
    def update_combined_field(self):
        self.combined_field = (0.3*self.tactical_field*np.exp(1j*np.pi/3) +
                               0.5*self.positional_field +
                               0.2*self.strategic_field*np.exp(1j*2*np.pi/3))
    def get_spike_rate(self): return np.mean(self.spike_history) if self.spike_history else 0
class HybridBrainSystem:
    def __init__(self, grid_size=32):
        self.time_step = 0; self.grid_size = grid_size
        self.area_FEF = HybridEphapticField(grid_size, "FEF"); self.area_SEF = HybridEphapticField(grid_size, "SEF")
        self.area_PFC = HybridEphapticField(grid_size, "PFC"); self.area_Vision = HybridEphapticField(grid_size, "Vision")
        self.area_Vision.is_vision_area = True; self.vision_processor = VisionProcessor()
        self.inter_area_coupling = 0.05
    def get_all_areas(self): return [self.area_FEF, self.area_SEF, self.area_PFC, self.area_Vision]
    def step(self):
        vision_data, webcam_frame_gray = self.vision_processor.process_frame()
        self.area_Vision.apply_vision_input(vision_data)
        all_spikes = {area.area_name: area.update_neurons() for area in self.get_all_areas()}
        if self.time_step % 10 == 0:
            fef_g = laplace(np.abs(self.area_FEF.combined_field)); sef_g = laplace(np.abs(self.area_SEF.combined_field))
            pfc_g = laplace(np.abs(self.area_PFC.combined_field)); vis_g = laplace(np.abs(self.area_Vision.combined_field))
            self.area_SEF.positional_field += self.inter_area_coupling * fef_g
            self.area_PFC.positional_field += self.inter_area_coupling * sef_g
            self.area_FEF.positional_field += self.inter_area_coupling * pfc_g
            for area in self.get_all_areas(): area.update_combined_field()
        self.time_step += 1
        return all_spikes, webcam_frame_gray
# ==============================================================================
# PART 3: UNIFIED TKINTER APPLICATION
# ==============================================================================

class DreamViewerApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Unified AI Dream Viewer v2")
        self.root.geometry("1600x900")

        self.brain = HybridBrainSystem(grid_size=32)
        self.decoder = DecoderUNet(n_channels_in=4, n_channels_out=1).to(DEVICE)
        self.optimizer = optim.Adam(self.decoder.parameters(), lr=1e-4)
        self.loss_function = nn.MSELoss()

        self.running = False
        self.training_mode = True
        self.simulation_thread = None
        self.update_queue = queue.Queue()

        self.setup_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_gui(self):
        main_frame = ttk.Frame(self.root); main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        control_frame = ttk.LabelFrame(main_frame, text="Controls", width=300);
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5); control_frame.pack_propagate(False)
        brain_viz_frame = ttk.LabelFrame(main_frame, text="Brain Simulation");
        brain_viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        dream_frame = ttk.LabelFrame(main_frame, text="Vision & Dream Viewer", width=420)
        dream_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5); dream_frame.pack_propagate(False)

        self.populate_controls(control_frame)
        self.populate_brain_viz(brain_viz_frame)
        self.populate_dream_viewer(dream_frame)

    def populate_controls(self, parent):
        sys_frame = ttk.LabelFrame(parent, text="System"); sys_frame.pack(fill=tk.X, pady=5)
        self.start_button = ttk.Button(sys_frame, text="Start", command=self.toggle_simulation)
        self.start_button.pack(pady=2)
        vision_frame = ttk.LabelFrame(parent, text="Vision"); vision_frame.pack(fill=tk.X, pady=5)
        self.camera_var = tk.IntVar(value=0)
        cam_combo = ttk.Combobox(vision_frame, textvariable=self.camera_var, values=self.brain.vision_processor.available_cameras, state="readonly", width=15)
        cam_combo.pack(pady=2)
        if self.brain.vision_processor.available_cameras: cam_combo.set(self.brain.vision_processor.available_cameras[0])
        ttk.Button(vision_frame, text="Start Camera", command=lambda: self.brain.vision_processor.start_camera(self.camera_var.get())).pack()
        ttk.Button(vision_frame, text="Stop Camera", command=self.brain.vision_processor.stop_camera).pack()
        neuron_frame = ttk.LabelFrame(parent, text="Neuron Populations"); neuron_frame.pack(fill=tk.X, pady=5)
        self.neuron_controls = {}
        for area_name in ["FEF", "SEF", "PFC", "Vision"]:
            f = ttk.Frame(neuron_frame); f.pack(fill=tk.X)
            ttk.Label(f, text=f"{area_name}:", width=8).pack(side=tk.LEFT)
            var = tk.IntVar(value=50)
            area = next(a for a in self.brain.get_all_areas() if a.area_name == area_name)
            ttk.Scale(f, from_=0, to=200, variable=var, orient=tk.HORIZONTAL, command=lambda v, a=area: a.set_neuron_count(int(float(v)))).pack(side=tk.LEFT)
            self.neuron_controls[area_name] = var

    def populate_brain_viz(self, parent):
        self.brain_fig, self.brain_axes = plt.subplots(2, 2, figsize=(8, 8))
        self.brain_images, self.neuron_scatters = [], []
        for i, area in enumerate(self.brain.get_all_areas()):
            ax = self.brain_axes.flat[i]
            im = ax.imshow(np.abs(area.combined_field), cmap='viridis', origin='lower', vmin=0, vmax=1)
            scat = ax.scatter([], [], s=25, facecolors='none', edgecolors='red', linewidths=1.5)
            ax.set_title(area.area_name); ax.set_xticklabels([]); ax.set_yticklabels([])
            self.brain_images.append(im); self.neuron_scatters.append(scat)
        plt.tight_layout()
        self.brain_canvas = FigureCanvasTkAgg(self.brain_fig, parent)
        self.brain_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def populate_dream_viewer(self, parent):
        # --- FIX 2: Added a dedicated display for the live camera feed ---
        ttk.Label(parent, text="Live Camera Input", font=("Helvetica", 12, "bold")).pack(pady=(10, 2))
        self.camera_image_label = ttk.Label(parent)
        self.camera_image_label.pack(pady=5)
        self.camera_photo = None # To prevent garbage collection

        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=15)

        ttk.Label(parent, text="AI Decoder Output (Dream/Reconstruction)", font=("Helvetica", 12, "bold")).pack(pady=(10, 2))
        self.dream_image_label = ttk.Label(parent)
        self.dream_image_label.pack(pady=5)
        self.dream_photo = None

        self.mode_button = ttk.Button(parent, text="Switch to Dreaming Mode", command=self.toggle_mode)
        self.mode_button.pack(pady=20)
        
        self.status_mode_label = ttk.Label(parent, text="Mode: TRAINING", font=("Helvetica", 10))
        self.status_mode_label.pack()
        self.status_loss_label = ttk.Label(parent, text="Training Loss: N/A", font=("Helvetica", 10))
        self.status_loss_label.pack()
        # --- END FIX 2 ---

    def toggle_simulation(self):
        if not self.running:
            self.running = True
            self.start_button.config(text="Stop")
            self.simulation_thread = threading.Thread(target=self.simulation_and_training_loop, daemon=True)
            self.simulation_thread.start()
            self.periodic_gui_update()
        else:
            self.running = False; self.start_button.config(text="Start")

    def toggle_mode(self):
        self.training_mode = not self.training_mode
        mode_text = "TRAINING" if self.training_mode else "DREAMING"
        button_text = "Switch to Dreaming Mode" if self.training_mode else "Switch to Training Mode"
        self.status_mode_label.config(text=f"Mode: {mode_text}")
        self.mode_button.config(text=button_text)

    def simulation_and_training_loop(self):
        while self.running:
            start_time = time.time()
            spikes, webcam_frame_gray = self.brain.step()
            field_data = np.stack([np.abs(a.combined_field) for a in self.brain.get_all_areas()])
            field_tensor = torch.from_numpy(field_data).float().unsqueeze(0).to(DEVICE)
            decoded_image_np, loss_val = None, None

            if self.training_mode and webcam_frame_gray is not None:
                frame_resized = cv2.resize(webcam_frame_gray, (32, 32))
                target_tensor = torch.from_numpy(frame_resized/255.0).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
                self.decoder.train()
                self.optimizer.zero_grad()
                predicted = self.decoder(field_tensor)
                loss = self.loss_function(predicted, target_tensor)
                loss.backward(); self.optimizer.step()
                decoded_image_np, loss_val = predicted.squeeze().detach().cpu().numpy(), loss.item()
            elif not self.training_mode:
                self.decoder.eval()
                with torch.no_grad(): dream_tensor = self.decoder(field_tensor)
                decoded_image_np = dream_tensor.squeeze().detach().cpu().numpy()

            if self.brain.time_step % 2 == 0:
                 self.update_queue.put({
                    'fields': field_data, 'spikes': spikes, 'decoded_image': decoded_image_np,
                    'loss': loss_val, 'is_training': self.training_mode,
                    'camera_frame': self.brain.vision_processor.current_frame
                 })
            time.sleep(max(0, 0.1 - (time.time() - start_time)))

    def periodic_gui_update(self):
        try:
            while not self.update_queue.empty():
                data = self.update_queue.get_nowait()
                for i, area in enumerate(self.brain.get_all_areas()):
                    self.brain_images[i].set_array(data['fields'][i])
                    self.brain_images[i].set_clim(data['fields'][i].min(), data['fields'][i].max())
                    area_spikes = data['spikes'].get(area.area_name, [])
                    if area_spikes: self.neuron_scatters[i].set_offsets(np.array([[n.x, n.y] for n in area_spikes]))
                    else: self.neuron_scatters[i].set_offsets(np.empty((0,2)))
                self.brain_canvas.draw_idle()

                # --- FIX 2: Added logic to update the live camera feed label ---
                if data['camera_frame'] is not None:
                    img = Image.fromarray(data['camera_frame'])
                    img_resized = img.resize((400, 300), Image.LANCZOS)
                    self.camera_photo = ImageTk.PhotoImage(image=img_resized)
                    self.camera_image_label.config(image=self.camera_photo)
                # --- END FIX 2 ---

                if data['decoded_image'] is not None:
                    img_data = (data['decoded_image'] * 255).astype(np.uint8)
                    img = Image.fromarray(img_data).resize((384, 384), Image.NEAREST)
                    self.dream_photo = ImageTk.PhotoImage(image=img)
                    self.dream_image_label.config(image=self.dream_photo)
                if data['is_training'] and data['loss'] is not None: self.status_loss_label.config(text=f"Training Loss: {data['loss']:.6f}")
                elif not data['is_training']: self.status_loss_label.config(text="Training Loss: N/A")
        except queue.Empty: pass
        finally:
            if self.running: self.root.after(100, self.periodic_gui_update)

    def run(self): self.root.mainloop()
    def on_closing(self):
        self.running = False
        if self.simulation_thread: self.simulation_thread.join(timeout=1.0)
        self.brain.vision_processor.stop_camera()
        self.root.destroy()

if __name__ == "__main__":
    print("--- Starting Unified AI Dream Viewer Application v2 ---")
    print(f"--- Using processing device: {DEVICE} ---")
    if DEVICE == 'cpu': print("--- WARNING: AI training will be very slow on CPU. ---")
    app = DreamViewerApp()
    app.run()