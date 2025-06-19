import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle
import cv2
import threading
import time
from scipy.ndimage import laplace, sobel
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random

@dataclass
class DiscreteNeuron:
    """Individual neuron with membrane potential and ephaptic coupling"""
    x: float
    y: float
    membrane_potential: float = -70.0  # mV
    spike_threshold: float = -55.0      # mV
    refractory_period: int = 0
    spike_count: int = 0
    last_spike_time: int = 0
    area: str = "Brain"
    neuron_type: str = "excitatory"  # or "inhibitory"
    ephaptic_sensitivity: float = 0.1
    
    def update(self, local_field_value, vision_input=0.0, dt=1.0, noise_level=0.5):
        """Update neuron state based on local field, vision, and intrinsic dynamics"""
        if self.refractory_period > 0:
            self.refractory_period -= 1
            self.membrane_potential = -80.0  # Hyperpolarized during refractory
            return False
            
        # Ephaptic coupling from field
        ephaptic_current = self.ephaptic_sensitivity * local_field_value
        
        # Vision input (direct to neurons)
        vision_current = vision_input * 5.0  # Scale for mV
        
        # Leak current (return to rest)
        leak_current = (self.membrane_potential + 70.0) * 0.1
        
        # Noise
        noise = np.random.normal(0, noise_level)
        
        # Update membrane potential
        self.membrane_potential += dt * (-leak_current + ephaptic_current + vision_current + noise)
        
        # Check for spike
        if self.membrane_potential > self.spike_threshold:
            self.spike_count += 1
            self.last_spike_time = 0
            self.refractory_period = 3  # 3 timesteps refractory
            self.membrane_potential = -80.0
            return True
            
        self.last_spike_time += 1
        return False

class EphapticBrainArea:
    """Single brain area with ephaptic field and embedded neurons"""
    
    def __init__(self, grid_size=64, area_name="Brain", max_neurons=2000):
        self.grid_size = grid_size
        self.area_name = area_name
        self.max_neurons = max_neurons
        
        # Single unified neural field (no more chess complexity!)
        self.neural_field = self.generate_fractal_noise(grid_size, beta=1.0)
        self.field_memory = deque(maxlen=50)
        
        # Discrete neurons
        self.neurons: List[DiscreteNeuron] = []
        self.neuron_count = 0
        self.spike_history = deque(maxlen=1000)
        
        # Ephaptic coupling parameters
        self.ephaptic_strength = 0.2
        self.field_to_neuron_coupling = 0.1
        self.neuron_to_field_coupling = 0.05
        
        # Vision integration
        self.vision_coupling = 0.3
        self.current_vision = np.zeros((grid_size, grid_size))
        
        # Performance tracking
        self.step_count = 0
        
    def generate_fractal_noise(self, size, beta=1.0, seed=None):
        """Generate fractal noise with specified β exponent"""
        if seed is not None:
            np.random.seed(seed)
        kx = np.fft.fftfreq(size).reshape(1, -1)
        ky = np.fft.fftfreq(size).reshape(-1, 1)
        k = np.sqrt(kx*kx + ky*ky)
        k[0,0] = 1e-6
        spectrum = (k**(-beta/2)) * np.exp(2j*np.pi*np.random.rand(size, size))
        noise = np.real(np.fft.ifft2(spectrum))
        return (noise - noise.min()) / (noise.max() - noise.min())
        
    def set_neuron_count(self, count):
        """Set number of neurons in this area"""
        count = min(count, self.max_neurons)
        current_count = len(self.neurons)
        
        if count > current_count:
            # Add neurons
            for _ in range(count - current_count):
                x = np.random.uniform(2, self.grid_size-2)
                y = np.random.uniform(2, self.grid_size-2)
                neuron_type = "excitatory" if np.random.random() > 0.2 else "inhibitory"
                neuron = DiscreteNeuron(
                    x=x, y=y, 
                    area=self.area_name,
                    neuron_type=neuron_type,
                    ephaptic_sensitivity=0.1 if neuron_type == "excitatory" else -0.05
                )
                self.neurons.append(neuron)
        elif count < current_count:
            # Remove neurons
            self.neurons = self.neurons[:count]
            
        self.neuron_count = len(self.neurons)
        
    def apply_vision_input(self, vision_data):
        """Apply vision input to the neural field"""
        if vision_data.shape != (self.grid_size, self.grid_size):
            self.current_vision = cv2.resize(vision_data, (self.grid_size, self.grid_size))
        else:
            self.current_vision = vision_data.copy()
            
        # Add vision to neural field
        self.neural_field += self.vision_coupling * self.current_vision * 0.1
        
    def update_neurons(self, dt=1.0):
        """Update all neurons based on local field values and vision"""
        spike_count = 0
        
        for neuron in self.neurons:
            # Sample local field value at neuron position
            grid_x = int(np.clip(neuron.x, 0, self.grid_size-1))
            grid_y = int(np.clip(neuron.y, 0, self.grid_size-1))
            
            local_field = self.neural_field[grid_y, grid_x] * self.field_to_neuron_coupling
            vision_input = self.current_vision[grid_y, grid_x] * self.vision_coupling
            
            # Update neuron
            if neuron.update(local_field, vision_input, dt):
                spike_count += 1
                # Neuron spiked - influence local field
                self.apply_neuron_spike_to_field(neuron)
                
        self.spike_history.append(spike_count)
        return spike_count
        
    def apply_neuron_spike_to_field(self, neuron):
        """Apply neuron spike influence to surrounding field"""
        grid_x = int(np.clip(neuron.x, 0, self.grid_size-1))
        grid_y = int(np.clip(neuron.y, 0, self.grid_size-1))
        
        # Create Gaussian influence around spike location
        influence_radius = 3
        spike_strength = 0.1 if neuron.neuron_type == "excitatory" else -0.05
        
        for dx in range(-influence_radius, influence_radius+1):
            for dy in range(-influence_radius, influence_radius+1):
                fx = grid_x + dx
                fy = grid_y + dy
                
                if 0 <= fx < self.grid_size and 0 <= fy < self.grid_size:
                    distance = np.sqrt(dx*dx + dy*dy)
                    if distance <= influence_radius:
                        influence = spike_strength * np.exp(-(distance**2) / (2 * 1.5**2))
                        self.neural_field[fy, fx] += influence
        
    def ephaptic_feedback(self):
        """Apply ephaptic feedback from field to field"""
        field_influence = laplace(self.neural_field)
        
        if np.std(field_influence) > 1e-9:
            field_influence = field_influence / np.std(field_influence) * 0.1
            
        self.neural_field += self.ephaptic_strength * field_influence
        
        # Store in memory for analysis
        self.field_memory.append(self.neural_field.copy())
        
    def get_spike_rate(self, window_size=50):
        """Get recent spike rate"""
        if len(self.spike_history) < window_size:
            return 0.0
        return np.mean(list(self.spike_history)[-window_size:])
        
    def step(self):
        """Single simulation step"""
        # Update neurons
        spikes = self.update_neurons()
        
        # Ephaptic feedback every 100 steps
        if self.step_count % 100 == 0:
            self.ephaptic_feedback()
            
        # Field decay to prevent runaway
        self.neural_field *= 0.999
        
        self.step_count += 1
        return spikes

class VisionProcessor:
    """Enhanced vision processing for dream mode"""
    
    def __init__(self, target_size=(64, 64)):
        self.target_size = target_size
        self.cap = None
        self.current_frame = None
        self.processed_vision = np.zeros(target_size)
        self.predicted_vision = np.zeros(target_size)  # For dream mode
        self.camera_active = False
        self.dream_mode = False
        self.available_cameras = self.detect_cameras()
        
        # Dream mode parameters
        self.dream_memory = deque(maxlen=100)  # Store recent vision for prediction
        
    def detect_cameras(self):
        """Detect available cameras"""
        cameras = []
        for i in range(3):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        cameras.append(i)
                cap.release()
            except Exception:
                pass
        return cameras if cameras else [0]
        
    def start_camera(self, camera_index=0):
        """Start webcam capture"""
        try:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(camera_index)
            if self.cap.isOpened():
                self.camera_active = True
                return True
        except Exception as e:
            print(f"Camera error: {e}")
        return False
        
    def stop_camera(self):
        """Stop webcam capture"""
        self.camera_active = False
        if self.cap:
            self.cap.release()
            
    def set_dream_mode(self, dream_on):
        """Enable/disable dream mode"""
        self.dream_mode = dream_on
        
    def process_frame(self):
        """Process webcam frame or generate dream prediction"""
        if self.dream_mode:
            # Dream mode - predict next frame based on history
            return self.generate_dream_prediction()
        else:
            # Normal mode - process real webcam
            return self.process_real_frame()
            
    def process_real_frame(self):
        """Process real webcam frame"""
        if not self.camera_active or not self.cap:
            return self.processed_vision
            
        try:
            ret, frame = self.cap.read()
            if ret:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Resize to target size
                resized = cv2.resize(gray, self.target_size)
                
                # Normalize to 0-1 range
                self.processed_vision = resized.astype(np.float32) / 255.0
                
                # Apply edge detection for more neural-relevant features
                edges = cv2.Sobel(resized, cv2.CV_64F, 1, 1, ksize=3)
                edge_strength = np.abs(edges) / 255.0
                
                # Combine brightness and edge information
                self.processed_vision = 0.7 * self.processed_vision + 0.3 * edge_strength
                
                # Store in dream memory
                self.dream_memory.append(self.processed_vision.copy())
                
                self.current_frame = frame
                
        except Exception as e:
            print(f"Frame processing error: {e}")
            
        return self.processed_vision
        
    def generate_dream_prediction(self):
        """Generate predicted vision based on recent history"""
        if len(self.dream_memory) < 5:
            # Not enough history, return noise
            self.predicted_vision = np.random.rand(*self.target_size) * 0.1
        else:
            # Simple prediction: weighted average of recent frames with trend
            recent = list(self.dream_memory)[-5:]
            weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])  # More weight on recent
            
            prediction = np.zeros(self.target_size)
            for i, frame in enumerate(recent):
                prediction += weights[i] * frame
                
            # Add trend from last two frames
            if len(recent) >= 2:
                trend = recent[-1] - recent[-2]
                prediction += 0.3 * trend
                
            # Add some neural-like noise
            prediction += 0.05 * np.random.randn(*self.target_size)
            
            # Clip to valid range
            self.predicted_vision = np.clip(prediction, 0, 1)
            
        return self.predicted_vision

class StreamlinedBrainSystem:
    """Streamlined brain system with single area and vision"""
    
    def __init__(self, grid_size=64):
        self.grid_size = grid_size
        self.time_step = 0
        
        # Single brain area
        self.brain_area = EphapticBrainArea(grid_size, "Brain", max_neurons=2000)
        
        # Vision system
        self.vision_processor = VisionProcessor(target_size=(grid_size, grid_size))
        
        # System state
        self.running = False
        self.total_spikes = 0
        
    def step(self):
        """Single simulation step"""
        # Process vision input
        vision_data = self.vision_processor.process_frame()
        self.brain_area.apply_vision_input(vision_data)
        
        # Update brain
        spikes = self.brain_area.step()
        self.total_spikes += spikes
        
        self.time_step += 1
        return spikes

class StreamlinedBrainGUI:
    """Minimal GUI focused on vision and neuron control"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Streamlined Ephaptic Brain Computer")
        self.root.geometry("1200x800")
        
        # Brain system
        self.brain = StreamlinedBrainSystem(grid_size=64)
        
        # GUI state
        self.running = False
        self.update_thread = None
        self.fps_target = 60  # Much faster without complex visualization
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup minimal GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,5))
        control_frame.pack_propagate(False)
        
        # Right panel - Vision
        vision_frame = ttk.LabelFrame(main_frame, text="Vision")
        vision_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_controls(control_frame)
        self.setup_vision_panel(vision_frame)
        
    def setup_controls(self, parent):
        """Setup control panel"""
        # System controls
        sys_frame = ttk.LabelFrame(parent, text="System")
        sys_frame.pack(fill=tk.X, pady=5)
        
        self.start_button = ttk.Button(sys_frame, text="Start", command=self.toggle_simulation)
        self.start_button.pack(pady=2)
        
        # Vision controls
        vision_frame = ttk.LabelFrame(parent, text="Vision System")
        vision_frame.pack(fill=tk.X, pady=5)
        
        # Camera selection
        ttk.Label(vision_frame, text="Camera:").pack()
        self.camera_var = tk.IntVar()
        available_cameras = self.brain.vision_processor.available_cameras
        camera_combo = ttk.Combobox(vision_frame, textvariable=self.camera_var, 
                                   values=available_cameras if available_cameras else [0],
                                   state="readonly", width=15)
        camera_combo.pack(pady=2)
        if available_cameras:
            camera_combo.set(available_cameras[0])
            self.camera_var.set(available_cameras[0])
        else:
            camera_combo.set(0)
            self.camera_var.set(0)
            
        ttk.Button(vision_frame, text="Start Camera", 
                  command=self.start_camera).pack(pady=2)
        ttk.Button(vision_frame, text="Stop Camera", 
                  command=self.stop_camera).pack(pady=2)
        
        # Dream mode
        self.dream_var = tk.BooleanVar()
        ttk.Checkbutton(vision_frame, text="Dream Mode", 
                       variable=self.dream_var,
                       command=self.toggle_dream_mode).pack(pady=2)
        
        # Neuron count control - MUCH HIGHER MAXIMUM
        neuron_frame = ttk.LabelFrame(parent, text="Neural Population")
        neuron_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(neuron_frame, text="Neuron Count:").pack()
        
        self.neuron_var = tk.IntVar(value=500)
        neuron_scale = ttk.Scale(neuron_frame, from_=0, to=2000, variable=self.neuron_var, 
                               orient=tk.HORIZONTAL, length=250,
                               command=lambda val: self.update_neuron_count(int(float(val))))
        neuron_scale.pack(pady=5)
        
        neuron_spinbox = ttk.Spinbox(neuron_frame, from_=0, to=2000, textvariable=self.neuron_var, 
                                   width=10, command=lambda: self.update_neuron_count(self.neuron_var.get()))
        neuron_spinbox.pack(pady=2)
        
        # Coupling controls
        coupling_frame = ttk.LabelFrame(parent, text="Neural Parameters")
        coupling_frame.pack(fill=tk.X, pady=5)
        
        # Ephaptic coupling
        ttk.Label(coupling_frame, text="Ephaptic Strength:").pack()
        self.ephaptic_var = tk.DoubleVar(value=0.2)
        ttk.Scale(coupling_frame, from_=0.0, to=1.0, variable=self.ephaptic_var,
                 orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # Vision coupling
        ttk.Label(coupling_frame, text="Vision Coupling:").pack()
        self.vision_coupling_var = tk.DoubleVar(value=0.3)
        ttk.Scale(coupling_frame, from_=0.0, to=2.0, variable=self.vision_coupling_var,
                 orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # Status display
        status_frame = ttk.LabelFrame(parent, text="Status")
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_text = tk.Text(status_frame, height=12, width=35, font=('Courier', 8))
        self.status_text.pack(fill=tk.BOTH, expand=True)
        
    def setup_vision_panel(self, parent):
        """Setup vision visualization panel"""
        self.vision_fig, (self.camera_ax, self.processed_ax) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Camera feed
        self.camera_ax.set_title("Camera / Dream Feed")
        self.camera_ax.axis('off')
        
        # Processed vision data
        self.processed_ax.set_title("Neural Vision Input")
        self.processed_vision_im = self.processed_ax.imshow(
            np.zeros((64, 64)), cmap='viridis', origin='lower'
        )
        self.processed_ax.axis('off')
        
        plt.tight_layout()
        
        self.vision_canvas = FigureCanvasTkAgg(self.vision_fig, parent)
        self.vision_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def update_neuron_count(self, count):
        """Update neuron count"""
        try:
            self.brain.brain_area.set_neuron_count(int(count))
        except Exception as e:
            print(f"Error updating neuron count: {e}")
            
    def start_camera(self):
        """Start camera capture"""
        try:
            camera_index = self.camera_var.get()
            if self.brain.vision_processor.start_camera(camera_index):
                self.update_status("Camera started successfully")
            else:
                self.update_status("Failed to start camera")
        except Exception as e:
            self.update_status(f"Camera error: {e}")
            
    def stop_camera(self):
        """Stop camera capture"""
        self.brain.vision_processor.stop_camera()
        self.update_status("Camera stopped")
        
    def toggle_dream_mode(self):
        """Toggle dream mode"""
        dream_on = self.dream_var.get()
        self.brain.vision_processor.set_dream_mode(dream_on)
        self.update_status(f"Dream mode: {'ON' if dream_on else 'OFF'}")
        
    def toggle_simulation(self):
        """Start/stop simulation"""
        if not self.running:
            self.running = True
            self.start_button.config(text="Stop")
            self.update_thread = threading.Thread(target=self.simulation_loop, daemon=True)
            self.update_thread.start()
            self.update_status("Simulation started")
        else:
            self.running = False
            self.start_button.config(text="Start")
            self.update_status("Simulation stopped")
            
    def simulation_loop(self):
        """High-speed simulation loop"""
        while self.running:
            start_time = time.time()
            
            # Update parameters
            self.brain.brain_area.ephaptic_strength = self.ephaptic_var.get()
            self.brain.brain_area.vision_coupling = self.vision_coupling_var.get()
            
            # Run multiple simulation steps per frame for speed
            for _ in range(10):  # 10x faster than old system
                if self.running:
                    self.brain.step()
            
            # Update visualization every few cycles
            if self.brain.time_step % 30 == 0:  # Update every 30 steps
                self.root.after(0, self.update_visualization)
                
            # Control frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, 1.0/self.fps_target - elapsed)
            time.sleep(sleep_time)
            
    def update_visualization(self):
        """Update vision displays"""
        try:
            # Update camera/dream display
            if self.brain.vision_processor.dream_mode:
                # Show predicted vision
                predicted = self.brain.vision_processor.predicted_vision
                self.camera_ax.clear()
                self.camera_ax.imshow(predicted, cmap='viridis', origin='lower')
                self.camera_ax.set_title("Dream Prediction")
                self.camera_ax.axis('off')
            else:
                # Show real camera feed
                if self.brain.vision_processor.current_frame is not None:
                    frame_rgb = cv2.cvtColor(self.brain.vision_processor.current_frame, cv2.COLOR_BGR2RGB)
                    self.camera_ax.clear()
                    self.camera_ax.imshow(frame_rgb)
                    self.camera_ax.set_title("Camera Feed")
                    self.camera_ax.axis('off')
            
            # Show processed vision going to neurons
            processed = self.brain.vision_processor.processed_vision
            self.processed_vision_im.set_array(processed)
            self.processed_vision_im.set_clim(0, 1)
            
            self.vision_canvas.draw_idle()
            
            # Update status
            spike_rate = self.brain.brain_area.get_spike_rate()
            neuron_count = len(self.brain.brain_area.neurons)
            
            status = f"""Time: {self.brain.time_step}
Neurons: {neuron_count}
Spike Rate: {spike_rate:.1f} Hz
Total Spikes: {self.brain.total_spikes}

Ephaptic Strength: {self.brain.brain_area.ephaptic_strength:.3f}
Vision Coupling: {self.brain.brain_area.vision_coupling:.3f}

Camera: {'Active' if self.brain.vision_processor.camera_active else 'Inactive'}
Dream Mode: {'ON' if self.brain.vision_processor.dream_mode else 'OFF'}

Field Stats:
  Mean: {np.mean(self.brain.brain_area.neural_field):.3f}
  Std: {np.std(self.brain.brain_area.neural_field):.3f}
  Max: {np.max(self.brain.brain_area.neural_field):.3f}
  Min: {np.min(self.brain.brain_area.neural_field):.3f}"""

            self.status_text.delete(1.0, tk.END)
            self.status_text.insert(1.0, status)
            
        except Exception as e:
            print(f"Visualization update error: {e}")
            
    def update_status(self, message):
        """Update status message"""
        print(f"Status: {message}")
        
    def run(self):
        """Start the GUI"""
        # Initialize with default neuron count
        self.update_neuron_count(self.neuron_var.get())
        
        self.update_status("GUI ready - click Start to begin simulation")
        
        # Start GUI
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
        
    def on_closing(self):
        """Handle application closing"""
        self.running = False
        self.brain.vision_processor.stop_camera()
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)
        self.root.destroy()

if __name__ == "__main__":
    print("🧠⚡ Launching Streamlined Ephaptic Brain Computer ⚡🧠")
    print("Features:")
    print("• Single unified neural field (no chess complexity)")
    print("• Up to 2000 neurons with direct vision input")
    print("• Dream mode for predicted vision")
    print("• High-speed simulation (10x faster)")
    print("• Real-time ephaptic coupling dynamics")
    print()
    
    try:
        app = StreamlinedBrainGUI()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Error", f"Application failed to start: {e}")