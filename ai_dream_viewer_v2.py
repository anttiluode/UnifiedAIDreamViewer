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
    area: str = "Vision"
    neuron_type: str = "excitatory"  # or "inhibitory"
    ephaptic_sensitivity: float = 0.1
    
    def update(self, local_field_value, external_input=0.0, dt=1.0, noise_level=0.5):
        """Update neuron state based on local field and external input"""
        if self.refractory_period > 0:
            self.refractory_period -= 1
            self.membrane_potential = -80.0  # Hyperpolarized during refractory
            return False
            
        # Ephaptic coupling from field
        ephaptic_current = self.ephaptic_sensitivity * local_field_value
        
        # External input (vision or top-down feedback)
        external_current = external_input * 3.0  # Scale for mV
        
        # Leak current (return to rest)
        leak_current = (self.membrane_potential + 70.0) * 0.1
        
        # Noise
        noise = np.random.normal(0, noise_level)
        
        # Update membrane potential
        self.membrane_potential += dt * (-leak_current + ephaptic_current + external_current + noise)
        
        # Check for spike
        if self.membrane_potential > self.spike_threshold:
            self.spike_count += 1
            self.last_spike_time = 0
            self.refractory_period = 3
            self.membrane_potential = -80.0
            return True
            
        self.last_spike_time += 1
        return False

class HierarchicalBrainArea:
    """Brain area in hierarchical processing chain with feedback"""
    
    def __init__(self, grid_size=64, area_name="Vision", max_neurons=500, hierarchy_level=0):
        self.grid_size = grid_size
        self.area_name = area_name
        self.max_neurons = max_neurons
        self.hierarchy_level = hierarchy_level  # 0=Vision, 1=FEF, 2=SEF, 3=PFC
        
        # Single neural field for this area
        self.neural_field = self.generate_fractal_noise(grid_size, beta=0.8 + hierarchy_level*0.3)
        self.field_memory = deque(maxlen=50)
        
        # Discrete neurons
        self.neurons: List[DiscreteNeuron] = []
        self.neuron_count = 0
        self.spike_history = deque(maxlen=1000)
        
        # Ephaptic coupling parameters
        self.ephaptic_strength = 0.15 + hierarchy_level * 0.05  # Higher areas have stronger coupling
        self.field_to_neuron_coupling = 0.1
        self.neuron_to_field_coupling = 0.05
        
        # Hierarchical connections
        self.bottom_up_input = np.zeros((grid_size, grid_size))
        self.top_down_input = np.zeros((grid_size, grid_size))
        self.bottom_up_strength = 0.3
        self.top_down_strength = 0.2 + hierarchy_level * 0.1  # Higher areas have stronger top-down
        
        # Processing degradation (higher areas add more noise/transformation)
        self.processing_noise = 0.1 * (hierarchy_level + 1)
        self.temporal_integration = max(1, hierarchy_level)  # Higher areas integrate over more time
        
        self.step_count = 0
        
    def generate_fractal_noise(self, size, beta=1.0, seed=None):
        """Generate fractal noise with hierarchy-specific characteristics"""
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
            self.neurons = self.neurons[:count]
            
        self.neuron_count = len(self.neurons)
        
    def apply_external_input(self, input_data):
        """Apply external input (vision for lowest level, or top-down feedback)"""
        if input_data.shape != (self.grid_size, self.grid_size):
            resized_input = cv2.resize(input_data, (self.grid_size, self.grid_size))
        else:
            resized_input = input_data.copy()
            
        # Add to neural field
        self.neural_field += resized_input * 0.1
        
    def set_bottom_up_input(self, input_field):
        """Receive bottom-up input from lower area"""
        if input_field.shape == (self.grid_size, self.grid_size):
            # Add processing degradation and transformation
            degraded = input_field + np.random.normal(0, self.processing_noise, input_field.shape)
            
            # Temporal integration for higher areas
            if self.temporal_integration > 1:
                # Simple temporal smoothing
                self.bottom_up_input = 0.7 * self.bottom_up_input + 0.3 * degraded
            else:
                self.bottom_up_input = degraded
                
    def set_top_down_input(self, input_field):
        """Receive top-down input from higher area"""
        if input_field.shape == (self.grid_size, self.grid_size):
            # Top-down predictions are more stable but degraded
            prediction = input_field + np.random.normal(0, self.processing_noise * 0.5, input_field.shape)
            self.top_down_input = prediction
            
    def generate_top_down_prediction(self):
        """Generate top-down prediction to send to lower areas"""
        # Higher areas generate predictions based on their current state
        prediction = self.neural_field.copy()
        
        # Add hierarchical transformation (higher areas predict simpler patterns)
        if self.hierarchy_level > 0:
            # Apply smoothing - higher areas predict broader patterns
            from scipy.ndimage import gaussian_filter
            prediction = gaussian_filter(prediction, sigma=self.hierarchy_level * 0.5)
            
        # Add prediction noise
        prediction += np.random.normal(0, 0.1, prediction.shape)
        
        return prediction
        
    def update_neurons(self, dt=1.0):
        """Update all neurons based on local field and hierarchical inputs"""
        spike_count = 0
        
        for neuron in self.neurons:
            # Sample local field value at neuron position
            grid_x = int(np.clip(neuron.x, 0, self.grid_size-1))
            grid_y = int(np.clip(neuron.y, 0, self.grid_size-1))
            
            local_field = self.neural_field[grid_y, grid_x] * self.field_to_neuron_coupling
            
            # Combine bottom-up and top-down inputs
            bottom_up = self.bottom_up_input[grid_y, grid_x] * self.bottom_up_strength
            top_down = self.top_down_input[grid_y, grid_x] * self.top_down_strength
            
            external_input = bottom_up + top_down
            
            # Update neuron
            if neuron.update(local_field, external_input, dt):
                spike_count += 1
                self.apply_neuron_spike_to_field(neuron)
                
        self.spike_history.append(spike_count)
        return spike_count
        
    def apply_neuron_spike_to_field(self, neuron):
        """Apply neuron spike influence to surrounding field"""
        grid_x = int(np.clip(neuron.x, 0, self.grid_size-1))
        grid_y = int(np.clip(neuron.y, 0, self.grid_size-1))
        
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
        """Apply ephaptic feedback within this area"""
        field_influence = laplace(self.neural_field)
        
        if np.std(field_influence) > 1e-9:
            field_influence = field_influence / np.std(field_influence) * 0.1
            
        self.neural_field += self.ephaptic_strength * field_influence
        
        # Store in memory
        self.field_memory.append(self.neural_field.copy())
        
    def step(self):
        """Single simulation step for this area"""
        # Apply hierarchical inputs to neural field
        self.neural_field += self.bottom_up_strength * self.bottom_up_input * 0.05
        self.neural_field += self.top_down_strength * self.top_down_input * 0.05
        
        # Update neurons
        spikes = self.update_neurons()
        
        # Ephaptic feedback every 100 steps
        if self.step_count % 100 == 0:
            self.ephaptic_feedback()
            
        # Field decay to prevent runaway
        self.neural_field *= 0.999
        
        # Reset inputs for next step
        self.bottom_up_input *= 0.9  # Gradual decay
        self.top_down_input *= 0.9
        
        self.step_count += 1
        return spikes
        
    def get_spike_rate(self, window_size=50):
        """Get recent spike rate"""
        if len(self.spike_history) < window_size:
            return 0.0
        return np.mean(list(self.spike_history)[-window_size:])

class VisionProcessor:
    """Vision processing with memory for hierarchical feedback"""
    
    def __init__(self, target_size=(64, 64)):
        self.target_size = target_size
        self.cap = None
        self.current_frame = None
        self.processed_vision = np.zeros(target_size)
        self.camera_active = False
        self.available_cameras = self.detect_cameras()
        
        # Vision memory for temporal processing
        self.vision_memory = deque(maxlen=20)
        
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
            
    def process_frame(self):
        """Process webcam frame"""
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
                
                # Apply edge detection
                edges = cv2.Sobel(resized, cv2.CV_64F, 1, 1, ksize=3)
                edge_strength = np.abs(edges) / 255.0
                
                # Combine brightness and edge information
                self.processed_vision = 0.6 * self.processed_vision + 0.4 * edge_strength
                
                # Store in memory
                self.vision_memory.append(self.processed_vision.copy())
                
                self.current_frame = frame
                
        except Exception as e:
            print(f"Frame processing error: {e}")
            
        return self.processed_vision

class HierarchicalBrainSystem:
    """Hierarchical brain system with feedback loops"""
    
    def __init__(self, grid_size=64):
        self.grid_size = grid_size
        self.time_step = 0
        
        # Hierarchical brain areas
        self.vision_area = HierarchicalBrainArea(grid_size, "Vision", max_neurons=500, hierarchy_level=0)
        self.fef_area = HierarchicalBrainArea(grid_size, "FEF", max_neurons=400, hierarchy_level=1)
        self.sef_area = HierarchicalBrainArea(grid_size, "SEF", max_neurons=300, hierarchy_level=2)
        self.pfc_area = HierarchicalBrainArea(grid_size, "PFC", max_neurons=200, hierarchy_level=3)
        
        # Vision system
        self.vision_processor = VisionProcessor(target_size=(grid_size, grid_size))
        
        # System state
        self.running = False
        self.dream_mode = False
        self.total_spikes = 0
        
        # Degradation parameters for feedback loops
        self.bottom_up_degradation = 0.1
        self.top_down_degradation = 0.15
        
    def get_all_areas(self):
        """Get all brain areas in hierarchical order"""
        return [self.vision_area, self.fef_area, self.sef_area, self.pfc_area]
        
    def set_dream_mode(self, dream_on):
        """Enable/disable dream mode"""
        self.dream_mode = dream_on
        
    def apply_hierarchical_connections(self):
        """Apply bottom-up and top-down connections between areas"""
        areas = self.get_all_areas()
        
        # Bottom-up connections (with degradation)
        for i in range(len(areas) - 1):
            lower_area = areas[i]
            higher_area = areas[i + 1]
            
            # Send degraded signal up the hierarchy
            bottom_up_signal = lower_area.neural_field.copy()
            bottom_up_signal += np.random.normal(0, self.bottom_up_degradation, bottom_up_signal.shape)
            higher_area.set_bottom_up_input(bottom_up_signal)
        
        # Top-down connections (predictions with degradation)
        for i in range(len(areas) - 1, 0, -1):
            higher_area = areas[i]
            lower_area = areas[i - 1]
            
            # Send prediction down the hierarchy
            top_down_prediction = higher_area.generate_top_down_prediction()
            top_down_prediction += np.random.normal(0, self.top_down_degradation, top_down_prediction.shape)
            lower_area.set_top_down_input(top_down_prediction)
        
    def step(self):
        """Single simulation step with hierarchical processing"""
        # In normal mode, apply vision input
        if not self.dream_mode:
            vision_data = self.vision_processor.process_frame()
            self.vision_area.apply_external_input(vision_data)
        else:
            # In dream mode, PFC drives the entire hierarchy
            # PFC generates spontaneous activity that flows down
            dream_seed = np.random.normal(0, 0.2, (self.grid_size, self.grid_size))
            self.pfc_area.apply_external_input(dream_seed)
        
        # Apply hierarchical connections
        self.apply_hierarchical_connections()
        
        # Update all areas
        total_spikes = 0
        for area in self.get_all_areas():
            spikes = area.step()
            total_spikes += spikes
            
        self.total_spikes += total_spikes
        self.time_step += 1
        
        return total_spikes

class HierarchicalBrainGUI:
    """GUI for hierarchical brain system with feedback visualization"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hierarchical Feedback Ephaptic Brain")
        self.root.geometry("1400x900")
        
        # Brain system
        self.brain = HierarchicalBrainSystem(grid_size=64)
        
        # GUI state
        self.running = False
        self.update_thread = None
        self.fps_target = 30
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup GUI layout"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", width=280)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,5))
        control_frame.pack_propagate(False)
        
        # Right panel - Hierarchical visualization
        viz_frame = ttk.LabelFrame(main_frame, text="Hierarchical Brain Areas")
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_controls(control_frame)
        self.setup_hierarchical_visualization(viz_frame)
        
    def setup_controls(self, parent):
        """Setup control panel"""
        # System controls
        sys_frame = ttk.LabelFrame(parent, text="System")
        sys_frame.pack(fill=tk.X, pady=5)
        
        self.start_button = ttk.Button(sys_frame, text="Start", command=self.toggle_simulation)
        self.start_button.pack(pady=2)
        
        # Dream mode
        self.dream_var = tk.BooleanVar()
        ttk.Checkbutton(sys_frame, text="Dream Mode (PFC-driven)", 
                       variable=self.dream_var,
                       command=self.toggle_dream_mode).pack(pady=2)
        
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
        
        ttk.Button(vision_frame, text="Start Camera", 
                  command=self.start_camera).pack(pady=2)
        ttk.Button(vision_frame, text="Stop Camera", 
                  command=self.stop_camera).pack(pady=2)
        
        # Neuron population controls
        neuron_frame = ttk.LabelFrame(parent, text="Neural Populations")
        neuron_frame.pack(fill=tk.X, pady=5)
        
        self.neuron_controls = {}
        areas = ["Vision", "FEF", "SEF", "PFC"]
        max_neurons = [500, 400, 300, 200]
        
        for area, max_n in zip(areas, max_neurons):
            area_frame = ttk.Frame(neuron_frame)
            area_frame.pack(fill=tk.X, pady=1)
            
            ttk.Label(area_frame, text=f"{area}:", width=8).pack(side=tk.LEFT)
            
            var = tk.IntVar(value=max_n//2)
            scale = ttk.Scale(area_frame, from_=0, to=max_n, variable=var, 
                             orient=tk.HORIZONTAL, length=120)
            scale.pack(side=tk.LEFT, padx=2)
            
            spinbox = ttk.Spinbox(area_frame, from_=0, to=max_n, textvariable=var, 
                                 width=5, command=lambda a=area, v=var: self.update_neuron_count(a, v.get()))
            spinbox.pack(side=tk.LEFT, padx=2)
            
            self.neuron_controls[area] = var
            var.trace('w', lambda *args, a=area, v=var: self.update_neuron_count(a, v.get()))
        
        # Hierarchical parameters
        hier_frame = ttk.LabelFrame(parent, text="Hierarchical Parameters")
        hier_frame.pack(fill=tk.X, pady=5)
        
        # Bottom-up degradation
        ttk.Label(hier_frame, text="Bottom-up Degradation:").pack()
        self.bottomup_var = tk.DoubleVar(value=0.1)
        ttk.Scale(hier_frame, from_=0.0, to=0.5, variable=self.bottomup_var,
                 orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # Top-down degradation
        ttk.Label(hier_frame, text="Top-down Degradation:").pack()
        self.topdown_var = tk.DoubleVar(value=0.15)
        ttk.Scale(hier_frame, from_=0.0, to=0.5, variable=self.topdown_var,
                 orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # Status display
        status_frame = ttk.LabelFrame(parent, text="Status")
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_text = tk.Text(status_frame, height=8, width=32, font=('Courier', 8))
        self.status_text.pack(fill=tk.BOTH, expand=True)
        
    def setup_hierarchical_visualization(self, parent):
        """Setup hierarchical brain area visualization"""
        self.hier_fig, self.hier_axes = plt.subplots(2, 2, figsize=(12, 10))
        self.hier_fig.suptitle("Hierarchical Ephaptic Brain - Feedback Loops")
        
        # Initialize plots for each area
        areas = self.brain.get_all_areas()
        area_names = ["Vision (Level 0)", "FEF (Level 1)", "SEF (Level 2)", "PFC (Level 3)"]
        cmaps = ['viridis', 'plasma', 'inferno', 'magma']
        
        self.area_images = []
        
        for i, (area, name, cmap) in enumerate(zip(areas, area_names, cmaps)):
            row, col = i // 2, i % 2
            ax = self.hier_axes[row, col]
            
            # Field visualization
            field_data = area.neural_field
            im = ax.imshow(field_data, cmap=cmap, origin='lower')
            self.area_images.append(im)
            
            ax.set_title(f"{name}")
            ax.set_xlim(0, area.grid_size)
            ax.set_ylim(0, area.grid_size)
            ax.axis('off')
            
        plt.tight_layout()
        
        self.hier_canvas = FigureCanvasTkAgg(self.hier_fig, parent)
        self.hier_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def update_neuron_count(self, area_name, count):
        """Update neuron count for specific area"""
        area_map = {
            "Vision": self.brain.vision_area,
            "FEF": self.brain.fef_area,
            "SEF": self.brain.sef_area,
            "PFC": self.brain.pfc_area
        }
        
        if area_name in area_map:
            try:
                area_map[area_name].set_neuron_count(int(count))
            except Exception as e:
                print(f"Error updating {area_name} neuron count: {e}")
                
    def start_camera(self):
        """Start camera capture"""
        try:
            camera_index = self.camera_var.get()
            if self.brain.vision_processor.start_camera(camera_index):
                self.update_status("Camera started")
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
        self.brain.set_dream_mode(dream_on)
        self.update_status(f"Dream mode: {'ON - PFC driving' if dream_on else 'OFF - Vision driving'}")
        
    def toggle_simulation(self):
        """Start/stop simulation"""
        if not self.running:
            self.running = True
            self.start_button.config(text="Stop")
            self.update_thread = threading.Thread(target=self.simulation_loop, daemon=True)
            self.update_thread.start()
            self.update_status("Hierarchical simulation started")
        else:
            self.running = False
            self.start_button.config(text="Start")
            self.update_status("Simulation stopped")
            
    def simulation_loop(self):
        """High-speed hierarchical simulation loop"""
        while self.running:
            start_time = time.time()
            
            # Update parameters
            self.brain.bottom_up_degradation = self.bottomup_var.get()
            self.brain.top_down_degradation = self.topdown_var.get()
            
            # Run multiple simulation steps
            for _ in range(5):  # 5 steps per frame
                if self.running:
                    self.brain.step()
            
            # Update visualization
            if self.brain.time_step % 15 == 0:  # Update every 15 steps
                self.root.after(0, self.update_visualization)
                
            # Control frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, 1.0/self.fps_target - elapsed)
            time.sleep(sleep_time)
            
    def update_visualization(self):
        """Update hierarchical visualization"""
        try:
            # Update all area displays
            areas = self.brain.get_all_areas()
            area_names = ["Vision", "FEF", "SEF", "PFC"]
            
            for i, (area, name) in enumerate(zip(areas, area_names)):
                # Update field display
                field_data = area.neural_field
                self.area_images[i].set_array(field_data)
                self.area_images[i].set_clim(field_data.min(), field_data.max())
                
                # Update title with neuron count and spike rate
                spike_rate = area.get_spike_rate()
                row, col = i // 2, i % 2
                mode_indicator = "🧠" if not self.brain.dream_mode else "💭"
                level_indicator = f"L{area.hierarchy_level}"
                self.hier_axes[row, col].set_title(
                    f"{mode_indicator} {name} {level_indicator} ({len(area.neurons)}n, {spike_rate:.1f}Hz)"
                )
            
            self.hier_canvas.draw_idle()
            
            # Update status
            areas = self.brain.get_all_areas()
            total_neurons = sum(len(area.neurons) for area in areas)
            total_spike_rate = sum(area.get_spike_rate() for area in areas)
            
            status = f"""Time: {self.brain.time_step}
Total Neurons: {total_neurons}
Total Spike Rate: {total_spike_rate:.1f} Hz
Total Spikes: {self.brain.total_spikes}

Mode: {'🧠 DREAM (PFC→Vision)' if self.brain.dream_mode else '👁️ AWAKE (Vision→PFC)'}

Hierarchical Flow:
Vision → FEF → SEF → PFC
  ↑                   ↓
  ←── Feedback Loop ──┘

Area Details:
Vision: {len(self.brain.vision_area.neurons)}n, {self.brain.vision_area.get_spike_rate():.1f}Hz
FEF: {len(self.brain.fef_area.neurons)}n, {self.brain.fef_area.get_spike_rate():.1f}Hz
SEF: {len(self.brain.sef_area.neurons)}n, {self.brain.sef_area.get_spike_rate():.1f}Hz
PFC: {len(self.brain.pfc_area.neurons)}n, {self.brain.pfc_area.get_spike_rate():.1f}Hz

Degradation:
Bottom-up: {self.brain.bottom_up_degradation:.3f}
Top-down: {self.brain.top_down_degradation:.3f}

Camera: {'Active' if self.brain.vision_processor.camera_active else 'Inactive'}"""

            self.status_text.delete(1.0, tk.END)
            self.status_text.insert(1.0, status)
            
        except Exception as e:
            print(f"Visualization update error: {e}")
            
    def update_status(self, message):
        """Update status message"""
        print(f"Status: {message}")
        
    def run(self):
        """Start the GUI"""
        # Initialize with default neuron counts
        for area_name, var in self.neuron_controls.items():
            self.update_neuron_count(area_name, var.get())
            
        self.update_status("Hierarchical brain ready - toggle Dream Mode to see feedback!")
        
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
    print("🧠⚡ Launching Hierarchical Feedback Ephaptic Brain ⚡🧠")
    print("Features:")
    print("• 4-level hierarchy: Vision → FEF → SEF → PFC")
    print("• Natural degradation through processing levels")
    print("• Dream mode: PFC drives Vision (hallucination)")
    print("• Awake mode: Vision drives PFC (perception)")
    print("• Bidirectional feedback loops with ephaptic coupling")
    print("• Real attractor basin dynamics")
    print()
    
    try:
        app = HierarchicalBrainGUI()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Error", f"Application failed to start: {e}")