import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from matplotlib.animation import FuncAnimation
import random

class MaxwellBoltzmannSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Maxwell-Boltzmann Distribution Simulator")
        self.root.state("zoomed")
        self.root.configure(bg='#0f172a')
        
        # Physics constants
        self.k = 1.380649e-23  # Boltzmann constant (J/K)
        
        # Gas properties
        self.gases = {
            'Hydrogen (H₂)': 2.016,
            'Helium (He)': 4.003,
            'Nitrogen (N₂)': 28.014,
            'Oxygen (O₂)': 31.998,
            'Argon (Ar)': 39.948,
            'Carbon Dioxide (CO₂)': 44.01,
            'Xenon (Xe)': 131.29
        }
        
        # Initial state
        self.temperature = 300  # Kelvin
        self.molar_mass = 28.014  # N2 default
        self.comparison_temp = 500
        self.show_comparison = False
        
        # Particle system
        self.particles = []
        self.num_particles = 50
        
        # Setup GUI
        self.setup_gui()
        
        # Initial update
        self.update_all()
        
    # def setup_gui(self):
    #     """Setup the complete GUI layout"""
        
    #     # Main container
    #     root_container = tk.Frame(self.root, bg='#0f172a')
    #     root_container.pack(fill=tk.BOTH, expand=True)

    #     # main_frame = tk.Frame(self.root, bg='#0f172a')
    #     # main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    #     scroll_container = ScrollableFrame(root_container)
    #     scroll_container.pack(fill=tk.BOTH, expand=True)

    #     main_frame = scroll_container.scrollable_frame

    #     # Left panel - Controls
    #     self.setup_control_panel(main_frame)
        
    #     # Right panel - Visualizations
    #     self.setup_visualization_panel(main_frame)
    def setup_gui(self):
        """Setup the complete GUI layout"""

        # Root container (USES GRID)
        root_container = tk.Frame(self.root, bg='#0f172a')
        root_container.grid(row=0, column=0, sticky="nsew")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Scrollable container (MUST USE GRID)
        scroll_container = ScrollableFrame(root_container)
        scroll_container.grid(row=0, column=0, sticky="nsew")

        root_container.grid_rowconfigure(0, weight=1)
        root_container.grid_columnconfigure(0, weight=1)

        # This is where YOUR content goes
        main_frame = scroll_container.scrollable_frame

        # Left panel - Controls
        self.setup_control_panel(main_frame)

        # Right panel - Visualizations
        self.setup_visualization_panel(main_frame)

    def setup_control_panel(self, parent):
        """Setup the control panel with sliders and options"""
        
        control_frame = tk.Frame(parent, bg='#1e293b', relief=tk.RAISED, borderwidth=2)
        control_frame.grid(row=0, column=0, sticky='ns', padx=(0, 10))
        
        # Title
        title_label = tk.Label(
            control_frame, 
            text="Controls", 
            font=('Arial', 16, 'bold'),
            bg='#1e293b', 
            fg='#06b6d4'
        )
        title_label.pack(pady=10)
        
        # Temperature Control
        temp_frame = tk.LabelFrame(
            control_frame, 
            text="Temperature", 
            font=('Arial', 11, 'bold'),
            bg='#1e293b', 
            fg='#06b6d4',
            padx=15, 
            pady=10
        )
        temp_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.temp_label = tk.Label(
            temp_frame, 
            text=f"{self.temperature} K", 
            font=('Arial', 14, 'bold'),
            bg='#1e293b', 
            fg='#06b6d4'
        )
        self.temp_label.pack()
        
        self.temp_slider = tk.Scale(
            temp_frame,
            from_=100, 
            to=1000,
            orient=tk.HORIZONTAL,
            resolution=10,
            command=self.on_temperature_change,
            bg='#334155',
            fg='#94a3b8',
            highlightthickness=0,
            troughcolor='#475569',
            activebackground='#06b6d4',
            length=250
        )
        self.temp_slider.set(self.temperature)
        self.temp_slider.pack(pady=5)
        
        range_label = tk.Label(
            temp_frame, 
            text="100 K ← → 1000 K", 
            font=('Arial', 9),
            bg='#1e293b', 
            fg='#64748b'
        )
        range_label.pack()
        
        # Gas Selection
        gas_frame = tk.LabelFrame(
            control_frame, 
            text="Gas Type", 
            font=('Arial', 11, 'bold'),
            bg='#1e293b', 
            fg='#a855f7',
            padx=15, 
            pady=10
        )
        gas_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.gas_var = tk.StringVar(value='Nitrogen (N₂)')
        self.gas_dropdown = ttk.Combobox(
            gas_frame,
            textvariable=self.gas_var,
            values=list(self.gases.keys()),
            state='readonly',
            width=25
        )
        self.gas_dropdown.pack(pady=5)
        self.gas_dropdown.bind('<<ComboboxSelected>>', self.on_gas_change)
        
        self.mass_label = tk.Label(
            gas_frame, 
            text=f"Molar mass: {self.molar_mass} g/mol", 
            font=('Arial', 9),
            bg='#1e293b', 
            fg='#64748b'
        )
        self.mass_label.pack()
        
        # Comparison Mode
        comp_frame = tk.LabelFrame(
            control_frame, 
            text="Compare Temperatures", 
            font=('Arial', 11, 'bold'),
            bg='#1e293b', 
            fg='#ec4899',
            padx=15, 
            pady=10
        )
        comp_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.comparison_var = tk.BooleanVar(value=False)
        comparison_check = tk.Checkbutton(
            comp_frame,
            text="Enable Comparison",
            variable=self.comparison_var,
            command=self.toggle_comparison,
            bg='#1e293b',
            fg='#94a3b8',
            selectcolor='#334155',
            activebackground='#1e293b',
            activeforeground='#ec4899',
            font=('Arial', 10)
        )
        comparison_check.pack(pady=5)
        
        self.comp_temp_label = tk.Label(
            comp_frame, 
            text=f"Comparison: {self.comparison_temp} K", 
            font=('Arial', 10),
            bg='#1e293b', 
            fg='#a855f7'
        )
        self.comp_temp_label.pack()
        
        self.comp_temp_slider = tk.Scale(
            comp_frame,
            from_=100, 
            to=1000,
            orient=tk.HORIZONTAL,
            resolution=10,
            command=self.on_comparison_temp_change,
            bg='#334155',
            fg='#94a3b8',
            highlightthickness=0,
            troughcolor='#475569',
            activebackground='#a855f7',
            length=250,
            state=tk.DISABLED
        )
        self.comp_temp_slider.set(self.comparison_temp)
        self.comp_temp_slider.pack(pady=5)
        
        # Info Section
        info_frame = tk.LabelFrame(
            control_frame, 
            text="About", 
            font=('Arial', 11, 'bold'),
            bg='#1e293b', 
            fg='#64748b',
            padx=15, 
            pady=10
        )
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        info_text = (
            "The Maxwell-Boltzmann distribution\n"
            "describes the probability of finding\n"
            "a molecule at a particular speed in\n"
            "an ideal gas.\n\n"
            "Higher temperatures shift the\n"
            "distribution to higher speeds,\n"
            "while heavier molecules have\n"
            "lower average speeds."
        )
        
        info_label = tk.Label(
            info_frame,
            text=info_text,
            font=('Arial', 9),
            bg='#1e293b',
            fg='#94a3b8',
            justify=tk.LEFT,
            wraplength=250
        )
        info_label.pack()
        
        # Formula
        formula_label = tk.Label(
            info_frame,
            text="f(v) = 4π(m/2πkT)³ᐟ² v² e⁻ᵐᵛ²ᐟ²ᵏᵀ",
            font=('Courier', 9),
            bg='#0f172a',
            fg='#64748b',
            padx=5,
            pady=5
        )
        formula_label.pack(pady=10)
        
    def setup_visualization_panel(self, parent):
        """Setup the visualization panel with charts and statistics"""
        
        viz_frame = tk.Frame(parent, bg='#0f172a')
        viz_frame.grid(row=0, column=1, sticky='nsew')
        
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)
        
        # Title
        title_label = tk.Label(
            viz_frame,
            text="Maxwell-Boltzmann Distribution Simulator",
            font=('Arial', 20, 'bold'),
            bg='#0f172a',
            fg='#06b6d4'
        )
        title_label.pack(pady=10)
        
        subtitle_label = tk.Label(
            viz_frame,
            text="Visualize how molecular speeds are distributed in an ideal gas",
            font=('Arial', 11),
            bg='#0f172a',
            fg='#64748b'
        )
        subtitle_label.pack(pady=(0, 20))
        
        # Distribution Chart
        self.setup_distribution_chart(viz_frame)
        
        # Particle Simulation
        self.setup_particle_simulation(viz_frame)
        
        # Statistics Display
        self.setup_statistics(viz_frame)
        
    def setup_distribution_chart(self, parent):
        """Setup the main distribution chart"""
        
        chart_frame = tk.LabelFrame(
            parent,
            text="Speed Distribution",
            font=('Arial', 12, 'bold'),
            bg='#1e293b',
            fg='#94a3b8',
            padx=10,
            pady=10
        )
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 4), facecolor='#1e293b')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#0f172a')
        
        # Style the plot
        self.ax.spines['bottom'].set_color('#334155')
        self.ax.spines['top'].set_color('#334155')
        self.ax.spines['left'].set_color('#334155')
        self.ax.spines['right'].set_color('#334155')
        self.ax.tick_params(colors='#94a3b8')
        self.ax.xaxis.label.set_color('#94a3b8')
        self.ax.yaxis.label.set_color('#94a3b8')
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def setup_particle_simulation(self, parent):
        """Setup the particle simulation visualization"""
        
        particle_frame = tk.LabelFrame(
            parent,
            text="Particle Simulation (50 particles)",
            font=('Arial', 12, 'bold'),
            bg='#1e293b',
            fg='#94a3b8',
            padx=10,
            pady=10
        )
        particle_frame.pack(fill=tk.BOTH, padx=10, pady=10)
        
        # Create canvas for particles
        self.particle_canvas = tk.Canvas(
            particle_frame,
            width=1000,
            height=250,
            bg='#0f172a',
            highlightthickness=1,
            highlightbackground='#334155'
        )
        self.particle_canvas.pack()
        
        # Legend
        legend_frame = tk.Frame(particle_frame, bg='#1e293b')
        legend_frame.pack(pady=5)
        
        colors = [('#3b82f6', 'Slow'), ('#06b6d4', 'Medium'), 
                  ('#f97316', 'Fast'), ('#ef4444', 'Very Fast')]
        
        for color, label in colors:
            dot = tk.Canvas(legend_frame, width=15, height=15, bg='#1e293b', 
                          highlightthickness=0)
            dot.create_oval(2, 2, 13, 13, fill=color, outline=color)
            dot.pack(side=tk.LEFT, padx=5)
            
            tk.Label(legend_frame, text=label, bg='#1e293b', fg='#64748b',
                    font=('Arial', 9)).pack(side=tk.LEFT, padx=(0, 15))
        
        # Initialize particles
        self.init_particles()
        
        # Start animation
        self.animate_particles()
        
    def setup_statistics(self, parent):
        """Setup the statistics display"""
        
        stats_frame = tk.Frame(parent, bg='#0f172a')
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Three stat cards
        self.stat_cards = []
        
        stats_info = [
            ('Most Probable Speed (vₚ)', '√(2kT/m)', '#f97316'),
            ('Mean Speed (v̄)', '√(8kT/πm)', '#22c55e'),
            ('RMS Speed (vᵣₘₛ)', '√(3kT/m)', '#ec4899')
        ]
        
        for i, (label, formula, color) in enumerate(stats_info):
            card = tk.Frame(stats_frame, bg='#1e293b', relief=tk.RAISED, 
                          borderwidth=2)
            card.grid(row=0, column=i, padx=10, sticky='ew')
            stats_frame.grid_columnconfigure(i, weight=1)
            
            # Label
            tk.Label(card, text=label, font=('Arial', 10, 'bold'),
                    bg='#1e293b', fg='#94a3b8').pack(pady=(10, 0))
            
            # Formula
            tk.Label(card, text=formula, font=('Courier', 9),
                    bg='#1e293b', fg='#64748b').pack()
            
            # Value
            value_label = tk.Label(card, text="0 m/s", 
                                  font=('Arial', 18, 'bold'),
                                  bg='#1e293b', fg=color)
            value_label.pack(pady=(5, 10))
            
            self.stat_cards.append(value_label)
    
    def calculate_distribution(self, T, M):
        """Calculate Maxwell-Boltzmann distribution"""
        
        m = M / 1000 / 6.022e23  # Convert to kg per molecule
        
        # Speed range
        max_speed = np.sqrt(8 * self.k * T / (np.pi * m)) * 3
        speeds = np.linspace(0, max_speed, 200)
        
        # Maxwell-Boltzmann distribution
        factor = 4 * np.pi * np.power(m / (2 * np.pi * self.k * T), 1.5)
        probabilities = factor * speeds**2 * np.exp(-m * speeds**2 / (2 * self.k * T))
        
        # Normalize for better visualization
        probabilities = probabilities * 1e6
        
        return speeds, probabilities
    
    def calculate_speeds(self, T, M):
        """Calculate characteristic speeds"""
        
        m = M / 1000 / 6.022e23
        
        vp = np.sqrt(2 * self.k * T / m)  # Most probable
        vavg = np.sqrt(8 * self.k * T / (np.pi * m))  # Mean
        vrms = np.sqrt(3 * self.k * T / m)  # RMS
        
        return vp, vavg, vrms
    
    def update_distribution_chart(self):
        """Update the distribution chart"""
        
        self.ax.clear()
        
        # Calculate main distribution
        speeds, probs = self.calculate_distribution(self.temperature, self.molar_mass)
        
        # Plot main distribution
        self.ax.fill_between(speeds, probs, alpha=0.4, color='#06b6d4', 
                            label=f'{self.temperature} K')
        self.ax.plot(speeds, probs, color='#06b6d4', linewidth=2.5)
        
        # Plot comparison if enabled
        if self.show_comparison:
            speeds_comp, probs_comp = self.calculate_distribution(
                self.comparison_temp, self.molar_mass)
            self.ax.fill_between(speeds_comp, probs_comp, alpha=0.3, 
                               color='#a855f7', label=f'{self.comparison_temp} K')
            self.ax.plot(speeds_comp, probs_comp, color='#a855f7', linewidth=2)
        
        # Calculate and plot characteristic speeds
        vp, vavg, vrms = self.calculate_speeds(self.temperature, self.molar_mass)
        
        max_prob = np.max(probs)
        self.ax.axvline(vp, color='#f97316', linestyle='--', linewidth=1.5, 
                       label='vₚ (most probable)', alpha=0.8)
        self.ax.axvline(vavg, color='#22c55e', linestyle='--', linewidth=1.5, 
                       label='v̄ (mean)', alpha=0.8)
        self.ax.axvline(vrms, color='#ec4899', linestyle='--', linewidth=1.5, 
                       label='vᵣₘₛ', alpha=0.8)
        
        # Labels and styling
        self.ax.set_xlabel('Speed (m/s)', fontsize=11, color='#94a3b8')
        self.ax.set_ylabel('Probability Density', fontsize=11, color='#94a3b8')
        self.ax.legend(loc='upper right', facecolor='#1e293b', 
                      edgecolor='#334155', labelcolor='#94a3b8')
        self.ax.grid(True, alpha=0.2, color='#334155')
        
        # Style
        self.ax.set_facecolor('#0f172a')
        self.ax.spines['bottom'].set_color('#334155')
        self.ax.spines['top'].set_color('#334155')
        self.ax.spines['left'].set_color('#334155')
        self.ax.spines['right'].set_color('#334155')
        self.ax.tick_params(colors='#94a3b8')
        
        self.canvas.draw()
    
    def update_statistics(self):
        """Update the statistics display"""
        
        vp, vavg, vrms = self.calculate_speeds(self.temperature, self.molar_mass)
        
        self.stat_cards[0].config(text=f"{vp:.1f} m/s")
        self.stat_cards[1].config(text=f"{vavg:.1f} m/s")
        self.stat_cards[2].config(text=f"{vrms:.1f} m/s")
    
    def init_particles(self):
        """Initialize particle system"""
        
        self.particles = []
        
        for _ in range(self.num_particles):
            particle = {
                'id': self.particle_canvas.create_oval(0, 0, 10, 10, fill='white'),
                'x': random.uniform(10, 990),
                'y': random.uniform(10, 240),
                'vx': 0,
                'vy': 0,
                'speed': 0
            }
            self.particles.append(particle)
        
        self.update_particle_speeds()
    
    def sample_speed(self):
        """Sample speed from Maxwell-Boltzmann distribution using Box-Muller"""
        
        m = self.molar_mass / 1000 / 6.022e23
        sigma = np.sqrt(self.k * self.temperature / m)
        
        # Box-Muller transform for 3D velocities
        u1, u2, u3 = random.random(), random.random(), random.random()
        
        vx = sigma * np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        vy = sigma * np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
        vz = sigma * np.sqrt(-2 * np.log(u3)) * np.cos(2 * np.pi * random.random())
        
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        
        return speed, vx, vy
    
    def get_particle_color(self, speed):
        """Get particle color based on speed"""
        
        _, _, vrms = self.calculate_speeds(self.temperature, self.molar_mass)
        normalized = speed / vrms
        
        if normalized < 0.5:
            return '#3b82f6'  # Blue - slow
        elif normalized < 0.8:
            return '#06b6d4'  # Cyan - medium
        elif normalized < 1.2:
            return '#f97316'  # Orange - fast
        else:
            return '#ef4444'  # Red - very fast
    
    def update_particle_speeds(self):
        """Update particle speeds from distribution"""
        
        for particle in self.particles:
            speed, vx, vy = self.sample_speed()
            
            # Scale velocities for visualization
            scale = 0.05
            particle['vx'] = vx * scale
            particle['vy'] = vy * scale
            particle['speed'] = speed
            
            # Update color and size
            color = self.get_particle_color(speed)
            _, _, vrms = self.calculate_speeds(self.temperature, self.molar_mass)
            size = 6 + (speed / vrms) * 6
            
            # Update canvas item
            coords = self.particle_canvas.coords(particle['id'])
            if coords:
                cx, cy = (coords[0] + coords[2]) / 2, (coords[1] + coords[3]) / 2
                self.particle_canvas.coords(
                    particle['id'],
                    cx - size/2, cy - size/2,
                    cx + size/2, cy + size/2
                )
                self.particle_canvas.itemconfig(particle['id'], fill=color)
    
    def animate_particles(self):
        """Animate particles"""
        
        for particle in self.particles:
            # Get current position
            coords = self.particle_canvas.coords(particle['id'])
            if not coords:
                continue
                
            cx = (coords[0] + coords[2]) / 2
            cy = (coords[1] + coords[3]) / 2
            
            # Update position
            cx += particle['vx']
            cy += particle['vy']
            
            # Bounce off walls
            if cx < 10 or cx > 990:
                particle['vx'] *= -1
                cx = max(10, min(990, cx))
            
            if cy < 10 or cy > 240:
                particle['vy'] *= -1
                cy = max(10, min(240, cy))
            
            # Update canvas
            size = (coords[2] - coords[0]) / 2
            self.particle_canvas.coords(
                particle['id'],
                cx - size, cy - size,
                cx + size, cy + size
            )
        
        # Schedule next frame
        self.root.after(30, self.animate_particles)
    
    def update_all(self):
        """Update all visualizations"""
        
        self.update_distribution_chart()
        self.update_statistics()
        self.update_particle_speeds()
    
    def on_temperature_change(self, value):
        """Handle temperature slider change"""
        
        self.temperature = int(float(value))
        self.temp_label.config(text=f"{self.temperature} K")
        self.update_all()
    
    def on_comparison_temp_change(self, value):
        """Handle comparison temperature change"""
        
        self.comparison_temp = int(float(value))
        self.comp_temp_label.config(text=f"Comparison: {self.comparison_temp} K")
        if self.show_comparison:
            self.update_distribution_chart()
    
    def on_gas_change(self, event):
        """Handle gas selection change"""
        
        gas_name = self.gas_var.get()
        self.molar_mass = self.gases[gas_name]
        self.mass_label.config(text=f"Molar mass: {self.molar_mass} g/mol")
        self.update_all()
    
    def toggle_comparison(self):
        """Toggle comparison mode"""
        
        self.show_comparison = self.comparison_var.get()
        
        if self.show_comparison:
            self.comp_temp_slider.config(state=tk.NORMAL)
        else:
            self.comp_temp_slider.config(state=tk.DISABLED)
        
        self.update_distribution_chart()


class ScrollableFrame(tk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        # ✅ Store canvas on self
        self.canvas = tk.Canvas(
            container,
            bg='#0f172a',
            highlightthickness=0
        )

        v_scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(container, orient="horizontal", command=self.canvas.xview)

        self.scrollable_frame = tk.Frame(self.canvas, bg='#0f172a')

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.canvas.configure(
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set
        )

        # Layout (GRID ONLY – no pack)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # ✅ Bind scrolling AFTER canvas exists
        self.bind_scroll_events()



    def bind_scroll_events(self):
        def on_touchpad_scroll(event):
            # If Shift is pressed → horizontal
            if event.state & 0x0001:
                self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
            else:
                self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.canvas.bind_all("<MouseWheel>", on_touchpad_scroll)





def main():
    """Main entry point"""
    
    root = tk.Tk()
    app = MaxwellBoltzmannSimulator(root)
    root.mainloop()


if __name__ == "__main__":
    main()