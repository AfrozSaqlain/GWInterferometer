import numpy as np

class Interferometer:
    def __init__(self, length=10, amplitude=1, frequency=10,
                 speed_of_signal=1, duration=8, sampling_rate=1024):

        self.length = length
        self.amplitude = amplitude
        self.frequency = frequency
        self.speed_of_signal = speed_of_signal

        self.duration = duration
        self.sampling_rate = sampling_rate

        self.time = np.linspace(0, duration, duration * sampling_rate)

    def generate_delta_phi(self): 
        random_shift = np.random.uniform(-1, 1, self.duration * self.sampling_rate) 
        return 2 * np.pi * self.frequency * (random_shift / self.speed_of_signal)
    
    def generate_fields(self):
        omega = 2 * np.pi * self.frequency

        delta_phi = self.generate_delta_phi()

        E1 = self.amplitude * np.exp(1j * omega * self.time)
        E2 = self.amplitude * np.exp(1j * (omega * self.time + delta_phi))

        return E1, E2

    def gw_signal(self, noise_template):
        N = len(noise_template)
        dt = self.time[1] - self.time[0]

        # --- GW duration (2 sec) ---
        gw_duration = 2
        gw_samples = int(gw_duration / dt)

        # --- Random start index ---
        start_idx = np.random.randint(0, N - gw_samples)

        # --- Time array for GW ---
        t_gw = np.linspace(0, gw_duration, gw_samples)

        # --- Chirp signal (toy model) ---
        f0 = 5     # start freq
        f1 = 50    # end freq

        # Linear chirp phase
        phase = 2 * np.pi * (f0 * t_gw + 0.5 * (f1 - f0) / gw_duration * t_gw**2)

        # Amplitude envelope (smooth turn-on/off)
        envelope = np.sin(np.pi * t_gw / gw_duration)**2

        gw = 0.0001 * envelope * np.sin(phase)  # small amplitude

        # --- Inject into template ---
        injected = noise_template.copy()
        injected[start_idx:start_idx + gw_samples] += gw

        return injected, start_idx

    def setup_interferometer(self, inject_gw=False):
        E1, E2 = self.generate_fields()
        intensity = np.abs(E1 + E2)**2

        if inject_gw:
            intensity, start_idx = self.gw_signal(intensity)
            return intensity, start_idx

        return intensity
