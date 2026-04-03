import numpy as np

class Interferometer:
    def __init__(self, length=10, amplitude=1, frequency=10,
                 speed_of_signal=1, duration=8, sampling_rate=1024,
                 phase_bias=np.pi / 2, phase_noise_std=0.05,
                 gw_strain_amplitude=1e-3, target_snr=None):

        self.length = length
        self.amplitude = amplitude
        self.frequency = frequency
        self.speed_of_signal = speed_of_signal
        self.phase_bias = phase_bias
        self.phase_noise_std = phase_noise_std
        self.gw_strain_amplitude = gw_strain_amplitude
        self.target_snr = target_snr

        self.duration = duration
        self.sampling_rate = sampling_rate

        self.time = np.linspace(0, duration, duration * sampling_rate, endpoint=False)

    def generate_delta_phi(self): 
        return np.random.normal(
            loc=0.0,
            scale=self.phase_noise_std,
            size=self.duration * self.sampling_rate,
        )

    def gw_signal(self, strain_amplitude=None, start_idx=None):
        N = len(self.time)
        dt = self.time[1] - self.time[0]

        gw_duration = 2
        gw_samples = int(gw_duration / dt)
        if start_idx is None:
            start_idx = np.random.randint(0, N - gw_samples)

        t_gw = np.linspace(0, gw_duration, gw_samples)

        f0 = 5
        f_merger = 80
        merger_time = 0.8 * gw_duration

        phase = np.zeros_like(t_gw)
        inspiral = t_gw <= merger_time
        ringdown = ~inspiral

        t_inspiral = t_gw[inspiral]
        phase[inspiral] = 2 * np.pi * (
            f0 * t_inspiral
            + 0.5 * (f_merger - f0) / merger_time * t_inspiral**2
        )

        phase_merger = phase[inspiral][-1]
        f_ringdown = 0.6 * f_merger
        tau = 0.08
        t_ringdown = t_gw[ringdown] - merger_time
        phase[ringdown] = phase_merger + 2 * np.pi * f_ringdown * t_ringdown

        envelope = np.zeros_like(t_gw)
        envelope[inspiral] = (t_inspiral / merger_time) ** 1.5
        envelope[ringdown] = np.exp(-t_ringdown / tau)

        if strain_amplitude is None:
            strain_amplitude = self.gw_strain_amplitude

        strain = np.zeros(N)
        strain[start_idx:start_idx + gw_samples] = (
            strain_amplitude * envelope * np.sin(phase)
        )

        return strain, start_idx, gw_samples

    def gw_to_delta_phi(self, strain):
        wavelength = self.speed_of_signal / self.frequency
        delta_length = 0.5 * self.length * strain
        return (4 * np.pi / wavelength) * delta_length

    def detector_output(self, delta_phi_1, delta_phi_2):
        omega = 2 * np.pi * self.frequency
        E1 = self.amplitude * np.exp(1j * (omega * self.time + delta_phi_1))
        E2 = self.amplitude * np.exp(1j * (omega * self.time + delta_phi_2))
        intensity = np.abs(E1 + E2)**2
        return intensity - np.mean(intensity)

    def measure_snr(self, noise_output, signal_response, start_idx, gw_samples):
        noise_segment = np.concatenate((
            noise_output[:start_idx],
            noise_output[start_idx + gw_samples:],
        ))
        noise_std = np.std(noise_segment)
        signal_rms = np.sqrt(
            np.mean(signal_response[start_idx:start_idx + gw_samples] ** 2)
        )
        if noise_std == 0:
            return np.inf
        return signal_rms / noise_std

    def generate_fields(self, inject_gw=False, gw_arm=2):
        omega = 2 * np.pi * self.frequency
        delta_phi_1 = np.zeros_like(self.time)
        delta_phi_2 = self.phase_bias + self.generate_delta_phi()
        start_idx = None
        gw_samples = None

        if inject_gw:
            strain, start_idx, gw_samples = self.gw_signal()
            if self.target_snr is not None:
                noise_output = self.detector_output(delta_phi_1, delta_phi_2)
                gw_delta_phi = self.gw_to_delta_phi(strain)

                if gw_arm == 1:
                    signal_response = (
                        self.detector_output(delta_phi_1 + gw_delta_phi, delta_phi_2)
                        - noise_output
                    )
                else:
                    signal_response = (
                        self.detector_output(delta_phi_1, delta_phi_2 + gw_delta_phi)
                        - noise_output
                    )

                current_snr = self.measure_snr(
                    noise_output, signal_response, start_idx, gw_samples
                )
                if np.isfinite(current_snr) and current_snr > 0:
                    strain *= self.target_snr / current_snr

            gw_delta_phi = self.gw_to_delta_phi(strain)

            if gw_arm == 1:
                delta_phi_1 += gw_delta_phi
            else:
                delta_phi_2 += gw_delta_phi

        E1 = self.amplitude * np.exp(1j * (omega * self.time + delta_phi_1))
        E2 = self.amplitude * np.exp(1j * (omega * self.time + delta_phi_2))

        return E1, E2, start_idx, gw_samples

    def setup_interferometer(self, inject_gw=False, gw_arm=2):
        E1, E2, start_idx, gw_samples = self.generate_fields(
            inject_gw=inject_gw, gw_arm=gw_arm
        )
        intensity = np.abs(E1 + E2)**2
        detector_output = intensity - np.mean(intensity)

        if inject_gw:
            return detector_output, start_idx

        return detector_output
