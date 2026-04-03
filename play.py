from interferometer import Interferometer
import matplotlib.pyplot as plt

detector = Interferometer(amplitude=1, sampling_rate=512, duration=12, target_snr=1)

signal, idx = detector.setup_interferometer(inject_gw=True)

plt.figure(figsize=(20, 6))
plt.plot(detector.time, signal)
plt.savefig('Signal.png')
plt.show()
