from interferometer  import Interferometer
import matplotlib.pyplot as plt

detector = Interferometer(amplitude=0.01, sampling_rate=512)

signal, idx = detector.setup_interferometer(inject_gw=True)

plt.plot(detector.time, signal)
plt.axvline(detector.time[idx], color='r', linestyle='--', label="GW start")
plt.legend()
plt.show()
