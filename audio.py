import pyaudio
import numpy as np

# Audio settings
FORMAT = pyaudio.paInt16  # Audio format (16-bit resolution)
CHANNELS = 1              # Mono channel
RATE = 44100              # Sampling rate (44.1kHz)
CHUNK = 1024              # Number of samples per chunk

# Threshold for detecting sound
THRESHOLD = 500  # Adjust this value based on your environment

# Initialize PyAudio object
p = pyaudio.PyAudio()

# Open stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Listening for sound...")

try:
    while True:
        # Read audio data from the stream
        data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)

        # Compute the average volume (amplitude)
        volume = np.linalg.norm(data)

        # Check if the volume exceeds the threshold
        if volume > THRESHOLD:
            print("Sound detected!")
        else:
            print("No sound detected")

except KeyboardInterrupt:
    print("Program stopped.")

# Stop and close the stream
stream.stop_stream()
stream.close()

# Terminate PyAudio object
p.terminate()
