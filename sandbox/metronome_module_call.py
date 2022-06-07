import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../modules')
from metronome import Metronome

contact = "tapping"
size = "fingertip"
force = "moderate"
audio_version = "long"

m = Metronome(audioFolder="../cues")

for speed, duration in [[10, 10], [3, 10], [24, 10]]:
    m.init_metronome(contact=contact,
                     size=size,
                     force=force,
                     audio_version=audio_version,
                     speed=speed, vertical=False, duration=duration)
    m.start_metronome()

print("meow")