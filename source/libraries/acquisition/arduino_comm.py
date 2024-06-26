from libraries.acquisition.triggerbox import TriggerBox
from psychopy import core


# Interact with arduino connected through USB port


class ArduinoComm:

    def __init__(self):
        self.trigger = TriggerBox()
        self.pulse_duration = 70  # in ms
        self.pulse = self.trigger.make_analog_signal(channel=3, voltage=5, duration=self.pulse_duration)  # in ms
        self.running = self.trigger.make_analog_signal(channel=3, voltage=5, duration=0)  # infinite
        self.stop = self.trigger.make_cancel_signal(channel=3)  # stop ch 3
        self.timer = core.Clock()

        self.trigger.ser.write(self.stop)

    def send_pulses(self, nb_pulse):
        try:
            for current_pulse in range(nb_pulse):
                self.timer.reset()
                self.trigger.ser.write(self.pulse)
                while self.timer.getTime() < self.pulse_duration / 1000 * 2:
                    pass
        except:
            pass

        return self

    def send_on_signal(self):
        try:
            self.trigger.ser.write(self.running)
        except:
            pass

    def stop_signal(self):
        try:
            self.trigger.ser.write(self.stop)
        except:
            pass
        return self

    def close(self):
        self.trigger.ser.close()
