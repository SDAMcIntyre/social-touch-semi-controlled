from modules.arduino_comm import ArduinoComm

# Fake interface when the arduino isn't connected


class ArduinoCommMock(ArduinoComm):

    def __init__(self):
        self.trigger = None

    def send_pulses(self, nb_pulse):
        return True

    def send_on_signal(self):
        pass

    def stop_signal(self):
        return True

    def close(self):
        return True





