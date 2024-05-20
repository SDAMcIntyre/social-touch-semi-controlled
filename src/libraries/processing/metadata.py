
class Metadata:
    def __init__(self, csv_filename):
        self.csv_filename = csv_filename

        self.trial_id: list[float] = []
        self.unit_id: list[str] = []
        self.unit_type: list[str] = []

        self.time: list[float] = []
