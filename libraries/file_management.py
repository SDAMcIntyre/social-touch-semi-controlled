
import os


class FileManager:

    def __init__(self, _data_folder, _filename_prefix):
        self.data_folder = './' + _data_folder + '/'
        self.filename_prefix = _filename_prefix
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

        # initialise files
        try:
            self.dataFile = open(self.data_folder + self.filename_prefix + '_stimuli.csv', 'w')
            self.dataFile.write('trial,type,speed,contact_area,force,pulse_code,kinect_recording\n')
        except IOError:
            input("Could not open" + self.filename_prefix + '_stimuli.csv' + " file!")
        self.dataFile.close()

        try:
            self.logFile = open(self.data_folder + self.filename_prefix + '_log.csv', 'w')
            self.logFile.write('time,event\n')
        except IOError:
            input("Could not open" + self.filename_prefix + '_log.csv' + " file!")
        self.logFile.close()

    def generate_infoFile(self, exptInfo):
        infoFile = None
        try:
            infoFile = open(self.data_folder + self.filename_prefix + '_info.csv', 'w')
        except IOError:
            input("Could not open" + self.filename_prefix + '_info.csv' + " file!")

        for k, v in exptInfo.items():
            infoFile.write(k + ',' + str(v) + '\n')
        infoFile.close()

        return self

    # def logEvent(time,event,logFile)
    # @brief: Write event with its time into logFile.
    def logEvent(self, time, event):
        self.logFile = open(self.data_folder + self.filename_prefix + '_log.csv', 'a')
        self.logFile.write('{},{}\n'.format(time, event))
        self.logFile.close()
        print('LOG: {} {}'.format(time, event))

        return self

    def dataWrite(self, trialData):
        lineFormatting = ','.join(['{}']*len(trialData))+'\n'
        self.dataFile = open(self.data_folder + self.filename_prefix + '_stimuli.csv', 'a')
        self.dataFile.write(lineFormatting.format(*trialData))
        self.dataFile.close()

        return self

    def abort(self, keyTime):
        self.logEvent(keyTime, 'experiment aborted')
        self.dataFile.close()
        self.logFile.close()

        return self

    def end(self, time):
        self.logEvent(time, 'experiment finished')
        self.dataFile.close()
        self.logFile.close()
