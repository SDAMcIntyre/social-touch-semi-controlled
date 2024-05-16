
import socket
import os


def get_computer_name():
    return socket.gethostname()


computer_name = get_computer_name()
print("Computer Name:", computer_name)

if computer_name == "baz":
    data_dir_base = "E:\\"
else:
    data_dir_base = 'C:\\Users\\basdu83'
data_dir_base = os.path.join(data_dir_base, 'OneDrive - Linköpings universitet', '_Teams', 'touch comm MNG Kinect', 'basil_tmp', 'data')
data_dir = os.path.join(data_dir_base, 'processed')
plot_dir = os.path.join(data_dir_base, 'analysed')
#data_dir = 'data/semi_control_IFF/'
#plot_dir = 'plots/semi_control_IFF/'

label_size = 20
tick_size = 17
legend_size = 14

features = ['area', 'depth', 'velAbs', 'velLong', 'velVert', 'velLat']
unit_order = ['SAI', 'SAII', 'HFA', 'Field', 'CT']
