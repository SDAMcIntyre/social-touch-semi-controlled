import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
import seaborn as sns


class ContactData:
    def __init__(self, csv_filename):
        # Define properties with types
        self.unit_id: list[str] = []
        self.trial_id: list[str] = []
        self.unit_type: list[str] = []

        self.stim_type: list[str] = []
        self.stim_vel: list[float] = []
        self.stim_size: list[float] = []
        self.stim_force: list[float] = []

        self.contact_area: list[float] = []
        self.contact_depth: list[float] = []
        self.contact_vel: list[float] = []

        # Load data from CSV file
        self.load_data_from_csv(csv_filename)

    def load_data_from_csv(self, csv_filename):
        contact_dataframe = pd.read_csv(csv_filename)

        # remove lines that contains NaN values
        contact_dataframe.dropna(inplace=True)

        # metadata
        self.unit_id = contact_dataframe.unit.values  #.reshape(-1, 1)  # ensure being a column array
        self.trial_id = contact_dataframe.trial_id.values
        self.unit_type = contact_dataframe.type.values

        # stimulus info
        self.stim_type = contact_dataframe.stimulus.values
        self.stim_vel = contact_dataframe.vel.values
        self.stim_size = contact_dataframe.finger.values
        self.stim_force = contact_dataframe.force.values

        # processed tracking data
        self.contact_area = contact_dataframe.area_mean.values
        self.contact_depth = contact_dataframe.depth_mean.values
        self.contact_vel = contact_dataframe.velAbs_mean.values

def redefine_stimulus_groups(self):
    #vel_expected = adapt_contact_helper(method, df_contact)
    #df_contact['vel'] = vel_expected
    vel = self.contact_vel
    vel_expected = self.stim_vel

    has_nan = np.isnan(vel).any()
    if has_nan:
        not_nan = ~np.isnan(vel)
        vel = vel[not_nan]
        vel_expected = vel_expected[not_nan]
    else:
        not_nan = np.array(True * len(self.stim_vel))

    vel = vel.reshape(-1, 1)
    print('data len:', len(vel), 'label len:', len(vel_expected))

    # ------------ variational Gaussian Mixture (find the number of clusters)
    n = 8
    bgm = BayesianGaussianMixture(n_components=n, weight_concentration_prior=100, random_state=0)
    bgm.fit(vel)
    pred = bgm.predict(vel)
    n_new = len(np.unique(pred))
    print('number of cluster:', str(n_new))

    valid_mean = [bgm.means_[i,0] for i in np.unique(pred)]
    idx_ord = np.argsort(valid_mean)
    print(valid_mean, idx_ord)
    label_ord = np.zeros_like(pred) - 1
    for i in range(len(idx_ord)):
        label_ord[pred == np.sort(np.unique(pred))[idx_ord[i]]] = i
    label_ = label_ord + 1#['# ' + str(int(i)) for i in label_ord + 1]

    if has_nan:
        vel_expected = np.array([np.nan] * len(self.stim_vel))
        vel_expected[not_nan] = label_
    else:
        vel_expected = label_

    return vel_expected



class ContactDataPlot:
    def __init__(self):
        self.label_size = 20
        self.tick_size = 17
        self.legend_size = 14

    def plot_contact_hist(self, cd: ContactData, tracking_kinect_datatype: str):
        """
            :param cd:
                ContactData object
            :param tracking_kinect_datatype:
            :return:
                figure
        """
        match tracking_kinect_datatype:
            case "area":
                expected_data = cd.stim_size
                recorded_data = cd.contact_area
            case "depth":
                expected_data = cd.stim_force
                recorded_data = cd.contact_depth
            case "velocity":
                expected_data = cd.stim_vel
                recorded_data = cd.contact_vel
            case _:
                raise Exception("Data type doesn't exist.")

        # define the range of the tracked kinect data for each expected value of the data
        data_ranges = []
        for label in np.unique(expected_data):
            if label != 0:
                mask = (expected_data == label)
                data_label = recorded_data[mask]
                val_min = round(np.nanmin(data_label), 2)
                val_max = round(np.nanmax(data_label), 2)
                vel_range = '[' + str(val_min) + ', ' + str(val_max) + ']'
                data_ranges.append(vel_range)
                print(label)
                print(vel_range)

        # initialise figure
        fig, axes = plt.subplots(2, 1, sharex='all', sharey='all')
        fig.set_size_inches(8, 5, forward=True)

        # plot for each type
        # starting with "stroke"
        type_id = 0
        for current_type in ['stroke', 'tap']:
            ax = axes[type_id]

            mask = (cd.stim_type == current_type)
            nlabels = len(np.unique(expected_data[mask]))
            palette = sns.color_palette('Set2', n_colors=nlabels)

            sns.histplot(x=recorded_data[mask], hue=expected_data[mask],
                         bins=50, palette=palette, multiple="stack", ax=ax)
            ax.set_title(current_type, size=self.label_size)
            ax.set_xlabel('', fontsize=self.label_size)
            ax.yaxis.label.set_size(self.label_size)
            ax.xaxis.set_tick_params(labelsize=self.tick_size, rotation=0)
            ax.yaxis.set_tick_params(labelsize=self.tick_size)
            if current_type == 'tap':
                old_legend = ax.legend_
                handles = old_legend.legendHandles
                labels_ = [t.get_text() for t in old_legend.get_texts()]
                expected_data = ['group ' + labels_[i] + ': ' + data_ranges[i] for i in range(len(labels_))]
                title = old_legend.get_title().get_text()
                ax.legend(handles, expected_data, title=title, frameon=False, loc=0, fontsize=self.legend_size, title_fontsize=self.legend_size, handletextpad=0.3,
                          markerscale=0.5, handlelength=1., labelspacing=.2, ncol=1, columnspacing=1)
            else:
                ax.legend_.remove()

            type_id += 1

        fig.suptitle(tracking_kinect_datatype, size=self.label_size)

        plt.tight_layout()
        sns.despine(trim=True)

        return fig
