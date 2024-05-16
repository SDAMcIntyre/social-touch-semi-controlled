
import keras
import pandas as pd
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv1D, Flatten, Dropout, GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D, Input, GaussianNoise, Activation, BatchNormalization, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.regularizers import L1L2
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.utils import Sequence
from semi_control_IFF import *

from params_semi_control_IFF import *
from semi_control_IFF import adapt_contact_condition



def adapt_label_vel(adapt_method=None, split_gesture=False):
    labels = pd.read_csv(data_dir + 'labels.csv')

    if adapt_method is not None:
        df_contact, _ = adapt_contact_condition(method=adapt_method, split_gesture=split_gesture)
        df_contact['unique_id'] = df_contact['unit'] + '_' + df_contact['trial_id']
        for id in df_contact['unique_id'].unique():
            labels.loc[labels['unique_id'] == id, 'vel'] = df_contact.loc[df_contact['unique_id'] == id, 'vel']
        name = '_' + adapt_method + '_' + 'split_ges' if split_gesture else '_' + adapt_method + '_' + 'combine_ges'
        labels.to_csv(data_dir + 'labels' + name + '.csv', index=False, header=True)


def data_prepare():
    df_all = pd.read_pickle(data_dir+'data_all.pkl')
    df_all.dropna(subset=['trial_id'], inplace=True)
    df_all['trial_id'] = df_all['block_id'].astype(str) + '_' + df_all['trial_id'].astype('Int32').astype(str)
    df_all['unique_id'] = df_all['unit'] + '_' + df_all['trial_id']

    win_len = 7000

    spike_trains = []
    labels = pd.DataFrame()

    for i_trial in df_all['unique_id'].unique():
        df_trial = df_all[df_all['unique_id'] == i_trial].reset_index()
        spike = list(df_trial['spike'].values)
        if len(spike) < win_len:
            new_spike = [0 for i in range(win_len - len(spike))] + spike
        else:
            max_idx = np.convolve(spike, np.ones(win_len,dtype=int),'valid').argmax()
            new_spike = spike[max_idx: max_idx + win_len]
        print(i_trial, len(new_spike))
        if len(new_spike) != win_len:
            raise SystemExit("length not match")
        spike_trains.append(new_spike)

        df_trial_label = df_trial.loc[[0], ['type', 'unit', 'block_id', 'trial_id', 'unique_id', 'stimulus', 'vel', 'finger', 'force']]
        labels = pd.concat([labels, df_trial_label])

    with open(data_dir + 'spike_trains.pkl', "wb") as fp:
        pickle.dump(spike_trains, fp)

    labels.to_csv(data_dir + 'labels.csv', index=False, header=True)

def CNN_classifier(inputs, n_output, filters=64, kernel_size_1=21, kernel_size=3, stride=3,
                   pool_size=3, dropout=0.2, l1=0.01, l2=0.01, batchnorm=False, flatten=False):

    x = Conv1D(filters, kernel_size=kernel_size_1, strides=1, padding='same')(inputs)
    if batchnorm: x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(filters * 2, kernel_size=kernel_size, strides=stride, padding='same', kernel_regularizer=L1L2(l1=l1, l2=l2))(x)
    if batchnorm: x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size, padding='same')(x)

    x = Conv1D(filters * 4, kernel_size=kernel_size, strides=stride, padding='same', kernel_regularizer=L1L2(l1=l1, l2=l2))(x)
    if batchnorm: x = BatchNormalization()(x)
    x = Activation("relu")(x)
    if dropout > 0: x = Dropout(rate=dropout)(x)

    x = Conv1D(filters * 8, kernel_size=kernel_size, strides=stride-1, padding='same', kernel_regularizer=L1L2(l1=l1, l2=l2))(x)
    if batchnorm: x = BatchNormalization()(x)
    x = Activation("relu")(x)
    if dropout > 0: x = Dropout(rate=dropout)(x)
    x = MaxPooling1D(pool_size, padding='same')(x)

    x = Conv1D(filters * 16, kernel_size=kernel_size, strides=1, padding='same')(x)
    if batchnorm: x = BatchNormalization()(x)
    x = Activation("relu")(x)


    if flatten == False:
        x = GlobalMaxPooling1D()(x)
    else:
        x = Flatten()(x)

    x = Dropout(0.3)(x)
    x = Dense(filters * 8, activation='relu', kernel_regularizer=L1L2(l1=l1, l2=l2))(x)
    x = Dropout(0.3)(x)
    x = Dense(filters * 2, activation='relu', kernel_regularizer=L1L2(l1=l1, l2=l2))(x)
    outputs = Dense(n_output, activation='softmax', name='prediction')(x)

    CNN_model = Model(inputs, outputs)
    # print(CNN_model.summary())
    return CNN_model

def Kfold_classification(X, Y, encoder, name, label_order, folder_name, k=5):
    sss = StratifiedShuffleSplit(n_splits=k, test_size=.2)
    cms, accs = [], []
    n_label = len(label_order)

    i_fold = 1
    for train_index, test_index in sss.split(X, Y):
        print('\n', name, 'fold:', i_fold)
        x_train, y_train = X[train_index], Y[train_index]
        x_test, y_test = X[test_index], Y[test_index]

        # split validation set in training set
        print(x_test.shape, x_train.shape)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True,  # 0.1
                                                          stratify=y_train)

        x_train = np.expand_dims(x_train, axis=2)
        x_val = np.expand_dims(x_val, axis=2)
        x_test = np.expand_dims(x_test, axis=2)
        print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

        batch_size = 32
        batchnorm = True
        if x_train.shape[0] <= 100:
            batch_size = 16
            batchnorm = False
            if x_train.shape[0] <= 50:
                batch_size = 8
        print('Batch Size: ', batch_size)

        inputs = Input(shape=(x_train[0].shape[0], 1))


        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=75,
                                                      min_delta=.001, min_lr=0.000001, verbose=1)
        earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=200,
                                                     min_delta=.0001, verbose=1, restore_best_weights=True)
        callbacks = [reduce_lr, earlystopper]


        nerve_model = CNN_classifier(inputs, n_label, filters=64, kernel_size_1=21, kernel_size=3, stride=3,
                   pool_size=3, dropout=0.2, l1=0, l2=0.01, batchnorm=batchnorm, flatten=True)
        optimizer = Adam(learning_rate=.001)
        nerve_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


        history = nerve_model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_val, y_val),
                        shuffle=True, verbose=0, epochs=100000, callbacks=callbacks)

        scores = nerve_model.evaluate(x_test, y_test, verbose=0)
        y_pred = nerve_model.predict(x_test)
        y_pred_label = encoder.inverse_transform(y_pred)
        y_test_label = encoder.inverse_transform(y_test)
        acc = np.sum(y_pred_label == y_test_label) / x_test.shape[0]
        cm = confusion_matrix(y_test_label, y_pred_label, labels=label_order)
        cms.append(cm)
        accs.append(acc)
        print(scores, acc)

        fig = plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc=0)
        fig.savefig(plot_dir + folder_name + '/loss_' + name + str(i_fold)+'.png', dpi=200)

        with open(data_dir + folder_name + '/CNN_clf_history_' + name + str(i_fold)+'.pkl', 'wb') as fp:
            pickle.dump(history.history, fp)
        i_fold += 1

    np.save(data_dir + folder_name + '/CNN_clf_CMs_' + name, np.array(cms))

    return cms, accs

def run_CNN(adapt_label=None, split_gesture=False):

    spike_trains = np.array(np.load(data_dir + 'spike_trains.pkl', allow_pickle=True))

    if adapt_label is not None:
        name = '_' + adapt_label + '_' + 'split_ges' if split_gesture else '_' + adapt_label + '_' + 'combine_ges'
    else:
        name = ''
    folder_name = 'CNN' + name
    if not os.path.exists(plot_dir+folder_name):
        os.makedirs(plot_dir+folder_name)
    if not os.path.exists(data_dir+folder_name):
        os.makedirs(data_dir+folder_name)

    labels = pd.read_csv(data_dir + 'labels' + name + '.csv')
    labels.replace({'stroke': 'S', 'tap': 'T'}, inplace=True)
    labels['label'] = labels['stimulus'] + '_' + labels['vel'].astype('int').astype('str')

    labels_sort = labels[['stimulus', 'vel', 'label']]
    labels_sort = labels_sort.sort_values(by=['stimulus', 'vel'])
    label_order = labels_sort['label'].unique()

    encoder = OneHotEncoder(sparse=False)
    y_one_hot = encoder.fit_transform(labels['label'].values.reshape(-1, 1))
    print(spike_trains.shape, y_one_hot.shape)

    unit_type_list, fold_list, acc_list = [], [], []
    k = 5

    unit_order = ['HFA']#'SAI', 'SAII', 'Field',,'CT'
    for unit_type in unit_order:
        print(unit_type)
        Y = y_one_hot[labels['type'] == unit_type]
        X = spike_trains[labels['type'] == unit_type]

        cms, accs = Kfold_classification(X, Y, encoder, unit_type, label_order, folder_name, k=k)

        unit_type_list = np.concatenate((unit_type_list, [unit_type] * k))
        fold_list = np.concatenate((fold_list, np.array(range(k)) + 1))
        acc_list = np.concatenate((acc_list, accs))

        fig = plt.figure()
        fig.set_size_inches(4, 4, forward=True)
        cm = np.sum(cms, axis=0)
        ax = fig.add_subplot(1, 1, 1)
        plot_confusion_matrix(cm, ax, labels=label_order, normalize=True,
                              title=unit_type + ' (' + str(len(Y)) + ')', label_combined='stimulus_vel')
        grid_stimuli = len(label_order) // 2
        ax.hlines([grid_stimuli], *ax.get_xlim(), color="gray")
        ax.vlines([grid_stimuli], *ax.get_xlim(), color="gray")
        plt.subplots_adjust(left=0.086, right=0.99, top=0.943, bottom=0.107, hspace=0.39, wspace=0.28)
        fig.savefig(plot_dir+folder_name+'/CNN_clf_CM_' + unit_type + '.png', dpi=200)

    df_results = pd.DataFrame({'unit_type': unit_type_list, 'fold': fold_list, 'accuracy': acc_list})
    print(df_results)
    df_results.to_csv(data_dir+folder_name+'/CNN_clf_results', index=False, header=True)


if __name__ == '__main__':

    sns.set(style="ticks", font='Arial')


    # data_prepare()

    # adapt_method = None
    # split_gesture = False

    adapt_method = 'VarGaussianMix' #'actual' #
    split_gesture = False

    # adapt_label_vel(adapt_method=adapt_method, split_gesture=split_gesture)

    # run_CNN(adapt_label=adapt_method, split_gesture=split_gesture)
