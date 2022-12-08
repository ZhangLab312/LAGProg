import numpy as np
from sklearn.model_selection import KFold


def split_censor(data, status, time):
    censored_index = np.where(status == 0)[0]
    censored_time = time[censored_index]

    censored_data = [data[i] for i in censored_index]
    print('the number of censored samples: {}'.format(censored_index.shape[0]))

    uncensored_index = np.where(status == 1)[0]
    uncensored_time = time[uncensored_index]
    uncensored_data = [data[i] for i in uncensored_index]
    print('the number of censored samples: {}'.format(uncensored_index.shape[0]))

    return censored_time, censored_data, uncensored_time, uncensored_data


def split_data(seed, censored_time, censored_data, uncensored_time, uncensored_data, fold_num=4, nfold=5):
    kf = KFold(n_splits=nfold, shuffle=True, random_state=seed)

    num = 0
    for train_index, test_index in kf.split(X=censored_data):
        train_censored_data = [censored_data[i] for i in train_index]
        train_censored_time = censored_time[train_index]
        test_censored_data = [censored_data[i] for i in test_index]
        test_censored_time = censored_time[test_index]
        train_censored_event = np.zeros_like(train_censored_time)
        test_censored_event = np.zeros_like(test_censored_time)
        if num == fold_num:
            test_index_all = test_index.tolist()
            break
        num += 1

    num = 0
    for train_index, test_index in kf.split(X=uncensored_data):
        train_uncensored_data = [uncensored_data[i] for i in train_index]
        train_uncensored_time = uncensored_time[train_index]
        test_uncensored_data = [uncensored_data[i] for i in test_index]
        test_uncensored_time = uncensored_time[test_index]
        train_uncensored_event = np.ones_like(train_uncensored_time)
        test_uncensored_event = np.ones_like(test_uncensored_time)
        if num == fold_num:
            test_index_all.extend(test_index.tolist())
            break
        num += 1

    train_censored_data.extend(train_uncensored_data)
    train_time = np.vstack((train_censored_time, train_uncensored_time))
    train_event = np.vstack((train_censored_event, train_uncensored_event))
    test_censored_data.extend(test_uncensored_data)
    test_time = np.vstack((test_censored_time, test_uncensored_time))
    test_event = np.vstack((test_censored_event, test_uncensored_event))

    return train_censored_data, train_time, train_event, test_censored_data, test_time, test_event, test_index_all


def sort_data(data, time):
    T = -np.abs(np.squeeze(np.array(time)))
    sorted_idx = np.argsort(T)
    data_final = [data[i] for i in sorted_idx]
    return sorted_idx, data_final, time[sorted_idx]
