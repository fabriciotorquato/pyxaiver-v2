from sklearn.preprocessing import StandardScaler


def get_standard(data, standard):
    data = standard.transform(data)
    return data


def init_standard(data):
    standard = StandardScaler().fit(data)
    return get_standard(data, standard), standard
