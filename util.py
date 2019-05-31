import os
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences

def load_pickle_object(file_path):
    """Load a pickled Python object from local directory

    Parameters
    ----------
    file_path : string
        The path (absolute or relative) to the target directory where the pickle file is stored

    Returns
    -------
    pickle_obj : Python object
        The Python object from stored serialized representation
    """

    # check if file_path is string
    if not isinstance(file_path, str):
        raise TypeError('The file_path argument must be a string.')

    # check if the file exists
    if not os.path.exists(file_path):
        raise NameError('The file or path provided does not exist.')

    # verify .pickle file as target
    if not (file_path[-7:] == '.pickle' or file_path[-4:] == '.pkl'):
        raise ValueError('The file must end with a .pickle or .pkl suffix.')

    pickle_in = open(file_path, 'rb')
    pickle_obj = pickle.load(pickle_in)

    return pickle_obj
