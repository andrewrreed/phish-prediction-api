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

def generate_full_setlist(model, seed_setlist):
    '''
    Generate the remainder of a setlist given the previous 150 songs.
    
    Args:
        model (.hdf5) - a Phish prediction tensorflow model
        seed_setlist (ndarray) - encoded array of shape (150,)
    
    Returns:
        setlist (list) - generated sequence of encoded songs to complete the show
    
    '''
    
    setlist = []
    setlist_start = False
    pred_count = 0
    
    # generate remainder of setlist
    while setlist_start == False:
        # truncate sequences
        seq = pad_sequences([seed_setlist], maxlen=150, truncating='pre')[0]
        # predict next song
        next_song = model.predict_classes(np.array([seq])).item()
        # increment prediction counter
        pred_count += 1
        # check if a new setlist start is predicted (and its not the first song)
        if next_song == 8 and pred_count > 1:
            setlist_start = True
        else:
            # append to generated list
            setlist.append(next_song)
            # update seed_setlist to re-run for the next song
            seed_setlist = np.append(seed_setlist, next_song)
            
    return setlist