from medusa.components import SerializableComponent
from medusa.bci.cvep_spellers import LFSR, LFSR_PRIMITIVE_POLYNOMIALS
from gui import gui_utils as gu
from .app_constants import *
import numpy as np
import os
import math
import json


class Settings(SerializableComponent):

    def __init__(self, connection_settings=None, run_settings=None,
                 timings=None, colors=None, matrices=None):
        self.connection_settings = connection_settings if \
            connection_settings is not None else ConnectionSettings()
        self.run_settings = run_settings if \
            run_settings is not None else RunSettings()
        self.timings = timings if timings is not None else Timings()
        self.colors = colors if colors is not None else Colors()

        self.matrices = matrices
        if matrices is None:
            train_matrices, test_matrices, lags_info = \
                self.build_with_pary_sequences()
            self.matrices = {'train': train_matrices,
                             'test': test_matrices}

        self.colors = colors
        if colors is None:
            unique = self.get_unique_sequence_values()
            c_text, c_box = Colors.generate_grey_color_dicts(unique)
            self.colors = Colors(color_text_dict=c_text, color_box_dict=c_box)

    def to_serializable_obj(self):
        train_matrices = []
        test_matrices = []
        for matrix in self.matrices['train']:
            train_matrices.append(matrix.serialize())
        for matrix in self.matrices['test']:
            test_matrices.append(matrix.serialize())
        matrices_dict = {'train': train_matrices,
                         'test': test_matrices
                         }
        sett_dict = {'connection_settings': self.connection_settings.__dict__,
                     'run_settings': self.run_settings.__dict__,
                     'timings': self.timings.__dict__,
                     'matrices': matrices_dict,
                     'colors': self.colors.__dict__
                     }
        return sett_dict

    @staticmethod
    def from_serializable_obj(settings_dict):
        # Easy dicts
        conn_sett = ConnectionSettings(**settings_dict['connection_settings'])
        run_sett = RunSettings(**settings_dict['run_settings'])
        timings = Timings(**settings_dict['timings'])
        colors = Colors(**settings_dict['colors'])
        # Train matrices
        train_matrices = []
        for m in settings_dict['matrices']['train']:
            item_list = list()
            for i in m['item_list']:
                target = CVEPTarget(text=i['text'],
                                    label=i['label'],
                                    sequence=i['sequence'],
                                    functionality=i['functionality'],
                                    ignored=i['ignored'])
                item_list.append(target)
            matrix = CVEPMatrix(n_row=m['n_row'], n_col=m['n_col'])
            matrix.item_list = item_list
            matrix.organize_matrix()
            train_matrices.append(matrix)
        # Train matrices
        test_matrices = []
        for m in settings_dict['matrices']['test']:
            item_list = list()
            for i in m['item_list']:
                target = CVEPTarget(text=i['text'],
                                    label=i['label'],
                                    sequence=i['sequence'],
                                    functionality=i['functionality'],
                                    ignored=i['ignored'])
                item_list.append(target)
            matrix = CVEPMatrix(n_row=m['n_row'], n_col=m['n_col'])
            matrix.item_list = item_list
            matrix.organize_matrix()
            test_matrices.append(matrix)
        # Merge both
        matrices = {'train': train_matrices,
                    'test': test_matrices
                    }
        return Settings(connection_settings=conn_sett,
                        run_settings=run_sett,
                        timings=timings,
                        colors=colors,
                        matrices=matrices)

    def set_matrices(self, train_matrices, test_matrices):
        self.matrices = {'train': train_matrices,
                         'test': test_matrices}

    def get_dict_matrices(self):
        m_dict = {'train': [], 'test': []}
        for m in self.matrices['train']:
            m_dict['train'].append(m.serialize())
        for m in self.matrices['test']:
            m_dict['test'].append(m.serialize())
        return m_dict

    def get_unique_sequence_values(self):
        unique_values = []
        for m in self.matrices['train']:
            for i in m.item_list:
                unique_values += list(np.unique(i.sequence))
        for m in self.matrices['test']:
            for i in m.item_list:
                unique_values += list(np.unique(i.sequence))
        return list(np.unique(unique_values))

    @staticmethod
    def get_coords_from_labels(labels, matrices):
        coords = []
        for label in labels:
            label_coord = []
            for idx, matrix in enumerate(matrices):
                target = matrix.get_target_from_label(label)
                if len(target) > 1:
                    print('WARNING in get_codes_from_labels: more than one '
                          'command for label %s, taking the first one' % label)
                if len(target) > 0:
                    label_coord = [idx, target[0].row, target[0].col]
                    break
            coords.append(label_coord)
        return coords

    @staticmethod
    def build_with_pary_sequences(n_row=4, n_col=4, base=2, order=6):
        """ Computes a predefined standard c-VEP matrix that modulates commands
        using a single-sequence via circular shifting.

        Parameter
        -----------
        n_row: int
            Number of rows.
        n_col: int
            Number of columns.
        base: int
            Base of the sequence (must be prime).
        order: int
            Order of the sequence (length will be L = base^order - 1).

        Returns
        --------
        matrix : CVEPMatrix
            Structured matrix object.
        """
        # Number of commands supported
        no_commands = n_row * n_col
        mseqlen = base ** order - 1
        tau = mseqlen / no_commands

        # Init
        comms = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/*-+.,' \
                '_abcdefghijklmnopqrstuvwxyz'
        comms *= 20
        comms_ = comms[:no_commands]
        lags_ = list(range(no_commands))

        # M-sequence generation
        if base not in LFSR_PRIMITIVE_POLYNOMIALS['base']:
            raise ValueError('[cvep_speller/settings] Base %i not supported!'
                             % base)
        if order not in LFSR_PRIMITIVE_POLYNOMIALS['base'][base]['order']:
            raise ValueError('[cvep_speller/settings] Cannot find any '
                             'primitive polynomial of order %i for base %i'
                             % (order, base))
        poly_ = LFSR_PRIMITIVE_POLYNOMIALS['base'][base]['order'][order]
        seq = LFSR(poly_, base=base).sequence
        centered_seq = LFSR(poly_, base=base, center=True).sequence

        # Optimize correlations
        rxx_, tr_ = Settings.autocorr_circular(centered_seq)
        rxx_ = rxx_ / np.max(np.abs(rxx_))
        half_rxx = rxx_[int(len(rxx_)/2):]
        min_p = - min(np.abs(np.unique(half_rxx)))
        bad_indexes = np.where(half_rxx != min_p)[0][1:]
        lags = np.linspace(0, len(seq), no_commands + 1).astype(int)[:-1]
        if any(item in bad_indexes for item in lags):
            # Need to optimize: there are non-minimal correlations
            tau = (len(seq) - len(bad_indexes)) / no_commands
            temp_lags = np.linspace(0, len(seq) - len(bad_indexes),
                                    no_commands + 1).astype(int)[:-1]

            good_idxs = np.arange(0, len(seq))
            good_idxs = np.delete(good_idxs, bad_indexes)
            lags = good_idxs[temp_lags]
        lags_info = {
            'tau': tau,
            'lags': lags,
            'bad_lags': bad_indexes
        }
        if tau <= 1:
            print('[cvep_speller/settings] Sequence length is not enough to '
                  'encode all commands. Please, reduce the number of commands '
                  '(%i) or increment the sequence length (supported lengths: '
                  '31, 63, 127 or 255)' % no_commands)
        if any(item in bad_indexes for item in lags):
            print('[cvep_speller/settings] Could not find any optimal '
                  'distributions of lags for this configuration!')

        # Set up the test matrix
        test_matrix = CVEPMatrix(n_row, n_col)
        for idx, c in enumerate(comms_):
            seq_ = circular_shift(sequence=seq, lag=lags[idx])
            target = CVEPTarget(text=c, label=c, sequence=seq_)
            test_matrix.append(target)
        test_matrix.organize_matrix()

        # Set up the train matrix (1x1 without lag)
        train_matrix = CVEPMatrix(1, 1)
        seq_ = circular_shift(sequence=seq, lag=0)
        target = CVEPTarget(text='0', label='0', sequence=seq_)
        train_matrix.append(target)
        train_matrix.organize_matrix()

        # Return
        test_matrices = [test_matrix]
        train_matrices = [train_matrix]
        return train_matrices, test_matrices, lags_info

    @staticmethod
    def autocorr_circular(x):
        """ With circular shifts (periodic correlation) """
        N = len(x)
        rxx = []
        t = []
        for i in range(-(N - 1), N):
            rxx.append(np.sum(x * np.roll(x, i)))
            t.append(i)
        rxx = np.array(rxx)
        return rxx, t


class ConnectionSettings:
    def __init__(self, path_to_exe=None, ip="127.0.0.1", port=50000):
        self.path_to_exe = path_to_exe
        self.ip = ip
        self.port = port

        # Default .exe path
        if self.path_to_exe is None:
            self.path_to_exe = os.path.dirname(__file__) +  \
                               '/unity/P-ary c-VEP Speller.exe'


class RunSettings:
    def __init__(self, user="S0X", session="Train", run=1,
                 mode=TRAIN_MODE,
                 enable_photodiode=True,
                 train_cycles=10, train_trials=5,
                 test_cycles=10,
                 cvep_model_path='',
                 fps_resolution=120,
                 early_stopping=3.0):
        self.user = user
        self.session = session
        self.run = run
        self.mode = mode
        self.enable_photodiode = enable_photodiode
        self.train_cycles = train_cycles
        self.train_trials = train_trials
        self.test_cycles = test_cycles
        self.cvep_model_path = cvep_model_path
        self.fps_resolution = fps_resolution
        self.early_stopping = early_stopping


class Timings:

    def __init__(self, t_prev_text=1.0, t_prev_iddle=1.0, t_finish_text=1.0):
        self.t_prev_text = t_prev_text
        self.t_prev_iddle = t_prev_iddle
        self.t_finish_text = t_finish_text


class Colors:
    def __init__(self, color_box_dict=None,
                 color_text_dict=None,
                 color_background='#f9e1ff',
                 color_target_box='#ff195bff',
                 color_highlight_result_box='#03fc5aff',
                 color_result_info_box='#8c8c8cff',
                 color_result_info_label='#b7b7b7ff',
                 color_result_info_text='#f4f657ff',
                 color_fps_good='#5ee57dff',
                 color_fps_bad='#b43228ff',
                 color_point='#00807Bff'):
        self.color_background = color_background
        self.color_target_box = color_target_box
        self.color_highlight_result_box = color_highlight_result_box
        self.color_result_info_box = color_result_info_box
        self.color_result_info_label = color_result_info_label
        self.color_result_info_text = color_result_info_text
        self.color_fps_good = color_fps_good
        self.color_fps_bad = color_fps_bad
        self.color_box_dict = color_box_dict
        self.color_text_dict = color_text_dict
        self.color_point = color_point

    @staticmethod
    def generate_grey_color_dicts(unique_sequence_values):
        # Init color
        init_grey = (255, 255, 255)
        end_grey = (0, 0, 0)
        init_hex = gu.rgb_to_hex(init_grey)
        end_hex = gu.rgb_to_hex(end_grey)
        init_grey = gu.rgb_to_hsv(init_grey)
        end_grey = gu.rgb_to_hsv(end_grey)

        # HSV gradient
        h = np.linspace(init_grey[0], end_grey[0], len(unique_sequence_values))
        s = np.linspace(init_grey[1], end_grey[1], len(unique_sequence_values))
        v = np.linspace(init_grey[2], end_grey[2], len(unique_sequence_values))

        # Genetare the HEX dictionary
        color_text_dict = dict()
        color_box_dict = dict()
        for i in range(len(unique_sequence_values)):
            rgb1 = np.round(gu.hsv_to_rgb((h[i], s[i], v[i]))).astype(int)
            hex1 = gu.rgb_to_hex(tuple(rgb1))
            color_box_dict[str(unique_sequence_values[i])] = hex1
            if v[i] <= 50:
                color_text_dict[str(unique_sequence_values[i])] = init_hex
            else:
                color_text_dict[str(unique_sequence_values[i])] = end_hex
        return color_box_dict, color_box_dict

    @staticmethod
    def concat_dict(dict_):
        concat = []
        for key in dict_:
            concat.append([str(key), str(dict_[key])])
        return concat

class CVEPMatrix:

    def __init__(self, n_row=-1, n_col=-1):
        """ Class that represents a c-VEP matrix.

        Attribute `item_list` encompasses a vector of commands, while
        `matrix_list` contains these same commands arranged in a matrix-like
        fashion to be accessed using `row` and `col` as indexes. Therefore,
        it is mandatory to call `organize_matrix()` sometime.

        Parameters
        ----------
        n_row: int
            (Optional, default:-1) Number of rows of the output matrix.
            If -1, `n_row` will be taken from the class.
        n_col: int
            (Optional, default:-1) Number of columns of the output matrix.
            If -1, `n_col` will be taken from the class.
        """
        self.n_row = n_row
        self.n_col = n_col

        self.item_list = []  # Vector of targets
        self.matrix_list = []  # Matrix of targets.
        # Each target can be accessed matrix.matrix_list[row][col]

    def remove(self, index):
        """ Removes a CVEPTarget element from the list of targets."""
        self.item_list.pop(index)

    def append(self, new_item):
        """ Appends a new CVEPTarget item to the list of targets. """
        if type(new_item) != CVEPTarget:
            raise ValueError('Cannot append, object type is not CVEPTarget.')
        self.item_list.append(new_item)

    def organize_matrix(self, n_row=-1, n_col=-1):
        """ Arranges the target list into a matrix.

        Parameters
        ----------
        n_row: int
            (Optional, default:-1) Number of rows of the output matrix.
            If -1, `n_row` will be taken from the class.
        n_col: int
            (Optional, default:-1) Number of columns of the output matrix.
            If -1, `n_col` will be taken from the class.

        Returns
        -------
        matrix_list: 2D list of CVEPTarget items
            Matrix-shaped target list.
        """
        # If n_row or n_col are set to -1, take the current values
        if n_row == -1:
            n_row = self.n_row
        if n_col == -1:
            n_col = self.n_col
        # Check if the list of targets can be reorganized in n_row and n_col
        if len(self.item_list) != (n_col * n_row):
            raise ValueError('Cannot organize elements in matrix. Number of '
                             'elements (%d) does not match the product of'
                             ' %d rows and %d columns.' % (len(self.item_list),
                                                           n_row, n_col))

        # Re-organize matrix
        r_count = c_count = 0
        self.matrix_list = [[None for y in range(n_col)] for x in range(n_row)]
        for element in self.item_list:
            element.set_row(r_count)
            element.set_col(c_count)
            self.matrix_list[r_count][c_count] = element
            c_count += 1
            if c_count >= n_col:
                c_count = 0
                r_count += 1
        return self.matrix_list

    def get_target_from_label(self, label):
        """ This method returns a list of items for the given label. """
        target = []
        for row in self.matrix_list:
            for item in row:
                if label == item.label:
                    target.append(item)
        return target

    def get_row_col_from_idx(self, idx):
        """ This function returns the [row, col] for a item_list index. """
        row = math.floor(idx / self.n_col)
        col = idx % self.n_col
        return [row, col]

    def serialize(self):
        items = []
        for i in self.item_list:
            items.append(i.to_dict())
        return {"n_row": self.n_row,
                "n_col": self.n_col,
                "item_list": items}


class CVEPTarget:

    def __init__(self, row=-1, col=-1, text='', label='', sequence=None,
                 functionality=FUNC_NONE, ignored=False):
        """ Class that represents a target cell of the c-VEP speller matrix.

        Parameters
        ----------
        row: int
            Row of the target. It can be specified later using `set_row` method.
        col: int
            Column of the target. It can be specified later using `set_col`
             method.
        text: basestring
            Text that will be displayed in the target cell.
        label: basestring
            Label that identifies the target cell.
        sequence : list
            Sequence that modulates this target item. The modulation will
            always start with the first index, i.e., sequence[0]
        """

        # Useful parameters
        self.row = row
        self.col = col
        self.text = text
        self.label = label
        self.sequence = sequence

    def set_row(self, row):
        self.row = row

    def set_col(self, col):
        self.col = col

    def set_text(self, text):
        self.text = text

    def set_label(self, label):
        self.label = label

    def set_sequence(self, sequence):
        self.sequence = sequence

    def to_dict(self):
        return self.__dict__

    def to_json(self):
        return json.dumps(self.__dict__)


def circular_shift(sequence, lag):
    """ Shifts circularly a sequence list the desired lag to the left.

    Parameters
    ----------
    sequence : list
        Input sequence to shift.
    lag : int
        Lag to be shifted.

    Returns
    ---------
    shifted_sequence: list
        Shifted sequence. E.g., sequence = [0, 1, 2, 3, 4] and lag = 1 will
        return [1, 2, 3, 4, 0]
    """
    return np.roll(sequence, -lag).tolist()
