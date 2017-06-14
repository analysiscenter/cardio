""" contain Batch class for processing ECGs """
import wfdb
sys.path.append('..')
from dataset import Batch, action, inbatch_parallel

class EcgBatch(Batch):
    """
    Batch of ECG data
    """

    def __init__(self, index, preloaded=None):
        super().__init__(index, preloaded)

        self._init_data()
        self.history = []


    def _init_data(self, data=None, annotation=None, meta=None):
        self._data = data
        self._annotation = self.create_annotation_df() if annotation is None else annotation
        self._meta = dict() if meta is None else meta

    @staticmethod
    def create_annotation_df(data=None):
        """ Create a pandas dataframe with ECG annotations """
        return pd.DataFrame(data=data, columns=["ecg", "index", "value"])

    @action
    def load(self, src=None, fmt="wfdb"):
        """
        Loads data from different sources
        src is not used yet, so files locations are defined by the index
        """
        if fmt == "wfdb":
            self._load_wfdb(src=src)
        elif fmt == "npz":
            self._load_npz(src=src)
        else:
            raise TypeError("Incorrect type of source")

        # add info in self.history
        info = dict()
        info['method'] = 'load'
        info['params'] = {"src":src, "fmt":fmt}
        self.history.append(info)

        return self

    @action
    @inbatch_parallel(init='loader_init', post='loader_post', target='async')
    async def _load_wfdb(self, ix, src=None, *args, **kwargs):
        ecg = ix[1]
        pos = ix[0]
        if src:
            path = src[ecg]
        else:
            path = self.index.get_fullpath(ecg)

        signal, fields = wfdb.rdsamp(os.path.splitext(path)[0])
        signal = signal.T

        try:
            annot = wfdb.rdann(path, "atr")
        except FileNotFoundError:
            annot = self.create_annotation_df()  # pylint: disable=redefined-variable-type
        
        fields.update({"__pos": pos[0]})

        return (ecg, signal, annot, fields)

    @action
    @inbatch_parallel(init='loader_init', post='loader_post', target='async')
    async def _load_npz(self, ix, src=None, *args, **kwargs):
        ecg = ix[1]
        pos = ix[0]

        if src:
            path = src[ecg]
        else:
            path = self.index.get_fullpath(ecg)

        data = np.load(path)
        signal = data["signal"]
        annot = self.create_annotation_df(data["annotation"])
        fields = data["meta"].item()
        fields.update({"__pos": pos[0]})
        
        return (ecg, signal, annot, fields)

    def loader_init(self, *args, **kwargs):
        """
        Init method for parallelism in loading
        """
        
        init_indices = [[(init_val[0], init_val[1])] for init_val in np.ndenumerate(self.indices)]
        return init_indices

    def loader_post(self, list_of_results, *args, **kwargs):
        if any_action_failed(list_of_results):
            raise ValueError("Failed while parallelizing: ", self.get_errors(all_res))

        array_of_results = np.array(list_of_results)
        data = array_of_results[:,1]
        annot = pd.concat(array_of_results[:,2])
        meta = dict(zip(array_of_results[:,0],array_of_results[:,3]))
        
        self._init_data(data, annot, meta)

        return self

    @action
    def dump(self, dst, fmt="npz"):
        """
        Save each ecg in its own file named as '<index>.<fmt>'
        """
        if fmt == "npz":
            self._dump_npz(dst=dst)
        else:
            raise NotImplementedError("The format is not supported yet")

        return self
    
    @action
    @inbatch_parallel(init='indices', target='async')
    async def _dump_npz(self, ix, dst):
        signal, ann, meta = self[ix]
        del meta["__pos"]
        np.savez(os.path.join(dst, ix + ".npz"),
                signal=signal,
                annotation=ann,
                meta=meta)
        return os.path.join(dst, ix + ".npz")

    def __getitem__(self, index):
        if index in self.indices:
            pos = self._meta[index]['__pos']
            if self._annotation.empty:
                pos_annotation = self._annotation
            else:
                pos_annotation = self._annotation.loc[pos]
            return (self._data[pos], pos_annotation, self._meta[index])
        else:
            raise IndexError("There is no such index in the batch", index)

    def default_post(self, list_of_results, *args, **kwargs):  #pylint: disable=unused-argument
        """
        Default post for parallelism: collect results, make a numpy array
        and change self._data attribute to it.
        """
        # Check if all elements of the resulting list are numpy arrays.
        # If not - throw an exception, otherwise rewrite self._data attribute.
        if any_action_failed(list_of_results):
            raise ValueError("Could not assemble the batch: ", self.get_errors(all_res))

        if all(isinstance(x, np.ndarray) for x in list_of_results):
            self._data = np.array(list_of_results)
        else:
            raise ValueError("List of results contains non-numpy.ndarray elements.")
        return self

    @action
    @inbatch_parallel(init='indices', post='default_post', target='threads')
    def generate_subseqs(self, ix, length, step):
        """
        Function to generate a number of subsequnces of length,
        with step. Number of subseqs is defined by length of the
        initial signal and lenght.
        """
        sig = self[ix][0]
        n_splits = np.int((sig.shape[1] - length) / step) + 1
        splits = np.array([
            np.array(sig[:, i * step:(i * step + length)]) for i in range(n_splits)
        ])
        return splits
