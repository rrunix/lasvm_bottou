import subprocess
import tempfile
import os

import numpy as np
from sklearn import base
from sklearn.datasets import dump_svmlight_file

import pathlib

PARAMS = [
    "",
    "kernel_type",
    "kernel_param",
    "nr_class",
    "total_sv",
    "rho",
    "",
    "svs per class"
]


class LasvmTraininResult:
    
    def __init__(self, model, execution_info):
        self.model = model
        self.execution_info = execution_info


class LasvmModel:
    
    def __init__(self, model_filename, convert_back_zero):
        self.model_filename = model_filename
        self.convert_back_zero = convert_back_zero
        
        with open(self.model_filename) as fin:
            self.params = {k: v for k, v in zip(PARAMS, fin.readlines()) if len(k) > 0}
    
    def predict(self, X):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_name = self._dump_lasvm_dataset(X, tmp)
            preditions_file = os.path.join(tmp, 'out.test')
            
            # call model....
            command = [os.path.join(pathlib.Path().resolve(), "la_test"), dataset_name, self.model_filename, preditions_file]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            
            # handle cases where the call does not end successfully
            if result.returncode != 0:
                raise ValueError("Non-zero return code ({}). stdout: {} \n stderr: {}".format(result.returncode, result.stdout, result.stderr))
            
            y = self._retrieve_predictions(preditions_file)
            
            if self.convert_back_zero:
                y[y == -1] = 0
            
        return y
            
    def _retrieve_predictions(self, filename):
        with open(filename, 'r') as fin:
            y = list(map(float, fin.readlines()))
            return np.array(y).astype(int)
    
    def _dump_lasvm_dataset(self, X, folder):
        dataset_file = os.path.join(folder, 'dataset')
        y = np.full(shape=X.shape[0], fill_value=-1)
        dump_svmlight_file(X, y, dataset_file)
        return dataset_file
    
    
def parse_output(process_stdout, convert_back_zero=False):
    chunks = []
    curr_chunk = None
    for line in process_stdout.split("\n"):
        if not line.startswith("@"):
            # Ignore verbose data
            pass
        
        if line.startswith("@chunk"):
            curr_chunk = {}
            chunks.append(curr_chunk)
            curr_chunk['chunk_id'] = line.split(" ")[1]
            
        if line.startswith("@param"):
            if curr_chunk is None:
                raise ValueError("Execution param outside chunk data")
            
            param = line[line.find(" "): ]
            param_name, param_value = param.split("=")
            curr_chunk[param_name.strip()] = param_value.strip()
        
    result = []            
    for chunk in chunks:
        model_filename = chunk['model_file']
        model = LasvmModel(model_filename, convert_back_zero=convert_back_zero)
        result.append(LasvmTraininResult(model, chunk))
        
    return result
    
def train_fake_streaming(X, y, model_base_name, chunks=None, optimizer=0, kernel_type=2, selection_type=0, degree=3, gamma=-1, coef0=0, cost=1, epochs=1, deltamax=1_000):
    """[summary]

    Args:
        X ([type]): [description]
        y ([type]): [description]
        model_base_name ([type]): [description]
        chunks ([type], optional): [description]. Defaults to None.
        optimizer (int, optional): [description]. Defaults to 0.
        kernel_type (int, optional): [description]. Defaults to 2.
        selection_type (int, optional): [description]. Defaults to 0.
        degree (int, optional): [description]. Defaults to 3.
        gamma (int, optional): [description]. Defaults to -1.
        coef0 (int, optional): [description]. Defaults to 0.
        cost (int, optional): [description]. Defaults to 1.
        epochs (int, optional): [description]. Defaults to 1.
        deltamax ([type], optional): [description]. Defaults to 1_000.

    Raises:
        ValueError: [description]
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    
    y = np.array(y).astype(int)
    
    
    if len(set(y)) > 2:
        raise ValueError("Only binary classification tasks are supported")
    
    zeros_mask = y == 0
    convert_back_zero = False
    
    if any(zeros_mask):
        y[zeros_mask] = -1
        convert_back_zero = True
    
    if set(y) != {-1, 1}:
        raise ValueError("Classes should be labelled as -1 (or 0) and 1")
    
    if X.shape[0] != len(y):
        raise ValueError("The number of observations should match the number of labels {} != {}".format(len(X), len(y)))
        
    with tempfile.TemporaryDirectory() as tmp:
        
        # Dump dataset
        dataset_file = os.path.join(tmp, 'dataset')
        dump_svmlight_file(X, y, dataset_file)
        
        base_arguments = [
            '-o', optimizer,
            '-t', kernel_type,
            '-s', selection_type,
            '-d', degree,
            '-g', gamma,
            '-r', coef0,
            '-c', cost,
            '-p', epochs,
            '-D', deltamax
        ]

        if chunks is not None:
            if chunks[-1] > X.shape[0]:
                raise ValueError("Max chunk value should be strictly smaller or equals than the size of the dataset")
            
            base_arguments.append('-z')
            base_arguments.extend(chunks)
            
        command = [os.path.join(pathlib.Path().resolve(), './la_svm'), *base_arguments, dataset_file, model_base_name]
        command = list(map(str, command))
    
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        
        # handle cases where the call does not end successfully
        if result.returncode != 0:
            raise ValueError("Non-zero return code ({}). stdout: {} \n stderr: {}".format(result.returncode, result.stdout, result.stderr))
          
        output = parse_output(result.stdout, convert_back_zero=convert_back_zero)
        
        if chunks is None:
            output = output[0]
            
        return output
