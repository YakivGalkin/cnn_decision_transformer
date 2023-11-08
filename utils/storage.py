import os
import h5py
import numpy as np
import requests

# Function to save data to HDF5
def save_to_hdf5(data, filename, use_compression=True):
    with h5py.File(filename, 'w') as hdf5_file:
        for key, value in data.items():
            group = hdf5_file.create_group(key)
            for i, array in enumerate(value):
                if use_compression:
                    group.create_dataset(str(i), data=array, compression="gzip")
                else:
                    group.create_dataset(str(i), data=array)
    print(f"Data saved to {filename}")

# Function to load data from HDF5
def load_from_hdf5(filename):
    data = {}
    with h5py.File(filename, 'r') as hdf5_file:
        for key in hdf5_file.keys():
            group = hdf5_file[key]
            data[key] = [np.array(group[str(i)]) for i in range(len(group))]
    return data


def download_file(url, destination):
    if not os.path.exists(destination):
        print(f"Downloading file from {url} to {destination}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(destination, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    if chunk:
                        f.write(chunk)
        print("Download complete.")
    else:
        print(f"The file {destination} already exists. Download skipped.")


# nature_cnn_dql_pretrained.pt
# dql_car_racing.pt
def get_downloaded_model_file(pretrained_model_name):
    model_dir = './downloaded_models'
    model_path = os.path.join(model_dir, f'{pretrained_model_name}')
    
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    
    if not os.path.isfile(model_path):
        model_url = f"https://storage.googleapis.com/yakiv-dt-public/models/{pretrained_model_name}"
        download_file(model_url, model_path)
    return model_path


def get_pretrained_model(model, pretrained_model_name):
    model_path = get_downloaded_model_file(pretrained_model_name)
    
    # Load the model from the path
    loaded_model = model.load(model_path)
    return loaded_model

def load_dataset(dataset_name):
    dataset_dir = './downloaded_datasets'
    dataset_path = os.path.join(dataset_dir, f'{dataset_name}.hdf5')
    dataset_url = f"https://storage.googleapis.com/yakiv-dt-public/datasets/{dataset_name}.hdf5"
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    if not os.path.isfile(dataset_path):
        download_file(dataset_url, dataset_path)

    return load_from_hdf5(dataset_path)