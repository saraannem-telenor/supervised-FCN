from torch.utils.data import DataLoader

from preprocessing.preprocess_ucr import UCRDataset
from preprocessing.preprocess_uea import UEADataset
from preprocessing.preprocess_dk import DKDataset
from preprocessing.augmentations import Augmentations



def build_data_pipeline(dataset_name:str, num_workers:int, batch_size:int, dataset_importer, kind: str, **kwargs) -> DataLoader:
    """
    :param config:
    :param kind train/valid/test
    """
    shuffle = {'train':True, 'test':False}


    if dataset_name == 'UCR':
        dataset = UCRDataset(kind, dataset_importer)
    elif dataset_name == 'DK':
        dataset = DKDataset(kind, dataset_importer, **kwargs)
    elif dataset_name == 'UEA':
        Dataset = UEADataset(kind, dataset_importer, **kwargs)
    else:
        raise ValueError
    
    # DataLoader
    return DataLoader(dataset = dataset, 
                      shuffle = shuffle[kind], 
                      batch_size = batch_size, 
                      num_workers = num_workers, 
                      drop_last=False,
                      persistent_workers= True, 
                      pin_memory=True,
                      )  # `drop_last=False` due to some datasets with a very small dataset size.



# def build_data_pipeline(config: dict, kind: str) -> (DataLoader, DataLoader, DataLoader):
#     """
#     :param config:
#     :param kind train/valid/test
#     """
#     cf = config
#     dataset_name = cf['dataset']["name"]
#     batch_size = cf['dataset']["batch_size"]
#     num_workers = cf['dataset']["num_workers"]

#     if dataset_name == "UCR":
#         dataset_importer = DatasetImporterUCR(**cf['dataset'])
#         Dataset = UCRDataset

#     elif dataset_name == 'UEA':
#         dataset_importer = DatasetImporterUEA(**cf['dataset'])
#         Dataset = UEADataset
#     else:
#         raise ValueError("invalid `dataset_name`.")

#     # DataLoader
#     if kind == 'train':
#         train_dataset = Dataset("train", dataset_importer)
#         return DataLoader(train_dataset, batch_size, num_workers=num_workers, shuffle=True,
#                           drop_last=False)  # `drop_last=False` due to some datasets with a very small dataset size.
#     elif kind == 'test':
#         test_dataset = Dataset("test", dataset_importer)
#         return DataLoader(test_dataset, batch_size, num_workers=num_workers, shuffle=False, drop_last=False)
#     else:
#         raise ValueError


if __name__ == '__main__':
    # import os, sys
    # path = 'C:\projects\TSG-VQVAE\datasets'
    # os.chdir(path)
    # sys.path.append(path)

    config = {'dataset':
                  {'name': 'UCR',
                   'subset_name': 'FiftyWords',
                   'data_scaling': True,
                   'batch_size': 256,
                   'num_workers': 0}
              }

    data_loader = build_data_pipeline(config, 'train')
    print(data_loader)
