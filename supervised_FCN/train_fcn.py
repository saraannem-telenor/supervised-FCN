from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from preprocessing.preprocess_ucr import DatasetImporterUCR
from preprocessing.preprocess_uea import DatasetImporterUEA
from preprocessing.preprocess_dk import DatasetImporterDK, write_units

from experiments.exp_train import ExpFCN
from preprocessing.data_pipeline import build_data_pipeline
from utils import *


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    parser.add_argument('--config_dk', type=str, help="Path to the config_data_dk file.",
                        default=get_root_dir().joinpath('configs', 'config_data_dk.yaml'))
    return parser.parse_args()


if __name__ == '__main__':
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)
    config_data_dk = load_yaml_param_settings(args.config_dk)

    dataset_name = config['dataset']['name']
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']["num_workers"]
    additional_params = {}
    

    if dataset_name == 'UCR':
        # data pipeline
        dataset_subset_name = config['dataset']['subset_name']
        dataset_importer = DatasetImporterUCR(**config['dataset'])
        train_data_loader, test_data_loader = [build_data_pipeline(dataset_name = dataset_name,
                                                                    batch_size = batch_size,
                                                                    num_workers = num_workers,
                                                                    dataset_importer=dataset_importer, 
                                                                    kind=kind,
                                                                    **config,
                                                                    ) 
                                            for kind in ['train', 'test']]
    elif dataset_name == 'UEA':
        dataset_subset_name = config['dataset']['subset_name']
        dataset_importer = DatasetImporterUEA(**config['dataset'])
        train_data_loader, test_data_loader = [build_data_pipeline(dataset_name = dataset_name,
                                                                    batch_size = batch_size,
                                                                    num_workers = num_workers,
                                                                    dataset_importer=dataset_importer, 
                                                                    kind=kind,
                                                                    **config,
                                                                    ) 
                                            for kind in ['train', 'test']]
    elif dataset_name == 'DK':
        dataset_subset_name = config['dataset']['name']
        additional_params['input_length'] = config_data_dk['window_len']
        units = {}
        units['train'], units['test'] = [write_units(kind= kind, 
                                                     config= config_data_dk,)
                                        for kind in ['train', 'test']]
        
        units = sorted(list(set(units['train']) & set(units['test'])))

        # data pipeline
        train_dataset_importer, test_dataset_importer = [DatasetImporterDK(units = units,
                                                                           kind=kind,
                                                                           data_scaling = config['dataset']['data_scaling'],
                                                                           config=config_data_dk,)
                                                        for kind in ['train', 'test']]
        
        
        train_data_loader = build_data_pipeline(dataset_name = dataset_name,
                                                num_workers=num_workers, 
                                                batch_size=batch_size, 
                                                dataset_importer=train_dataset_importer,
                                                kind='train',
                                                **config_data_dk,
                                                )                                   
        test_data_loader = build_data_pipeline(dataset_name = dataset_name,
                                               num_workers=num_workers,
                                               batch_size=batch_size,
                                               dataset_importer=test_dataset_importer,
                                               kind='test',
                                               **config_data_dk,)

    # data pipeline
    

    # fit
    train_exp = ExpFCN(config, len(train_data_loader.dataset), len(np.unique(train_data_loader.dataset.Y)))
    wandb_logger = WandbLogger(project='supervised-FCN', name=dataset_subset_name, config=config)
    trainer = pl.Trainer(logger=wandb_logger,
                         enable_checkpointing=False,
                         callbacks=[LearningRateMonitor(logging_interval='epoch')],
                         **config['trainer_params'])
    trainer.fit(train_exp,
                train_dataloaders=train_data_loader,
                val_dataloaders=test_data_loader,)

    # test
    trainer.test(train_exp, test_data_loader)

    save_model({f"{dataset_subset_name}": train_exp.fcn})
