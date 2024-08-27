import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
import yaml

# copying functions because I didn't manage to make relative imports work

def get_root_dir():
    return Path(__file__).parent.parent

def load_yaml_param_settings(yaml_fname: str):
    """
    :param yaml_fname: .yaml file that consists of hyper-parameter settings.
    """
    stream = open(yaml_fname, 'r')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    return config

def write_all_units(kind: str, config: dict):   
    col_for_groupby = config['col_for_groupby']
    #path_desired_units = get_root_dir().joinpath("configs", units_filename)
    nan_limit = config['nan_limit']
    window_len = config['input_len']

    data_root = get_root_dir().joinpath(config['data_dir']['data_root'])
    # fetch an entire dataset
    if kind == 'train':
        df = pd.read_pickle(data_root.joinpath(config['data_dir']['train_pkl']))
    elif kind == 'test':
        df = pd.read_pickle(data_root.joinpath(config['data_dir']['test_pkl']))

    units = df[col_for_groupby].dropna().unique().tolist()
    print('# before filtering #')
    print(kind, '{} samples'.format(df.shape[0]), '{} unique units'.format(len(units)))
    #if path_desired_units.is_file() and (path_desired_units.stat().st_size > 0):
    #    desired_units = read_units(path_desired_units)
    #    units = list(set(units) & set(desired_units))
    #    print('file found with {} units'.format(len(units)))

    df = df[df[col_for_groupby].isin(units)]

    # min number of samples that are not NaNs for each unit
    min_unit_count = (
        df.groupby(col_for_groupby, sort=False)
        .agg({col: "count" for col in df.columns if col != col_for_groupby}) #count excludes the NaNs
        .min(axis=1)
        .values
    )
    unit_size = df.groupby(col_for_groupby, sort=False).size()
    nan_percentage = 100 * (unit_size.values - min_unit_count) / unit_size.values
    min_number_values = 2 * (window_len)
    nan_units = unit_size.index.values[
        (nan_percentage > nan_limit) | (unit_size.values < min_number_values)
    ]
    nan_units = sorted(nan_units.tolist())

    # drop units that exceed nan_limit (10% currently) or have less than twice the window length
    df = df[~df[col_for_groupby].isin(nan_units)]
    units = sorted(df[col_for_groupby].dropna().unique().tolist())
    print('# after filtering #')
    print(kind, '{} samples'.format(df.shape[0]), '{} unique units'.format(len(units)),
        '{} NaN units'.format(len(nan_units)))
    return units


if __name__ == '__main__':
    config_dk = get_root_dir().joinpath('configs', 'config_data_dk.yaml')
    config_data_dk = load_yaml_param_settings(config_dk)
    print('getting units...')
    units = {}
    units['train'], units['test'] = [write_all_units(kind=kind, 
                                                     config=config_data_dk)
                                        for kind in ['train', 'test']]
    print('sorting...')
    all_units_train = sorted(list(set(units['train'])))
    all_units_test = sorted(list(set(units['test'])))
    intersection_units = sorted(list(set(units['train']) & set(units['test']))) # this is the intersection
    all_units = sorted(list(set(units['train']) | set(units['test'])))
    only_train = sorted(list(set(units['train']) - set(units['test'])))
    only_test = sorted(list(set(units['test']) - set(units['train'])))
    print('# unique units after filtering #')
    print('all: {}, train: {}, test: {}, intersection: {}, only_train: {}, only_test: {}'.format(
        len(all_units), len(all_units_train), len(all_units_test), len(intersection_units), len(only_train), len(only_test)))
    
    print("# writing units' files...")
    name_list = ['units_train_all', 'units_test_all', 'units_intersection', 'units_all', 'units_only_train', 'units_only_test']
    units_set_list = [all_units_train, all_units_test, intersection_units, all_units, only_train, only_test] 
    for name, units_set in zip(name_list, units_set_list):
        filename = get_root_dir().joinpath("configs", name+'.txt')
        with open(filename, 'w') as f:
            for unit in units_set:
                f.write(f"{unit}\n")
            print('{} saved'.format(filename))
            f.close()