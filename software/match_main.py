from t_match import match_6_
import os
import yaml

this_cwd = os.getcwd()


def read_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(this_cwd, 'config.yaml')
    if not os.path.exists(config_path):
        raise OSError('file dose not exist!')
    file = open(config_path, 'r', encoding='utf-8')
    para = yaml.load(file, Loader=yaml.FullLoader)
    return para


def main_fun():
    para = read_config()
    simulated_outcat_path = para['match']['simulated_outcat_path']
    detected_outcat_path = para['match']['detected_outcat_path']
    match_result = para['match']['match_save_path']
    match_6_.match_simu_detect(simulated_outcat_path, detected_outcat_path, match_result)


if __name__ == '__main__':
    main_fun()
