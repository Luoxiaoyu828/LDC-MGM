from Generate import generate, synthetic_data
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
    if para['generate']['model'] == 'simulation':
        dim = para['generate']['simulation']['dim']
        n = para['generate']['simulation']['n']
        path1 = para['generate']['simulation']['path1']
        size_v = para['generate']['simulation']['size_v']
        size_y = para['generate']['simulation']['size_y']
        size_x = para['generate']['simulation']['size_x']
        number = para['generate']['simulation']['number']
        fits_header_path = para['generate']['simulation']['fits_header_path']
        history_info = para['generate']['simulation']['history_info']
        information = para['generate']['simulation']['information']
        simulation = generate.Simulation(dim=dim, n=n, path1=path1, size_v=size_v, size_y=size_y, size_x=size_x,
                                         number=number, fits_header_path=fits_header_path, history_info=history_info,
                                         information=information)
        total_data = simulation.make_clumps()
    else:
        n = para['generate']['synthetic']['n']
        real_data_path = para['generate']['synthetic']['real_data_path']
        path = para['generate']['synthetic']['path']
        core_sample_path = para['generate']['synthetic']['core_sample_path']
        number = para['generate']['synthetic']['number']
        synthetic = synthetic_data.Synthetic(n=n, real_data_path=real_data_path, path=path, core_sample_path=core_sample_path,
                                             number=number)
        total_data = synthetic.make_synthetic_clumps()
    return total_data


if __name__ == '__main__':
    total_data = main_fun()
