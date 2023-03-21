import yaml


def init_yaml():
    yaml_struct = {
        'generate': {
            'model': 'simulation',
            'simulation': {
                'dim': 3,
                'n': 10,
                'path1': '',
                'size_v': 100,
                'size_y': 100,
                'size_x': 100,
                'number': 100,
                'fits_header_path': '',
                'history_info': None,
                'information': None
            },
            'synthetic': {
                'n': 10,
                'real_data_path': '',
                'path': '',
                'core_sample_path': './Generate/sample.txt',
                'number': 10
            }
        },
        'detect': {
            'model': 'LDC',
            'data_name': '',
            'para': {
                'delta_min': 4,
                'gradmin': 0.01,
                'v_min': 27,
                'noise_times': 6,
                'rms_times': 5
            },
            'save_folder': '',
            'split': False,
            'save_loc': False,
            'use_mgm': False
        },
        'mgm': {
            'outcat_name_loc': '',
            'origin_name': '',
            'mask_name': '',
            'save_path': '',
            'thresh_num': 1,
            'save_png': False
        },
        'match': {
            'simulated_outcat_path': '',
            'detected_outcat_path': '',
            'match_save_path': ''
        }
    }
    with open('config.yaml', 'w') as f:
        yaml.dump(yaml_struct, f)


if __name__ == '__main__':
    init_yaml()
