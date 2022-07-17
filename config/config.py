import argparse
import yaml
import os

def get_parser():
    parser = argparse.ArgumentParser(description='IMAGE_DEMOIREING')
    parser.add_argument('--config', type=str, default='config/uhdm_config.yaml', help='path to config file')
    args_cfg = parser.parse_args()
    assert args_cfg.config is not None
    with open(args_cfg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)

    return args_cfg


args = get_parser()
