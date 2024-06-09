import argparse
import traceback
import logging
import yaml
import sys
import os
import torch
import numpy as np

from discovery import DiscoveryDiffusion
from configs.paths_config import HYBRID_MODEL_PATHS

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    # Mode
    parser.add_argument('--unseen_reconstruct', action='store_true')
    parser.add_argument('--inversion', action='store_true')
    parser.add_argument('--unseen_sample', action='store_true')
    parser.add_argument('--finetune', action='store_true')


    # Default
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1268, help='Random seed')
    parser.add_argument('--exp', type=str, default='./runs/', help='Path for saving running related data.')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--ni', type=int, default=1,  help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--align_face', type=int, default=1, help='align face or not')

    # Sampling
    parser.add_argument('--t_0', type=int, default=500, help='Return step in [0, 1000)')
    parser.add_argument('--n_inv_step', type=int, default=40, help='# of steps during generative pross for inversion')
    parser.add_argument('--n_train_step', type=int, default=6, help='# of steps during generative pross for train')
    parser.add_argument('--n_test_step', type=int, default=40, help='# of steps during generative pross for test')
    parser.add_argument('--sample_type', type=str, default='ddim', help='ddpm for Markovian sampling, ddim for non-Markovian sampling')
    parser.add_argument('--eta', type=float, default=0.0, help='Controls of varaince of the generative process')

    # Train & Test
    parser.add_argument('--n_precomp_img', type=int, default=100, help='# of images to precompute latents')
    parser.add_argument('--n_train_img', type=int, default=50, help='# of training images')
    parser.add_argument('--n_test_img', type=int, default=10, help='# of test images')
    parser.add_argument('--model_path', type=str, default=None, help='Test model path')
    parser.add_argument('--img_path', type=str, default=None, help='Image path to test')
    parser.add_argument('--deterministic_inv', type=int, default=1, help='Whether to use deterministic inversion during inference')


    # Loss & Optimization
    parser.add_argument('--n_iter', type=int, default=1, help='# of iterations of a generative process with `n_train_img` images')
    parser.add_argument('--scheduler', type=int, default=1, help='Whether to increase the learning rate')
    parser.add_argument('--sch_gamma', type=float, default=1.3, help='Scheduler gamma')

    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    if args.unseen_sample:
         args.exp = args.exp + f'_SAMPLE_{new_config.data.category}'
    elif args.unseen_reconstruct:
        args.exp = args.exp + f'_REC_{new_config.data.category}'
    elif args.inversion:
        args.exp = args.exp + f'_INV_{new_config.data.category}'
    elif args.finetune:
        args.exp = args.exp + f'_FT_{new_config.data.category}'


    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    os.makedirs(args.exp, exist_ok=True)
    os.makedirs('checkpoint', exist_ok=True)
    os.makedirs('precomputed', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    os.makedirs(args.exp, exist_ok=True)
    args.image_folder = os.path.join(args.exp, 'image_samples')
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input("Image folder already exists. Overwrite? (Y/N)")
            if response.upper() == 'Y':
                overwrite = True

        if overwrite:
            # shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder, exist_ok=True)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device


    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    print(">" * 80)
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print("<" * 80)


    runner = DiscoveryDiffusion(args, config)
    try:
        if args.inversion:
            runner.inversion()
        elif args.unseen_sample:
            runner.unseen_sample()
        elif args.finetune:
            runner.finetune()
        elif args.unseen_reconstruct:
            runner.unseen_reconstruct()
        else:
            print('Choose one mode!')
            raise ValueError
    except Exception:
        logging.error(traceback.format_exc())


    return 0


if __name__ == '__main__':
    sys.exit(main())
