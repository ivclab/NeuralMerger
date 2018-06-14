import argparse

arg_lists = []
parser = argparse.ArgumentParser()
def str2bool(v):
    return v.lower() in ('true', '1')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def setting():
    net_arg = add_argument_group('Calibration Setting')
    net_arg.add_argument('--merger_dir', type=str, default="./weight_loader/weight/lenetsound_lenetfashion/merge_ACCU/")
    net_arg.add_argument('--net', type=str, default="lenetsound_lenetfashion", choices=['lenetsound_lenetfashion', 'vggclothing_zfgender'])
    net_arg.add_argument('--save_model', type=str2bool, default="False")
    net_arg.add_argument('--max_step', type=int, default=20000)
    net_arg.add_argument('--decay_step', type=int, default=16000)
    net_arg.add_argument('--log_step', type=int, default=500)
    net_arg.add_argument('--save_step', type=int, default=5000)
    net_arg.add_argument('--random_seed', type=int, default=100)
    net_arg.add_argument('--batch_size', type=int, default=64)
    net_arg.add_argument('--lr_rate', type=float, default=0.0002)
    net_arg.add_argument('--data_split_num', type=int, default=5,choices=[1,2,3,4,5],
                                            help='data is divided into five groups. 5 means using 100% data')
    net_arg.add_argument('--data_path', type=str, default="./TFRecord/")
    net_arg.add_argument('--weight_dir', type=str, default="./weight_loader/weight/")
    config, unparsed = parser.parse_known_args()

    return config

