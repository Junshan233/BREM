import argparse
import yaml
from easydict import EasyDict as edict


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str,
                        default='configs/default.yaml', nargs='?')

    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--max_epoch', type=int)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--focal_loss', type=bool)

    parser.add_argument('--nms_thresh', type=float)
    parser.add_argument('--nms_sigma', type=float)
    parser.add_argument('--top_k', type=int)
    parser.add_argument('--output_json', type=str)

    parser.add_argument('--piou', type=float, default=0.5)
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--ngpu', type=int, default=1)

    parser.add_argument('--fusion', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', action='store_true')
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--not_freeze_bn', action='store_true')
    parser.add_argument('--bb_rate', type=float, default=1, help="Backbone learning rate = learning_rate * bb_rate")
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--world_size',
                        default=1,
                        type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url',
                        default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--output', type=str, default='')

    args = parser.parse_args()

    with open(args.config_file, 'r', encoding='utf-8') as f:
        tmp = f.read()
        data = yaml.load(tmp, Loader=yaml.FullLoader)

    data['training']['learning_rate'] = float(data['training']['learning_rate'])
    data['training']['weight_decay'] = float(data['training']['weight_decay'])

    if args.batch_size is not None:
        data['training']['batch_size'] = int(args.batch_size)
    if args.learning_rate is not None:
        data['training']['learning_rate'] = float(args.learning_rate)
    if args.weight_decay is not None:
        data['training']['weight_decay'] = float(args.weight_decay)
    if args.max_epoch is not None:
        data['training']['max_epoch'] = int(args.max_epoch)
    if args.checkpoint_path is not None:
        data['training']['checkpoint_path'] = args.checkpoint_path
        data['testing']['checkpoint_path'] = args.checkpoint_path
    if args.seed is not None:
        data['training']['random_seed'] = args.seed
    if args.focal_loss is not None:
        data['training']['focal_loss'] = args.focal_loss
    if args.not_freeze_bn:
        data['model']['freeze_bn'] = False
    data['training']['piou'] = args.piou
    data['training']['resume'] = args.resume
    data['ngpu'] = args.ngpu
    data['local_rank'] = args.local_rank
    data['world_size'] = args.world_size
    data['dist_url'] = args.dist_url
    data['dist'] = args.dist
    data['testing']['fusion'] = args.fusion
    data['freeze_backbone'] = args.freeze_backbone
    data['bb_rate'] = args.bb_rate
    data['fp16'] = args.fp16
    data['output'] = args.output
    if args.nms_thresh is not None:
        data['testing']['nms_thresh'] = args.nms_thresh
    if args.nms_sigma is not None:
        data['testing']['nms_sigma'] = args.nms_sigma
    if args.top_k is not None:
        data['testing']['top_k'] = args.top_k
    if args.output_json is not None:
        data['testing']['output_json'] = args.output_json
    data = edict(data)
    
    return data


config = get_config()
