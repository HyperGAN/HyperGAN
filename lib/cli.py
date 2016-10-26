import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train, run, and deploy your GANs.')
    #subparsers = parser.add_subparsers()
    #train_parser = subparsers.add_parser('train')
    #build_parser = subparsers.add_parser('build')
    #serve_parser = subparsers.add_parser('serve')
    parser.add_argument('--load_config', type=str)
    parser.add_argument('--epochs', type=int, default=10)

    parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--directory', type=str)
    parser.add_argument('--no_stop', type=bool)
    parser.add_argument('--crop', type=bool, default=True)

    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--seconds', type=int, default=2)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--format', type=str, default='png')
    parser.add_argument('--bitrate', type=int, default=16*1024)
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--server', type=bool, default=False)
    parser.add_argument('--save_every', type=int, default=0)
    parser.add_argument('--device', type=str, default='/gpu:0')
    parser.add_argument('--build', type=bool, default=False)

    return parser.parse_args()
