import argparse

def common(parser):
    parser.add_argument('--size', '-s', type=str, default='64x64x3')
    parser.add_argument('--batch', '-b', type=int, default=64)

def parse_args():
    parser = argparse.ArgumentParser(description='Train, run, and deploy your GANs.')
    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser('train')
    build_parser = subparsers.add_parser('build')
    serve_parser = subparsers.add_parser('serve')
    common(train_parser)
    common(build_parser)
    common(serve_parser)
    train_parser.add_argument('directory', action='store', type=str)
    train_parser.add_argument('--format', '-f', type=str, default='png')
    #parser.add_argument('--directory', type=str)
    parser.add_argument('--config', '-c', type=str)
    parser.add_argument('--epochs', type=int, default=1000)

    parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--crop', type=bool, default=True)


    #parser.add_argument('--bitrate', type=int, default=16*1024) TODO audio
    #parser.add_argument('--seconds', type=int, default=2) TODO audio
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--device', '-d', type=str, default='/gpu:0')
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--use_hc_experimental', action='store_true')


    return parser.parse_args()
