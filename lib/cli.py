import argparse

def common(parser):
    parser.add_argument('directory', action='store', type=str)
    parser.add_argument('--size', '-s', type=str, default='64x64x3')
    parser.add_argument('--batch', '-b', type=int, default=64)
    parser.add_argument('--config', '-c', type=str, default=None)
    parser.add_argument('--device', '-d', type=str, default='/gpu:0')
    parser.add_argument('--format', '-f', type=str, default='png')
    parser.add_argument('--crop', type=bool, default=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Train, run, and deploy your GANs.')
    subparsers = parser.add_subparsers(dest='method')
    train_parser = subparsers.add_parser('train')
    build_parser = subparsers.add_parser('build')
    serve_parser = subparsers.add_parser('serve')
    common(train_parser)
    common(build_parser)
    common(serve_parser)
    train_parser.add_argument('--epochs', type=int, default=1000)
    train_parser.add_argument('--save_every', type=int, default=10)
    train_parser.add_argument('--use_hc_io', action='store_true')
    train_parser.add_argument('--frame_sample', type=str, default=None)

    return parser.parse_args()
