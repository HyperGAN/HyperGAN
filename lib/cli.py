import argparse

def common(parser):
    parser.add_argument('directory', action='store', type=str, description='The location of your data.  Subdirectories are treated as different classes.  You must have at least 1 subdirectory.')
    parser.add_argument('--size', '-s', type=str, default='64x64x3', description='Size of your data.  For images it is widthxheightxchannels.')
    parser.add_argument('--batch_size', '-b', type=int, default=32, description='Number of samples to include in each batch.  If using batch norm, this needs to be preserved when in server mode')
    parser.add_argument('--config', '-c', type=str, default=None, description='The name of the config.  This is used for loading/saving the model and configuration.')
    parser.add_argument('--device', '-d', type=str, default='/gpu:0', description='In the form "/gpu:0", "/cpu:0", etc.  Always use a GPU (or TPU) to train')
    parser.add_argument('--format', '-f', type=str, default='png', description='jpg or png')
    parser.add_argument('--crop', type=bool, default=True, description='If your images are perfectly sized you can skip cropping.')
    parser.add_argument('--use_hc_io', type=bool, default=False, description='Set this to no unless you are feeling experimental.')
    parser.add_argument('--epochs', type=int, default=10000, description='The number of iterations through the data before stopping training.')
    parser.add_argument('--save_every', type=int, default=10, description='Saves the model every n epochs.')
    parser.add_argument('--frame_sample', type=str, default=None, description='Frame sampling is used for video creation.')

def parse_args():
    parser = argparse.ArgumentParser(description='Train, run, and deploy your GANs.')
    subparsers = parser.add_subparsers(dest='method')
    train_parser = subparsers.add_parser('train')
    build_parser = subparsers.add_parser('build')
    serve_parser = subparsers.add_parser('serve')
    common(train_parser)
    common(build_parser)
    common(serve_parser)

    return parser.parse_args()
