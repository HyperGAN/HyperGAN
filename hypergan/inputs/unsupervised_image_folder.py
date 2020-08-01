from PIL import Image
import os
import torch
import torchvision

class UnsupervisedImageFolder(torchvision.datasets.vision.VisionDataset):
    """
    Loads everything possible from a folder
    """
    def __init__(self, root, transform=None,
                 target_transform=None, is_valid_file=None, mode=None):
        extensions = torchvision.datasets.folder.IMG_EXTENSIONS
        if mode == "RGBA":
            loader = self.rgba_loader
        else:
            loader = torchvision.datasets.folder.default_loader
        super(UnsupervisedImageFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        samples = self._make_dataset(self.root, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.samples = samples


    def _make_dataset(self, dir, extensions=None, is_valid_file=None):
        images = []
        dir = os.path.expanduser(dir)
        if not ((extensions is None) ^ (is_valid_file is None)):
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x):
                return torchvision.datasets.folder.has_file_allowed_extension(x, extensions)
        d = dir
        for root, _, fnames in sorted(os.walk(d, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    images.append(path)

        return images

    def __getitem__(self, index):
        sample = self.loader(self.samples[index])
        if self.transform is not None:
            sample = self.transform(sample)

        return [sample]

    def __len__(self):
        return len(self.samples)


    def rgba_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGBA')

