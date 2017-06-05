import hypergan as hg
import tensorflow as tf
from hypergan.gan_component import ValidationException
from hypergan.loaders.image_loader import ImageLoader

def fixture_path():
    return ""
class ImageLoaderTest(tf.test.TestCase):
    def test_constructor(self):
        with self.test_session():
            loader = ImageLoader(32)
            self.assertEqual(loader.batch_size, 32)

    def test_load_non_existent_path(self):
        with self.assertRaises(ValidationException):
            loader = ImageLoader(32)
            loader.load("/tmp/nonexistentpath")
            
    def test_load_fixture(self):
        with self.test_session():
            loader = ImageLoader(32)
            x, y = loader.load(fixture_path(), width=4, height=4)
            self.assertEqual(y.get_shape(), [])
            self.assertEqual(int(x.get_shape()[1]), 4)
            self.assertEqual(int(x.get_shape()[2]), 4)

    def test_load_fixture(self):
        with self.test_session():
            loader = ImageLoader(32) #TODO crop=true?
            x, y = loader.load(fixture_path(), width=2, height=2)
            self.assertEqual(y.get_shape(), [])
            self.assertEqual(int(x.get_shape()[1]), 2)
            self.assertEqual(int(x.get_shape()[2]), 2)


    def test_load_fixture_resize(self):
        with self.test_session():
            loader = ImageLoader(32) #TODO crop=true?
            x, y = loader.load(fixture_path(), width=8, height=8, resize=True)
            self.assertEqual(y.get_shape(), [])
            self.assertEqual(int(x.get_shape()[1]), 8)
            self.assertEqual(int(x.get_shape()[2]), 8)


    def test_load_fixture_single(self):
        with self.test_session():
            loader = ImageLoader(32) #TODO crop=true?
            x, y = loader.load(fixture_path('images'), width=4, height=4)
            self.assertEqual(y.get_shape(), [])
            self.assertEqual(int(x.get_shape()[1]), 4)
            self.assertEqual(int(x.get_shape()[2]), 4)

    def test_load_fixture_single(self):
        with self.test_session():
            loader = ImageLoader(32) #TODO crop=true?
            x, y = loader.load(fixture_path(), width=4, height=4)
            self.assertEqual(loader.file_count, 2)

    def test_load_fixture_single_count(self):
        with self.test_session():
            loader = ImageLoader(32) #TODO crop=true?
            x, y = loader.load(fixture_path('images'), width=4, height=4)
            self.assertEqual(loader.file_count, 1)

if __name__ == "__main__":
    tf.test.main()
