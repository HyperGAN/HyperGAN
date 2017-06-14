import hypergan as hg
import hyperchamber as hc
import tensorflow as tf
import os
from hypergan.search.random_search import RandomSearch

from tests.inputs.image_loader_test import fixture_path
from tests.mocks import MockDiscriminator, mock_gan
import shutil


class RandomSearchTest(tf.test.TestCase):
    def test_overrides(self):
        rs = RandomSearch({"trainer": "test"})
        self.assertEqual('test', rs.options['trainer'])

    def test_range(self):
        rs = RandomSearch({})
        self.assertTrue(isinstance(rs.range(), list))

    def test_trainers(self):
        rs = RandomSearch({})
        self.assertTrue(rs.trainer()["class"] != None)

    def test_random_config(self):
        rs = RandomSearch({})
        self.assertTrue(rs.random_config()['trainer']["class"] != None)

if __name__ == "__main__":
    tf.test.main()
