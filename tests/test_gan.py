import hypergan as hg
from mocks import MockDiscriminator, mock_gan

class TestGan:
    def test_hg_gan(self):
        assert type(mock_gan()) == hg.gans.standard_gan.StandardGAN

    def test_can_create_default(self):
        config = hg.Configuration.load('default.json')
        assert type(mock_gan()) == hg.gans.standard_gan.StandardGAN

