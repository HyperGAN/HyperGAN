import hypergan as hg
import pytest

class TestParser:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.parser = hg.parser.Parser()

    def parse(self, string):
        return self.parser.parse_string(string)

    def test_parses_simple_string(self):
        line = "conv2d"
        assert self.parse(line) == ["conv2d", [], {}]

    def test_parses_args_int(self):
        line = "conv2d 256"
        assert self.parse(line) == ["conv2d", [256], {}]

    def test_parses_args_float(self):
        line = "conv2d 256.0"
        assert self.parse(line) == ["conv2d", [256.0], {}]

    def test_parses_args_str(self):
        line = "conv2d cat"
        assert self.parse(line) == ["conv2d", ["cat"], {}]

    def test_parses_options_int(self):
        line = "conv2d cat=2"
        assert self.parse(line) == ["conv2d", [], {"cat": 2}]

    def test_parses_options_float(self):
        line = "conv2d cat=2.0"
        assert self.parse(line) == ["conv2d", [], {"cat": 2.0}]

    def test_parses_options_str(self):
        line = "conv2d cat2=cat"
        assert self.parse(line) == ["conv2d", [], {"cat2": "cat"}]

    def test_parses_options_args_str(self):
        line = "conv2d cat cat=cat "
        assert self.parse(line) == ["conv2d", ["cat"], {"cat": "cat"}]

    def test_parses_options_messy_options(self):
        line = "conv2d cat2 = cat2 cat1=cat1"
        assert self.parse(line) == ["conv2d", [], {"cat2": "cat2", "cat1": "cat1"}]

    def test_parses_options_clobber_options(self):
        line = "conv2d options=1"
        assert self.parse(line) == ["conv2d", [], {"options": 1}]



