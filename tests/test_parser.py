import hypergan as hg
import pytest

class TestParser:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.parser = hg.parser.Parser()

    def parse(self, string):
        return self.parser.parse_string(string).to_list()

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

    def test_parses_options_bool(self):
        line = "conv2d cat=true"
        assert self.parse(line) == ["conv2d", [], {"cat": True}]

    def test_parses_options_bool_false(self):
        line = "conv2d cat=false"
        assert self.parse(line) == ["conv2d", [], {"cat": False}]

    def test_parses_options_none(self):
        line = "conv2d cat=null"
        assert self.parse(line) == ["conv2d", [], {"cat": None}]

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

    def test_parses_configurable_param(self):
        line = "conv2d (conv2d test)"
        obj = self.parse(line)[1][0][0] #TODO why is this nested, should be just [1]
        assert obj.to_list() == ["conv2d", ["test"], {}]

    def test_parses_configurable_param_in_options(self):
        line = "conv2d options=(conv2d test)"
        assert self.parse(line)[2]["options"][0].to_list() == ["conv2d", ["test"], {}]

    def test_parses_size(self):
        line = "reshape 64*64*3"
        assert self.parse(line) == ["reshape", ["64*64*3"], {}]

    def test_parses_multiple_args(self):
        line = "concat layer i"
        assert self.parse(line) == ["concat", ["layer", "i"], {}]

    def test_parses_underscore_in_options(self):
        line = "identity a_b=3"
        assert self.parse(line) == ["identity", [], {"a_b": 3}]

    

