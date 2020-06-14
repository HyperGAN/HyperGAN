import pyparsing
from pyparsing import alphas, alphanums, delimitedList, oneOf, pyparsing_common, Dict, Group, Suppress, Word, ZeroOrMore

class Parser:
    def to_options(self, options):
        retv = {}
        for sublist in options:
            retv[sublist[0]] = sublist[1]
        return retv

    def parse_string(self, string):
        label = Word(alphas, alphanums).setResultsName("layer_name")
        arg = (pyparsing_common.number | (Word(alphas, alphanums) + ~ Word("=")))
        args = arg[...].setResultsName("args")
        options = Dict(Group(Word(alphas, alphanums) + Suppress("=") + arg))[...].setResultsName("options")
        pattern =  label + args + options
        parsed = pattern.parseString(string, parseAll=True)
        retv = [parsed.layer_name, list(parsed.args), self.to_options(parsed.options)]
        return retv
