import pyparsing
from pyparsing import alphas, alphanums, delimitedList, replaceWith, nestedExpr, oneOf, pyparsing_common, Dict, Forward, Keyword, Group, Suppress, Word, ZeroOrMore

class Pattern:
    def __init__(self, tokens):
        self.__dict__.update(tokens.asDict())

    def to_list(self):
        return [self.layer_name, list(self.args), self.options]

class Parser:
    def __init__(self):
        FALSE = Keyword("false")
        NULL = Keyword("null")
        TRUE = Keyword("true")
        FALSE.setParseAction(replaceWith(False))
        NULL.setParseAction(replaceWith(None))
        TRUE.setParseAction(replaceWith(True))
        pattern = Forward()
        label = Word(alphas, alphanums+"_").setResultsName("layer_name")
        configurable_param = nestedExpr(content = pattern)
        arg = (NULL ^ FALSE ^ TRUE ^ pyparsing_common.number ^ (Word(alphanums+"*_") + ~ Word("=")) ^ configurable_param)
        args = arg[...].setResultsName("args")
        args.setParseAction(self.convert_list)
        options = Dict(Group(Word(alphas, alphanums) + Suppress("=") + arg))[...].setResultsName("options")
        options.setParseAction(self.convert_dict)
        pattern <<= label + args + options
        pattern.setParseAction(Pattern)
        self.pattern = pattern

    def convert_dict(self, s, l, toks):
        retv = {}
        for sublist in toks:
            retv[sublist[0]] = sublist[1]
        return retv

    def convert_list(self, s, l, toks):
        if len(toks) == 0:
            return [[]]

        return list(toks)

    def parse_string(self, string):

        parsed = self.pattern.parseString(string, parseAll=True)
        return parsed[0]
