# Copyright (c) Microsoft. All rights reserved.

from enum import IntEnum
class TaskType(IntEnum):
    Classification = 1
    Regression = 2
    Ranking = 3
    Span = 4
    SeqenceLabeling = 5
    MaskLM = 6
    Adversarial = 7

class DataFormat(IntEnum):
    PremiseOnly = 1
    PremiseAndOneHypothesis = 2
    PremiseAndMultiHypothesis = 3
    MRC = 4
    Seqence = 5
    MLM = 6

class EncoderModelType(IntEnum):
    BERT = 1
    ROBERTA = 2
    XLNET = 3
    SAN = 4

class AdditionalFeatures(IntEnum):
    cue_marker = 1
    scope_markers = 2


def get_enum_name_from_repr_str(s):
    """
    when read from config, enums are converted to repr(Enum)
    :param s:
    :return:
    """
    ename = s.split('.')[-1].split(':')[0]
    if ename == 'None':
        ename = None
    return ename
