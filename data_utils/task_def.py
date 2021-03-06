# Copyright (c) Microsoft. All rights reserved.

from enum import IntEnum
from ast import literal_eval
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
    cue_indicator = 1
    scope_indicator = 2
    sid = 3


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

def get_additional_feature_names(l):
    feature_names = []
    elms = l.split(',')
    for elm in elms:
        elm = elm.strip('[]')
        feature_names.append(get_enum_name_from_repr_str(elm))
    if len(feature_names) > 0:
        return feature_names
    else:
        return None
