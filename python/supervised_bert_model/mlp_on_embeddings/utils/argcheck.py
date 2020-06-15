import argparse


# Commandline parameter constrains
def check_int_positive(value):
    ivalue = int(value)
    if ivalue < 0:
         raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


def check_float_positive(value):
    ivalue = float(value)
    if ivalue < -1:
         raise argparse.ArgumentTypeError("%s is an invalid positive float value" % value)
    return ivalue


def check_emb_type(value):
    if value == '':
         raise argparse.ArgumentTypeError("Please specify embedding type(bert or xlmr)")
    if value not in ['bert','xlmr']:
         raise argparse.ArgumentTypeError("%s is an invalid embedding type(use bert or xlmr)" % value)
    return value
