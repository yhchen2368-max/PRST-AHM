"""Port of MRST's ``hm`` module (mrst-2026a/hm): history matching.

Directory layout mirrors the MATLAB module one-for-one:

    hm/ad_tracer/{models,utils}   passive tracer transport models
    hm/utils/                     history-matching helpers
    hm/utils/evaluate/            objective / mismatch evaluation
    hm/utils/observed/            observed-data readers
    hm/utils/optimizer/           bound- and constraint-handling optimizers
    hm/test/                      the module's driver scripts
"""
