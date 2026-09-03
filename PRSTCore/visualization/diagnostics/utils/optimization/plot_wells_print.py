"""MRST ``plotWellsPrint.m`` counterpart."""


def plot_wells_print(G, W, D=None):
    del D
    return [{"name": w.get("name"), "cells": w.get("cells")} for w in W]


plotWellsPrint = plot_wells_print

