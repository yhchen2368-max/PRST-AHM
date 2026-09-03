"""Port of MRST ``dispInfo``: print the dimensions of the Cartesian region of
the VOI grid (used to guide the user-specified well-region indices)."""


def dispInfo(GV):
    """Print the dimensions of the Cartesian region of the VOI grid ``GV``."""
    GV2 = GV['surfGrid']
    print('    -------------------------------------------------------------------------')
    print(f'  | * Info: The dimensions of Cartesian region of GV is ny = {GV2["cartDims"][1]:2.0f}, nz = {GV["layers"]["num"]:2.0f}.    |')
    print('  |         Users should specify the logical indices of well region in       |')
    print(f'  |         y-z plane within the range of   1 < Ind(1) < Ind(2) < {GV2["cartDims"][1]:2.0f}        |')
    print(f'  |                                         1 < Ind(3) < Ind(4) < {GV["layers"]["num"]:2.0f}        |')
    print('    -------------------------------------------------------------------------')
