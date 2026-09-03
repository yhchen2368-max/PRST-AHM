"""MRST private precompute/read helper counterparts."""

from pathlib import Path


def cleanup_dialogue(precompDir):
    return Path(precompDir)


def precompute_dialogue(casenm, precompDir):
    return Path(precompDir) / str(casenm)


def get_precomputed_diagnostics(casenm, steps, pdir):
    return {"case": casenm, "steps": steps, "dir": pdir}


def process_restart_diagnostics(casenm, **kwargs):
    return {"case": casenm, **kwargs}


def process_states_diagnostics(problem, **kwargs):
    return {"problem": problem, **kwargs}


def read_and_prepare_for_post_processor(casenm, steps=None, info=None, precomp=None):
    return None, {"case": casenm, "steps": steps, "info": info, "precomp": precomp}, None, None


def read_and_prepare_for_post_processor_mrst(problem, steps=None, info=None, precomp=None):
    model = problem.get("SimulatorSetup", {}).get("model", problem.get("model", {})) if isinstance(problem, dict) else {}
    G = model.get("G") if isinstance(model, dict) else getattr(model, "G", None)
    return G, {"problem": problem, "steps": steps, "info": info, "precomp": precomp}, G, None


def readwell_sol_data_for_post_processor(problem, **kwargs):
    return {"problem": problem, **kwargs}


cleanupDialogue = cleanup_dialogue
precomputeDialogue = precompute_dialogue
getPrecomputedDiagnostics = get_precomputed_diagnostics
processRestartDiagnostics = process_restart_diagnostics
processStatesDiagnostics = process_states_diagnostics
readAndPrepareForPostProcessor = read_and_prepare_for_post_processor
readAndPrepareForPostProcessorMRST = read_and_prepare_for_post_processor_mrst
readwellSolDataForPostProcessor = readwell_sol_data_for_post_processor

