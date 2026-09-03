$env:SPE9_MAX_STEPS='0'
$env:SPE9_VERBOSE='0'
$env:SPE9_NL_MAXIT='15'
$env:SPE9_LIN_TOL='1e-4'
Set-Location 'C:\Users\junji\Desktop\github\Cgnet'
python -u scripts\run_spe9_block_cpr.py
