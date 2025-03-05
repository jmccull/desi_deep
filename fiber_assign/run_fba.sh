module purge
source /global/common/software/desi/desi_environment.sh main
module load desisurveyops
export SKYHEALPIXS_DIR=/global/cfs/cdirs/desi/target/skyhealpixs/v1

mv ./fba_output/*.yaml ./
rm -r ./fba_output/*
mv ./*.yaml ./fba_output

python desi_fba_tertiary_9999.py --yamlfn ./fba_output/tertiary-config-9999.yaml
