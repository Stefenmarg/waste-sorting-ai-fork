# Runs the full 4-model comparison once per seed,
# each in its own output folder.

source ./.venv/bin/activate


python ./cli.py --data-root ../data --output-dir ../output-seed-1 --seed 1 > run_seed_1.log 2>&1
python ./cli.py --data-root ../data --output-dir ../output-seed-2 --seed 2 > run_seed_2.log 2>&1
python ./cli.py --data-root ../data --output-dir ../output-seed-3 --seed 3 > run_seed_3.log 2>&1
python ./cli.py --data-root ../data --output-dir ../output-seed-4 --seed 4 > run_seed_4.log 2>&1

