#!/bin/bash

python oad_recurrent_main.py
python make_proposal.py --yaml_path=yamls/canonical.yaml --json_name=main.json --load_model=main_thumos14_4_p0.025.pt
python convert.py --yaml_path=yamls/canonical.yaml --source=main.json --target=main.json
python oracle_hungarian.py --pred=main.json