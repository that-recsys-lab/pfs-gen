import argparse
import glob
import json
import os
from shutil import copyfile

from clearml import Task

import subprocess

import pandas as pd

from tools import to_clear_ml_params

config_template = """<?xml version="1.0"?>

<librec-auto>
	<random-seed>202001</random-seed>
	<thread-count>1</thread-count>
	<library src="system">default-algorithms.xml</library>
	
	<!-- DATA SECTION -->
	<data>
		<data-dir>./</data-dir>
		<format>UIR</format>
		<data-file format="text">dataset.csv</data-file>
	</data>
	
	<!-- SPLITTER SECTION -->
	<splitter>
		<model count="3">kcv</model>
		<dim>userfixed</dim>
		<ratio>0.8</ratio>
		<save>true</save>
	</splitter>
	
	<!-- ALGORITHM SECTION -->
	<!-- Using biased for demonstration purposes. -->
	<alg name="alg:{{RECOMMENDER_NAME}}">
		{{RECOMMENDER_PARAMS}}
	</alg>

	<!-- METRICS SECTION -->
	<metric>
		<ranking>true</ranking>
		<list-size>10</list-size>
		<class>ndcg,precision</class>
	</metric>
	
	<!-- POST-PROCESSING SECTION -->
	<post>
		<script lang="python3" src="system">
			<script-name>results_to_csv.py</script-name>
			<param name="option">all</param>
		</script> 
	</post> 
</librec-auto>
"""

DATASET_PATH = "dataset.csv"

CONFIG_NAME = "conf/config.xml"

COMMAND_TEMPLATE = f"python3 -m librec_auto run -t ./ -c config.xml -q"

LOG_FILE = "post/study-results-summary_*.csv"


def run_librec_auto_experiment(task, dataset_path, recommender, params):
    task.set_user_properties(**to_clear_ml_params(params))

    if not os.path.exists("post"):
        os.makedirs("post")

    if not os.path.exists("conf"):
        os.makedirs("conf")

    copyfile(dataset_path, DATASET_PATH)
    param_str = "\n".join(["<{0}>{1}</{0}>".format(k, v) for k, v in params.items()])
    config = config_template
    config = config.replace("{{RECOMMENDER_NAME}}", recommender)
    config = config.replace("{{RECOMMENDER_PARAMS}}", param_str)

    with open(CONFIG_NAME, "w") as f:
        f.write(config)
    task.upload_artifact('librec_auto_config', artifact_object=CONFIG_NAME)

    process = subprocess.Popen(COMMAND_TEMPLATE, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while True:
        line = process.stdout.readline()
        if not line:
            break
        # the real code does filtering here
        line = line.decode()
        task.logger.report_text(line.rstrip())
    process.wait()

    files = glob.glob(LOG_FILE)
    for f in files:
        b = pd.read_csv(f, sep=",")
        for c in b.columns:
            if c != "Experiment":
                task.logger.report_scalar(c, "Value", b[c][0], 0)
        print(b)


def main(task, dataset_task_id, recommender, extra):
    gen_task = Task.get_task(task_id=dataset_task_id)
    dataset_path = gen_task.artifacts["dataset"].get_local_copy()
    run_librec_auto_experiment(task, dataset_path, recommender, extra)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_id', type=str, action='store', required=True)
    parser.add_argument('--recommender', type=str, required=True)
    parser.add_argument('--parameters', type=str, action='store', required=False, default="{}")
    parser.add_argument('--final', action='store_true')

    args = parser.parse_args()
    extra_args = json.loads(args.parameters)

    tags = ["EvaluateRecommender"]
    if args.final:
        tags.append("Final")

    task = Task.init(project_name="Synthetic data generators", task_name="evaluate_recommender", tags=tags,
                     reuse_last_task_id=False)

    main(task, args.task_id, args.recommender, extra_args)
