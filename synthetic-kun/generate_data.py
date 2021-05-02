import argparse
import json
import pickle

from clearml import Task
from generators import SyntheticDataGenerator


def main(task, generator_task_id, extra):
    gen_task = Task.get_task(task_id=generator_task_id)
    generator_path = gen_task.artifacts["generator"].get_local_copy()
    with open(generator_path, "rb") as f:
        generator_obj = pickle.load(f)  # type: SyntheticDataGenerator
    generated = generator_obj.generate(task, **extra)

    with open('dataset.csv', "w") as f:
        for k, v in generated.items():
            f.write("%d,%d,%d\n" % (k[0], k[1], int(v * 2.5 + 2.5)))

    task.upload_artifact('dataset', artifact_object='dataset.csv', delete_after_upload=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_id', type=str, action='store', required=True)
    parser.add_argument('--parameters', type=str, action='store', required=False, default="{}")
    parser.add_argument('--final', action='store_true')

    args = parser.parse_args()
    extra_args = json.loads(args.parameters)

    tags = ["GenerateData"]
    if args.final:
        tags.append("Final")

    task = Task.init(project_name="Synthetic data generators", task_name="generate_data", tags=tags,
                     reuse_last_task_id=True, task_type=Task.TaskTypes.inference)

    main(task, args.task_id, extra_args)