import argparse
import json
import pickle
from time import sleep

from clearml import Task
import datasets
import generators
from datasets import Dataset
from generators import SyntheticDataGenerator
import generate_data


def main(task: Task, dataset_name, generator_name, extra):
    if generator_name not in generators.name_to_class:
        raise Exception("Unknown generator name" + ", ".join(generators.name_to_class.keys()))
    dataset_obj = datasets.get_dataset(dataset_name)  # type: Dataset
    generator_obj = generators.name_to_class[generator_name](verbose=True)  # type: SyntheticDataGenerator
    generator_obj.build(task, dataset_obj, **extra)
    with open("generator.pickle", "wb") as f:
        pickle.dump(generator_obj, f)
    task.upload_artifact('generator', artifact_object='generator.pickle', delete_after_upload=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, action='store', required=True)
    parser.add_argument('--generator', type=str, action='store', required=True)
    parser.add_argument('--parameters', type=str, action='store', required=False, default="{}")
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--generate', action='store_true')

    args = parser.parse_args()
    extra_args = json.loads(args.parameters)

    tags = ["TrainGenerator"]
    if args.final:
        tags.append("Final")

    task = Task.init(project_name="Synthetic data generators", task_name="train_generator", tags=tags,
                     reuse_last_task_id=not args.final)

    main(task, args.dataset, args.generator, extra_args)
    task.flush(wait_for_uploads=True)

    # Let server parse our uploads
    sleep(20.0)

    if args.generate:
        generate_data.main(task, task.task_id, {})
