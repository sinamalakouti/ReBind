from pathlib import Path


from dataset.rolebench import parse_action_triplet


root_dir = Path("<YOUR DATA DIR>/data")


def get_qa_eval_json_path(triplet, LLM):
    _, relation, _ = parse_action_triplet(triplet)
    relation = relation.replace(" ", "_")
    relation_dir = root_dir / relation
    triplet_dir = relation_dir / triplet
    prompt_dir = triplet_dir / "prompts" / LLM
    if not prompt_dir.exists():
        prompt_dir.mkdir(parents=True, exist_ok=True)
    eval_json_path = prompt_dir / "evaluation_qa.json"

    return eval_json_path


def get_data_dir(triplet, T2I, LLM):
    _, relation, _ = parse_action_triplet(triplet)
    relation = relation.replace(" ", "_")
    relation_dir = root_dir / relation
    triplet_dir = relation_dir / triplet
    prompt_dir = triplet_dir / "prompts" / LLM
    image_dir = triplet_dir / T2I
    prompt_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    return triplet_dir, prompt_dir, image_dir


def get_output_dir(output_dir, triplets, exp_name):
    if type(triplets) == list:
        target_tiplets_output = "_".join(triplets)
    else:
        target_tiplets_output = triplets
    output_dir = Path(output_dir, target_tiplets_output)
    output_dir = Path(output_dir, exp_name)

    return output_dir
