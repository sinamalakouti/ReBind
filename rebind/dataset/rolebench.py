import re

ACTION_RELATION_PROMPT_TEMPLATE = f"What action is the {{subject}} performing in relation to the {{object}}? Your answer has to be only one of the follwoing options: {{options}}. Do not return extra text. The performed action is "


RELATIONS = [
    "chasing",
    "riding",
    "throwing",
    "holding",
    "following",
    "feeding",
    "pulling",
    "lifting",
    "carrying",
    "kissing",
]

# RELATIONS = ["lifting"]
rolebench_data = {
    "chasing": {
        "frequent": "cat_chasing_mouse",
        "rare": "mouse_chasing_cat",
        "passive": ["boy_chasing_mouse"],
        "active": ["mouse_chasing_boy"],
    },
    "riding": {
        "frequent": "astronaut_riding_horse",
        "rare": "horse_riding_astronaut",
        "passive": ["bear_riding_horse"],
        "active": ["horse_riding_bear"],
    },
    "throwing": {
        "frequent": "boy_throwing_puppy",
        "rare": "puppy_throwing_boy",
        "passive": ["boy_throwing_cat"],
        "active": ["puppy_throwing_cat"],
    },
    "holding": {
        "frequent": "grandpa_holding_doll",
        "rare": "doll_holding_grandpa",
        "passive": ["man_holding_grandpa"],
        "active": ["doll_holding_baby"],
    },
    "following": {
        "frequent": "lion_following_cow",
        "rare": "cow_following_lion",
        "passive": ["person_following_dog"],
        "active": ["cow_following_person"],
    },
    "feeding": {
        "frequent": "woman_feeding_baby",
        "rare": "baby_feeding_woman",
        "passive": ["robot_feeding_woman"],
        "active": ["baby_feeding_doll"],
    },
    "kissing": {
        "frequent": "mother_kissing_baby",
        "rare": "baby_kissing_mother",
        "passive": ["daughter_kissing_mother"],
        "active": ["baby_kissing_doll"],
    },
    "pulling": {
        "frequent": "man_pulling_dog",
        "rare": "dog_pulling_man",
        "passive": ["boy_pulling_dog"],
        "active": ["dog_pulling_sled"],
    },
    "lifting": {
        "frequent": "zoo_trainer_lifting_monkey",
        "rare": "monkey_lifting_zoo_trainer",
        "passive": ["robot_lifting_trainer"],
        "active": ["monkey_lifting_robot"],
    },
    "carrying": {
        "frequent": "fireman_carrying_scientist",
        "rare": "scientist_carrying_fireman",
        "passive": ["robot_carrying_fireman"],
        "active": ["scientist_carrying_girl"],
    },
}


RELATION_BASE_PROMPT_TEMPLATE = {
    f"chasing": f"{{subject}} chasing {{object}}",
    f"feeding": f"{{subject}} feeding food to {{object}}",
    f"riding": f"{{subject}} riding on {{object}}",
    f"throwing": f"{{subject}} throwing a ball to {{object}}",
    # f"catching": f"{{subject}} catching {{object}}.",
    f"pulling": f"{{subject}} pulling {{object}}",
    f"pushing": f"{{subject}} pushing {{object}}",
    f"lifting": f"{{subject}} lifting {{object}}",
    # f"leading": f"{{subject}} leading {{object}}.",
    f"carrying": f"{{subject}} carrying {{object}}",
    f"holding": f"{{subject}} holding {{object}}",
    f"following": f"{{subject}} following {{object}}",
    f"kissing": f"{{subject}} kissing {{object}}",
}


def rel2str(rel):
    if rel == "pointing":
        return "pointing to"
    if rel == "looking":
        return "looking at"
    return rel


def parse_action_triplet(subject_rel_obj, novel_relation=False):
    pattern = f"^(.+)_({('|'.join(RELATIONS))})_(.+)$"
    match = re.match(pattern, subject_rel_obj)
    if not match:
        if not novel_relation:
            raise ValueError(f"No valid relation found in: {subject_rel_obj}")
        else:
            raise Warning(f"No valid relation found in: {subject_rel_obj}")
    subject, relation, obj = match.groups()
    subject = subject.replace("_", " ").lower()
    obj = obj.replace("_", " ").lower()
    return subject.lower(), relation.lower(), obj.lower()


def get_evaluation_triplets(triplet):
    eval_triplets = [
        triplet,
        reverse_triplet(triplet),
    ]
    return eval_triplets


def reverse_triplet(triplet):
    subj, relation, obj = parse_action_triplet(triplet)
    subj = subj.replace(" ", "_")
    obj = obj.replace(" ", "_")
    return f"{obj}_{relation}_{subj}"
