import inflect
import re

from dataset.rolebench import (
    ACTION_RELATION_PROMPT_TEMPLATE,
    RELATION_BASE_PROMPT_TEMPLATE,
    parse_action_triplet,
)


def get_base_prompt_from_action_triplet(
    action_triplet, prefix="A photo of ", add_determiners=True, novel_relation=False):
    p = inflect.engine()
    sub, relation, obj = parse_action_triplet(action_triplet, novel_relation=novel_relation)
    if "_" in sub:
        sub = sub.replace("_", " ")
    if "_" in obj:
        obj = obj.replace("_", " ")
    if add_determiners:
        sub = "one " + sub
        obj = "one " + obj
    if relation not in RELATION_BASE_PROMPT_TEMPLATE:
        raise ValueError(f"Unknown relation: {relation}")
    return prefix + RELATION_BASE_PROMPT_TEMPLATE[relation].format(
        subject=sub, object=obj
    )


def get_action_relation_question(subject_rel_obj):
    from dataset.rolebench import RELATIONS

    sub, _, obj = parse_action_triplet(subject_rel_obj)
    options = ", ".join(RELATIONS)
    question = ACTION_RELATION_PROMPT_TEMPLATE.format(
        subject=sub, object=obj, options=options
    )
    return question
