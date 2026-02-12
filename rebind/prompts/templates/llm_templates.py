def get_intermediate_prompts_template() -> dict:
    """
    Returns the template for generating intermediate prompts.
    """
    system_prompt = (
        "You are an expert at breaking down unusual image prompts into easier-to-generate "
        "intermediate steps. You understand that changes are counted by comparing triplet differences: "
        "1) Different subject = 1 change, 2) Different object = 1 change, 3) Different relation = 1 change. "
        "Generate diverse prompts covering all possible types of changes."
    )

    user_prompt = """Generate intermediate triplets:
1. Visually similar to target (maintain similar actions/poses)
2. Easy for image generation models to understand
3. Must include self-relation prompts (e.g., subject_relation_subject, object_relation_object)
4. Cover diverse types of changes systematically

IMPORTANT INSTRUCTIONS:
1. You must generate triplets in format of  <subject>_<relation>_<object>. number of changes is difened on how many of triplet constituents are chagned compared to the input triplet. (e.g., cat following cat has 3 changes compared to mouse chasing cat)
2. Generate 2-5 triplets for each nubmer of changes. Different types of changes are
    - one change:  change subject only, change object only, or chagne relation only
    - two changes: change subject and object, change subject and relation, change object and relation
    - three changes: change subject, object and relation
3. Generate a diverse set of triplets: 
    - objects can change to something similar, synonym, or with the other object in the triplet if not the same. For "one change" ensure  including symmetic triplets: <subject>_<relation>_<subject> and <object>_<relation>_<object>
    - relations can change to synonymous relation (e.g. following for chasing) or relations that are visually very similar (e.g., standing/sitting on for riding on)
3. For more complex scenes (two changes), consider meaningful combinations of changes that maintain visual coherence and be visually similar to the target triplet in terms of object, action, etc. 
4. You must avoid duplicate tiplets even with different textual prompt(i.e. same triplet will have same entities and relation). At-least one of the subject, object or relation should be different.
5. Scenes must be visually feasible to generate (e.g. mouse chasing itself is not feasible)


INPUT FORMAT:
triplet: str (in format of <subject>_<relation>_<object>)    
prompt: str (the simple prompt for the triplet)

Output format:
{{
    "one": [
        {{
            "triplet": str,
            "prompt": str,
            "changes": str,
            "reasoning": str   # Why easier + How transforms to target + Why visually similar
        }}
    ],
    "two": [],
    "three": []
}}

EXAMPLE:
Input:
    triplet: mouse_chasing_cat
    prompt: A photo of a mouse chasing a cat

Output:
{{
    "one": [
        {{
            "triplet": "mouse_chasing_mouse",
            "prompt": "A photo of a mouse chasing another mouse",
            "changes": "changed object only",
            "reasoning": "Same chasing action and mouse behavior, just replace second mouse with cat"
        }},
        {{
            "triplet": "cat_chasing_cat", 
            "prompt": "A photo of a cat chasing another cat",
            "changes": "changed subject only",
            "reasoning": "Same chasing dynamics with correct cat movement, replace first cat with mouse"
        }},
        {{
            "triplet": "mouse_following_cat",
            "prompt": "A photo of a mouse following a cat",
            "changes": "changed relation only",
            "reasoning": "Same entities and positioning, just adjust action to chasing"
        }},
        {{
            "triplet": "rat_chasing_cat",
            "prompt": "A photo of a rat chasing a cat",
            "changes": "changed subject only",
            "reasoning": "Same chasing action and mouse behavior, just replace second mouse with cat"
        }}
    ],
    "two": [
        
        {{
            "triplet": "mouse_chasing_dog",
            "prompt": "A photo of a mouse chasing a dog",
            "changes": "changed object + changed relation", 
            "reasoning": "Maintains mouse as pursuer but with different target and action"
        }},
        {{
            "triplet": "rat_chasing_dog",
            "prompt": "A photo of a rat chasing a dog",
            "changes": "changed object + changed subject", 
            "reasoning": "Maintains mouse as pursuer but with different target and action"
        }},
        {{
            "triplet": "rat_following_cat",
            "prompt": "A photo of a rat following a cat",
            "changes": "changed relation + changed subject", 
            "reasoning": "Maintains mouse as pursuer but with different target and action"
        }}
    ],
    "three": [
    {{
            "triplet": "dog_following_mouse",
            "prompt": "A photo of a dog following a mouse",
            "changes": "changed subject + changed relation + changed object",
            "reasoning": "Similar pursuit dynamic but with different subject and action"
        }},
        {{
            "triplet": "dog_following_bird",
            "prompt": "A photo of a dog following a bird",
            "changes": "changed subject + object + relation",
            "reasoning": "Similar pursuit scene that can be transformed to target"
        }}
    ]
}}

Now generate for:
triplet: {triplet}
prompt: {prompt}"""

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }


def get_intermediate_active_passive_triplet_template() -> dict:
    system_prompt = (
        "You are an expert at breaking down unusual image prompts into easier-to-generate intermediate steps."
        " Your task is to create two distinct intermediate active-passive pairs for each given triplet."
        " Intermediate steps must be realistic, meaning the subject must be physically capable of performing the action."
        " For each generated triplet, determine whether the relationship is frequent or rare."
        " Then, generate a contrast pair by reversing the subject and object."
    )

    user_prompt = """Generate intermediate active and passive triplets that are easy for image models to generate.

#### IMPORTANT INSTRUCTIONS:
1. Each triplet must follow the format <subject>_<relation>_<object>.
2. Generate 2-3 passive and active triplets for each target triplet:
   - Active triplets: Keep the subject from rare triplet, change the object (e.g., for "mouse_chasing_cat", "mouse_chasing_boy" keeps "mouse")
   - Passive triplets: Keep the object from rare triplet, change the subject (e.g., for "mouse_chasing_cat", "boy_chasing_cat" keeps "cat")
3. Choosing Intermediate Objects such that the intermediate triplet is easier to generate. Some instruction on how to choose the intermediate objects:
   a) Use must atleast choose one object from different categories. some examples: 
      - Animal → Human (e.g., mouse_chasing_cat → mouse_chasing_boy)
      - Animal → Object (e.g., mouse_chasing_cat → mouse_chasing_cheese)
      - Human → Object (e.g., baby_feeding_woman → robot_feeding_woman)
   
   b) Prefer frequent/common interactions when possible:
      - e.g., "dog_chasing_cat" is a good passive for mouse_chasing_catas it's a common interaction
   
   c) Can use unrelated but logical objects:
      - e.g., "mouse_chasing_cheese" is valid
      - e.g., "cat_following_fish" is not valid
   
   d) Ensure physical plausibility:
      - Objects should make sense size-wise
      - Action should be physically possible
      - e.g., "mouse_lifting_elephant" is not valid or "dog_feeding_woman" doesn't make sense as generating a photo of a dog feeding a woman is extremely difficult
    e) Creativity is allowed **if the scenario is not outright impossible** (e.g., `mouse_chasing_cheese` is okay, but `cart_pulling_boy` is not).
    f) do not use objects as intermediate objects that are similar to the other object in the triplet that make the object indistinguishable. 
    g) It's important to diversify, for instance scientist_carrying_girl and scientist_carrying_boy are too similar telling diffusion model that it can only carry kids. 
4. Each pair should contain a classification of **frequent** (common interaction) or **rare** (uncommon interaction).
5. In order to determine if <subject>_<relation>_<object> is frequent or rare, compare it against its dual triplet <object>_<relation>_<subject>: 
    - if the triplet is relatively more common in real life, animation, etc than its dual then its frequent even if its uncommon itself
    - if the triplet and the subject is more believable to take that action compared to object, then its frequent
6. Each triplet should use **unique objects**—avoid symmetrical cases (e.g., `mouse_chasing_mouse`) or ambiguous setups (e.g., `ball_throwing_ball`).
7. Avoid duplicate intermediate triplets for the same target.

#### OUTPUT FORMAT:

{{
    "target_triplet": {{
        "passive": [
            {{"triplet": "passive_triplet", "type": "frequent/rare"}},  {{"triplet": "passive_triplet", "type": "frequent/rare"}}, {{"triplet": "passive_triplet", "type": "frequent/rare"}}
        ],
        "active": [
            {{"triplet": "active_triplet", "type": "frequent/rare"}}, {{"triplet": "active_triplet", "type": "frequent/rare"}}, {{"triplet": "active_triplet", "type": "frequent/rare"}}
        ]
    }}
}}

The output format and intermediate objects/triplets should follow the format shown in the examples.

EXAMPLES:
#### Input:
    triplet: mouse_chasing_cat
    prompt: A photo of one mouse chasing one cat

#### Output:
{{
    "mouse_chasing_cat": {{
        "passive": [
            {{"triplet": "boy_chasing_cat", "type": "frequent"}},
            {{"triplet": "dog_chasing_cat", "type": "frequent"}}
        ],
        "active": [
            {{"triplet": "mouse_chasing_boy", "type": "rare"}},
            {{"triplet": "mouse_chasing_cheese", "type": "rare"}}
        ]
    }}
}}


#### Input:
    triplet: horse_riding_astronaut
    prompt: A photo of one horse riding one astronaut

#### Output:
{{
    "horse_riding_astronaut": {{
        "passive": [
            {{"triplet": "bear_riding_astronaut", "type": "frequent"}},
            {{"triplet": "dog_riding_astronaut", "type": "rare"}}
        ],
        "active": [
            {{"triplet": "horse_riding_bear", "type": "rare"}},
            {{"triplet": "horse_riding_dog", "type": "rare"}}
        ]
    }}
}}


#### Input:
    triplet: puppy_throwing_boy
    prompt: A photo of one puppy throwing ball for one boy

#### Output:
{{  
    "puppy_throwing_boy": {{
        "passive": [
            {{"triplet": "man_throwing_boy", "type": "frequent"}}
        ],
        "active": [
            {{"triplet": "puppy_throwing_cat", "type": "frequent"}}
        ]
    }}
}}


#### Input:
    triplet: doll_holding_grandpa
    prompt: A photo of one doll holding one grandpa

#### Output:
{{  
    "doll_holding_grandpa": {{
        "passive": [
            {{"triplet": "man_holding_grandpa", "type": "frequent"}}, {{"triplet": "monkey_holding_grandpa", "type": "frequent"}}
        ],
        "active": [
            {{"triplet": "doll_holding_baby", "type": "rare"}, {{}}}
        ]
    }}
}}


#### Input:
    triplet: lion_following_cow
    prompt: A photo of one lion following one cow from its back

#### Output:
{{  
    "lion_following_cow": {{
        "passive": [
            {{"triplet": "boy_following_cow ", "type": "frequent"}}
        ],
        "active": [
            {{"triplet": "lion_following_person", "type": "rare"}}, {{"triplet": "lion", "type": "frequent"}} # person is different object than lion and lamb is smaller scale
        ]
    }}
}}


#### Input:
    triplet: baby_feeding_woman
    prompt: A photo of one baby feeding one woman

#### Output:
{{
    "baby_feeding_woman": {{
        "passive": [
            {{"triplet": "robot_feeding_woman", "type": "frequent"}}
        ],
        "active": [
            {{"triplet": "baby_feeding_doll", "type": "frequent"}} # it's more common to see baby feed doll vs doll feeding baby
        ]
    }}
}}


#### Input:
    triplet: baby_kissing_mother
    prompt: A photo of one baby kissing one mother

#### Output:
{{
    "baby_kissing_mother": {{
        "passive": [
            {{"triplet": "daughter_kissing_mother", "type": "frequent"}}
        ],
        "active": [
            {{"triplet": "baby_kissing_doll", "type": "frequent"}} # it's more common to see baby kiss doll vs doll kissing baby
        ]
    }}
}}

#### Input:
    triplet: dog_pulling_man
    prompt: A photo of one dog pulling one man

#### Output:
{{
    "dog_pulling_man": {{
        "passive": [
            {{"triplet": "boy_pulling_dog", "type": "frequent"}}
        ],
        "active": [
            {{"triplet": "dog_pulling_sled", "type": "rare"}}
        ]
    }}
}}
    
#### Input:
    triplet: monkey_lifting_zoo_trainer
    prompt: A photo of one monkey lifting one zoo trainer

#### Output:
{{
    "monkey_lifting_zoo_trainer": {{
        "passive": [
            {{"triplet": "robot_lifting_zoo_trainer", "type": "frequent"}}
        ],
        "active": [
            {{"triplet": "monkey_lifting_robot", "type": "rare"}}
        ]
    }}
}}

#### Input:
    triplet: scientist_carrying_fireman
    prompt: A photo of one scientist carrying one fireman

#### Output:
{{
    "scientist_carrying_fireman": {{
        "passive": [
            {{"triplet": "gorilla_carrying_fireman", "type": "rare"}}, {{"triplet": "robot_carrying_fireman", "type": "rare"}} #fireman is more associated with carrying
        ],
        "active": [
            {{"triplet": "scientist_carrying_robot", "scientist_carrying_child", "type": "frequent"}}
        ]
    }}
}}

Now generate for:
    input: 
        triplet: {target_triplet}
        prompt: {target_prompt}
"""

    return {"system_prompt": system_prompt, "user_prompt": user_prompt}


def create_prompt_expander_template() -> dict:
    system_prompt = (
        "You are an expert at crafting precise, relation-focused descriptions for text-to-image models. "
        "Your specialty is describing the key entities and their interactions in a way that emphasizes "
        "their roles, poses, and spatial relationships. Keep descriptions concise yet vivid, using creative "
        "visualization when needed (e.g., 'apple with tiny legs running away' to show case ' mouse chasing cat)."
    )

    user_prompt = """Transform this prompt into a focused description that emphasizes:
1. Key entities and their states:
   - Poses, expressions, and orientations that match their roles
   - Distinctive features that support the action/relation
2. Spatial positioning that clearly shows the relationship. Use proper spatial perpositions (e.g., above, below, behind, in front of, etc.) to describe the relative postion of the entities (e.g., 'a mouse is behind a cat running towards the cat' to express 'mouse is chasing cat')
3. Simple creative elements if needed to clarify the interaction
4. Images must be photorealistic. Include this in the description. 
5. ensure count of entities are correct (e.g., 'a mouse is chasing a cat' should have one mouse and one cat)
6. avoid redundant descriptions
 

Input prompt: {prompt}

Provide a concise, clear description focusing on the entities and their relationship. Avoid background details unless essential to the interaction. Keep it brief but vivid."""

    return {"system_prompt": system_prompt, "user_prompt": user_prompt}
