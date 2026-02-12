import inflect




def create_llm_eval_template() -> str:
    template = """# Your Role: Expert Visual Relationship Evaluator

## Objective: Generate atomic yes/no questions to evaluate if an image accurately depicts a given a prompt describing the image.

## Question Categories
1. Detection
   - Verify presence of key objects
2. Count
   - Verify count of key objects
3. Spatial
   - Evaluate physical positioning between objects and their locations
4. Pose
   - Assess body postures and motion states
5. Orientation
   - Check facing directions and alignment
6. Interaction 
   - Evaluate visual engagement (e.g. gaze, facial expression) and/or dynamic of objects

## Key Guidelines
1. Questions must be atomic (evaluate exactly one aspect)
2. ALL questions must be answerable with '(yes/no)'
3. Focus on directly observable visual elements
4. Avoid redundancy and subjective interpretations
5. Assign a numeric weight (1–5) to each category based on its importance for verifying the main relationship: 5 = Critical or essential for confirming the main relationship, 4 = Very important but slightly less critical, 3 = Moderately important, 2 = Somewhat important but not central, 1 = Peripheral or minimal importance
6. for each category, provide both question and statement (statement should match the question)
## Examples:

Example 1:

Input: A photo of a cat chasing a mouse
{{
    "detection": {{
        "weight": 5,
        "questions": [
            "Is there a cat visible in the image? (yes/no)",
            "Is there a mouse visible in the image? (yes/no)"
        ],
        "statements": [
            "There is a cat in the image",
            "There is a mouse in the image"
        ]
    }},
    "count": {{
        "weight": 3,
        "questions": [
            "Is there exactly one cat in the image? (yes/no)",
            "Is there exactly one mouse in the image? (yes/no)"
        ],
        "statements": [
            "There is exactly one cat in the image",
            "There is exactly one mouse in the image"
        ]
    }},
    "spatial": {{
        "weight": 4,
        "questions": [
            "Is the cat positioned behind the mouse, indicating it is chasing? (yes/no)",
            "Is the mouse positioned in front of the cat, indicating it is being chased? (yes/no)"
        ],
        "statements": [
            "The cat is positioned behind the mouse, indicating it is chasing",
            "The mouse is positioned in front of the cat, indicating it is being chased"
        ]
    }},
    "pose": {{
        "weight": 3,
        "questions": [
            "Is the cat depicted in a motion state, such as running or leaping forward? (yes/no)",
            "Is the mouse depicted in a fleeing state, such as running or leaping away? (yes/no)"
        ],
        "statements": [
            "The cat is depicted in a motion state, such as running or leaping forward",
            "The mouse is depicted in a fleeing state, such as running or leaping away"
        ]   
    }},
    "orientation": {{
        "weight": 4,
        "questions": [
            "Is the cat oriented toward the mouse, suggesting a chasing action? (yes/no)",
            "Is the mouse oriented away from the cat, suggesting it is fleeing? (yes/no)"
        ],
        "statements": [
            "The cat is oriented toward the mouse, suggesting a chasing action",
            "The mouse is oriented away from the cat, suggesting it is fleeing"
        ]   
    }},
    "interaction": {{
        "weight": 3,
        "questions": [
            "Is the cat's gaze or body alignment directed toward the mouse? (yes/no)"
        ],
        "statements": [
            "The cat's gaze or body alignment is directed toward the mouse"
        ]
    }}
}}

Example 2:
Input: A photo of a horse riding on an astronaut
{{
    "detection": {{
        "weight": 5,
        "questions": [
            "Is there a horse visible in the image? (yes/no)",
            "Is there an astronaut visible in the image? (yes/no)"
        ],
        "statements": [
            "There is a horse in the image",
            "There is an astronaut in the image"
        ]   
    }},
    "count": {{
        "weight": 3,
        "questions": [
            "Is there exactly one horse in the image? (yes/no)",
            "Is there exactly one astronaut in the image? (yes/no)"
        ],
        "statements": [
            "There is exactly one horse in the image",
            "There is exactly one astronaut in the image"
        ]
    }},
    "spatial": {{
        "weight": 5,
        "questions": [
            "Is the horse positioned above the astronaut, suggesting it is riding on them? (yes/no)",
            "Is the astronaut positioned beneath the horse? (yes/no)"
        ],
        "statements": [
            "The horse is positioned above the astronaut, suggesting it is riding on them",
            "The astronaut is positioned beneath the horse"
        ]
    }},
    "pose": {{
        "weight": 4,
        "questions": [
            "Is the astronaut's body in position suggesting being ridden (e.g., on hands and knees, crouched down, or bent over to support the horse's weight)? (yes/no)",
            "Is the horse in a riding position (e.g., positioned upright on its hind legs, or its hands/legs are on the astronaut)?(yes/no)"
        ],
        "statements": [
            "The astronaut's body is in position suggesting being ridden by the horse (e.g., on hands and knees, crouched down, or bent over to support the horse's weight)",
            "The horse is in a riding position (e.g., positioned upright on its hind legs, or its hands/legs are on the astronaut)"
        ]
    }},
    "orientation": {{
        "weight": 3,
        "questions": [
            "Is the orientation aligned between horse and astronaut (i.e., are they both facing/pointing in the same direction)? (yes/no)"
        ],
        "statements": [
            "The orientation between horse and astronaut is aligned (i.e., they are both facing/pointing in the same direction)"
        ]
    }},
    "interaction": {{
        "weight": 1,
        "questions": [
            "Is the horse's facial expression or body dynamic suggesting it's riding? (yes/no)",
            "Is the astronaut's facial expression or body dynamic suggesting it's being ridden? (yes/no)"   
        ],
        "statements": [
            "The horse's facial expression or body dynamic is suggesting it's riding",
            "The astronaut's facial expression or body dynamic is suggesting it's being ridden"
        ]
    }}
}}

Example 3:
Input: A photo of a lion following a cow
{{
  "detection": {{
    "weight": 5,
    "questions": [
      "Is there a lion  visible in the image? (yes/no)",
      "Is there a cow  visible in the image? (yes/no)"
    ],
    "statements": [
      "There is a lion in the image",
      "There is a cow in the image"
    ]
  }},
  "count": {{
    "weight": 3,
    "questions": [
      "Is there exactly one lion in the image? (yes/no)",
      "Is there exactly one cow in the image? (yes/no)"
    ],
    "statements": [
      "There is exactly one lion in the image",
      "There is exactly one cow in the image"
    ]
  }},
  "spatial": {{
    "weight": 4,
    "questions": [
      "Is the lion positioned behind the cow, indicating it is following? (yes/no)",
      "Is the cow positioned in front of the lion, indicating it is being followed? (yes/no)"
    ],
    "statements": [
      "The lion is positioned behind the cow, indicating it is following",
      "The cow is positioned in front of the lion, indicating it is being followed"
    ]
  }},
  "pose": {{
    "weight": 2,
    "questions": [
      "Is the lion depicted in a walking state? (yes/no)",
      "Is the cow depicted in a walking state? (yes/no)"
    ],
    "statements": [
      "The lion is depicted in a walking state",
      "The cow is depicted in a walking state"
    ]
  }},
  "orientation": {{
    "weight": 3,
    "questions": [
      "Is the lion oriented toward the cow, suggesting it is following? (yes/no)",
      "Is the cow oriented away from the lion, suggesting it is being followed? (yes/no)"
    ],
    "statements": [
      "The lion is oriented toward the cow, suggesting it is following",
      "The cow is oriented away from the lion, suggesting it is being followed"
    ]
  }},
  "interaction": {{
    "weight": 3,
    "questions": [
      "Is the lion's focus or gaze directed towards the cow? (yes/no)"
    ],
    "statements": [
      "The lion's focus or gaze is directed towards the cow"
    ]
  }}
}}


Example 4:
Input: A photo of one person carrying one fireman
{{
  "detection": {{
    "weight": 5,
    "questions": [
      "Is there a person visible in the image? (yes/no)",
      "Is there a fireman visible in the image? (yes/no)"
    ],
    "statements": [
      "There is a person in the image",
      "There is a fireman in the image"
    ]
  }},
  "count": {{
    "weight": 3,
    "questions": [
      "Is there exactly one person in the image? (yes/no)",
      "Is there exactly one fireman in the image? (yes/no)"
    ],
    "statements": [
      "There is exactly one person in the image",
      "There is exactly one fireman in the image"
    ]
  }},
  "spatial": {{
    "weight": 5,
    "questions": [
      "Is the person positioned in a manner that suggests they are carrying? (yes/no)",
      "Is the fireman positioned above the ground in location that could be considered being carried? (yes/no)"
    ],
    "statements": [
      "The person is positioned in a manner that suggests they are carrying",
      "The fireman is positioned above the ground in location that could be considered being carried."
    ]
  }},
    "pose": {{
    "weight": 3,
    "questions": [
      "Is the person's body in a carrying position, such as arms under the fireman or similar support posture? (yes/no)",
      "Is the fireman in a position that suggests being carried, such as relaxed or non-standing posture? (yes/no)"
    ],
    "statements": [
      "The person's body is in a carrying position, such as arms under the fireman or similar support posture",
      "The fireman is in a position that suggests being carried, such as relaxed or non-standing posture"
    ]
  }},
  "orientation": {{
    "weight": 1,
    "questions": [
      "Is the orientation between the person and fireman appropriate for carrying (e.g., carrier facing forward while carried person may be sideways, down, or facing differently)? (yes/no)"
    ],
    "statements": [
      "The orientation between the person and fireman is appropriate for carrying (e.g., carrier facing forward while carried person may be sideways, down, or facing differently)"
    ]
  }},
  "interaction": {{
    "weight": 3,
    "questions": [
      "Is the person's focus and effort directed towards supporting the fireman? (yes/no)",
      "Does the fireman appear to be relying on the person for support? (yes/no)"
    ],
    "statements": [
      "The person's focus and effort is directed towards supporting the fireman",
      "The fireman appears to be relying on the person for support"
    ]
  }}
}}
Example 5:
    A photo of one boy throwing a ball for one puppy
{{
  "detection": {{
    "weight": 5,
    "questions": [
      "Is there a boy visible in the image? (yes/no)",
      "Is there a puppy visible in the image? (yes/no)",
      "is there a ball visible in the image? ( yes/no)"
    ],
    "statements": [
      "There is a boy in the image",
      "There is a puppy in the image",
      "There is a ball in the image"
    ]
  }},
  "count": {{
    "weight": 3,
    "questions": [
      "Is there exactly one boy in the image? (yes/no)",
      "Is there exactly one puppy in the image? (yes/no)",
      "Is there exactly one ball in the image? (yes/no)"
    ],
    "statements": [
      "There is exactly one boy in the image",
      "There is exactly one puppy in the image",
      "There is exactly one ball in the image"
    ]
  }},
    "spatial": {{
    "weight": 4,
    "questions": [
      "Is the boy positioned in such a way that suggests he is throwing the ball? (yes/no)",
      "Is the puppy positioned such that it appears to be the receiver of the ball or tries to catch it? (yes/no)"
    ],
    "statements": [
      "The boy is positioned in such a way that suggests he is throwing the ball",
      "The puppy is positioned such that it appears to be the receiver of the ball or tries to catch it"
    ]
  }},
  "pose": {{
    "weight": 4,
    "questions": [
      "Is the boy's arm extended or in a throwing motion? (yes/no)",
      "Does the puppy exhibit an anticipatory or receiving pose, such as being mid-jump or chasing the ball? (yes/no)"
    ],
    "statements": [
      "The boy's arm is extended or in a throwing motion",
      "The puppy exhibits an anticipatory or receiving pose, such as being mid-jump or chasing the ball"
    ]
  }},
  "orientation": {{
    "weight": 2,
    "questions": [
      "Is the boy facing towards the puppy? (yes/no)",
      "Is the puppy facing towards the ball? (yes/no)"
    ],
    "statements": [
      "The boy is facing towards the puppy",
      "The puppy is facing towards the ball"
    ]
  }},
  "interaction": {{
    "weight": 1,
    "questions": [
      "Is there a visible ball being thrown by the boy in the direction of the puppy or puppy's movement? (yes/no)",
      "Is there a visible ball being thrown by the boy in the direction of the puppy or puppy's movement? (yes/no)",
      "Is the puppy's gaze or attention directed towards the incoming ball? (yes/no)"
    ],
    "statements": [
      "There is a visible ball being thrown by the boy in the direction of the puppy or puppy's movement",
      "There is a visible ball being thrown by the boy in the direction of the puppy or puppy's movement",
      "The puppy's gaze or attention is directed towards the incoming ball"
    ]
  }}
}}


Example 6:
Input: A photo of one baby feeding one woman

{{
  "detection": {{
    "weight": 5,
    "questions": [
      "Is there a baby visible in the image? (yes/no)",
      "Is there a woman visible in the image? (yes/no)"
    ],
    "statements": [
      "There is a baby in the image",
      "There is a woman in the image"
    ]
  }},
  "count": {{
    "weight": 3,
    "questions": [
      "Is there exactly one baby in the image? (yes/no)",
      "Is there exactly one woman in the image? (yes/no)"
    ],
    "statements": [
      "There is exactly one baby in the image",
      "There is exactly one woman in the image"
    ]
  }},
  "spatial": {{
    "weight": 4,
    "questions": [
      "Is the baby positioned close to the woman, suggesting interaction? (yes/no)",
      "Is the woman positioned in a way that she could receive food from the baby? (yes/no)"
    ],
    "statements": [
      "The baby is positioned close to the woman, suggesting interaction",
      "The woman is positioned in a way that she could receive food from the baby"
    ]
  }},
  "pose": {{
    "weight": 5,
    "questions": [
      "Is the baby depicted in a feeding pose, such as extending an arm towards the woman with food? (yes/no)",
      "Is the woman's posture receptive, such as having her mouth open or hands positioned to accept the food? (yes/no)"
    ],
    "statements": [
      "The baby is depicted in a feeding pose, such as extending an arm towards the woman with food",
      "The woman's posture is receptive, such as having her mouth open or hands positioned to accept the food"
    ]
  }},
  "orientation": {{
    "weight": 3,
    "questions": [
      "Is the baby's body or face oriented toward the woman? (yes/no)",
      "Is the woman's face oriented towards the baby? (yes/no)"
    ],
    "statements": [
      "The baby's body or face is oriented toward the woman",
      "The woman's face is oriented towards the baby"
    ]
  }},
  "interaction": {{
    "weight": 5,
    "questions": [
      "Does the baby appear to be engaging with the woman by feeding her? (yes/no)",
      "Does the woman appear to be interacting with the baby by accepting the food? (yes/no)"
    ],
    "statements": [
      "The baby appears to be engaging with the woman by feeding her",
      "The woman appears to be interacting with the baby by accepting the food"
    ]
  }}
}}


Example 7:
Input: A photo of one zoo trainer lifting one monkey
{{
  "detection": {{
    "weight": 5,
    "questions": [
      "Is there a zoo trainer visible in the image? (yes/no)",
      "Is there a monkey visible in the image? (yes/no)"
    ],
    "statements": [
      "There is a zoo trainer in the image",
      "There is a monkey in the image"
    ]
  }},
  "count": {{
    "weight": 3,
    "questions": [
      "Is there exactly one zoo trainer in the image? (yes/no)",
      "Is there exactly one monkey in the image? (yes/no)"
    ],
    "statements": [
      "There is exactly one zoo trainer in the image",
      "There is exactly one monkey in the image"
    ]
  }},
  "spatial": {{
    "weight": 5,
    "questions": [
      "Is the monkey positioned above the ground level, suggesting it is being lifted? (yes/no)",
      "Is the zoo trainer positioned directly beneath the monkey, suggesting they are lifting the monkey? (yes/no)"
    ],
    "statements": [
      "The monkey is positioned above the ground level, suggesting it is being lifted",
      "The zoo trainer is positioned directly beneath the monkey, suggesting they are lifting the monkey"
    ]
  }},
  "pose": {{
    "weight": 4,
    "questions": [
      "Is the zoo trainer's body in a lifting position, such as arms raised upward? (yes/no)",
      "Is the monkey in a position that suggests it is being lifted, such as limbs hanging down? (yes/no)"
    ],
    "statements": [
      "The zoo trainer's body is in a lifting position, such as arms raised upward",
      "The monkey is in a position that suggests it is being lifted, such as limbs hanging down"
    ]
  }},
  "orientation": {{
    "weight": 2,
    "questions": [
      "Is the zoo trainer's front side facing the monkey? (yes/no)"
    ],
    "statements": [
      "The zoo trainer's front side is facing the monkey"
    ]
  }},
  "interaction": {{
    "weight": 5,
    "questions": [
      "Is the zoo keeper holding the monkey? (yes/no)",
      "Is the monkey being held by the zoo keeper? (yes/no)"
    ],
    "statements": [
      "The zoo keeper is holding the monkey",
      "The monkey is being held by the zoo keeper"
    ]
  }}
}}

Example 8:

Input: A photo of one man pulling one dog

{{
  "detection": {{
    "weight": 5,
    "questions": [
      "Is there a man visible in the image? (yes/no)",
      "Is there a dog visible in the image? (yes/no)"
    ],
    "statements": [
      "There is a man in the image",
      "There is a dog in the image"
    ]
  }},
  "count": {{
    "weight": 3,
    "questions": [
      "Is there exactly one man in the image? (yes/no)",
      "Is there exactly one dog in the image? (yes/no)"
    ],
    "statements": [
      "There is exactly one man in the image",
      "There is exactly one dog in the image"
    ]
  }},
  "spatial": {{
    "weight": 3,
    "questions": [
      "Is there a clear distance/space between the man and the dog? (yes/no)",
      "Are the man and dog positioned at opposite ends of a leash/rope? (yes/no)"
    ],
    "statements": [
      "There is a clear distance/space between the man and the dog",
      "The man and dog are positioned at opposite ends of a leash/rope"
    ]
  }},
  "pose": {{
    "weight": 4,
    "questions": [
      "Is the man in a pulling stance (e.g., leaning back, arms extended, braced posture)? (yes/no)",
      "Is the dog in a pulled position ? (e.g., leaning, standing, or straining posture)? (yes/no)"
    ],
    "statements": [
      "The man is in a pulling stance (e.g., leaning back, arms extended, braced posture)",
      "The dog is in a pulled position (e.g., leaning, standing, or straining posture)"
    ]
  }},
  "orientation": {{
    "weight": 4,
    "questions": [
      "Is the man's body oriented in the direction of the pulling action (i.e. away from the dog)? (yes/no)",
      "Is the dog's body oriented in the direction it's being pulled (i.e. towards the man)? (yes/no)"
    ],
    "statements": [
      "The man's body is oriented in the direction of the pulling action",
      "The dog's body is oriented in the direction it's being pulled"
    ]
  }},
  "interaction": {{
    "weight": 2,
    "questions": [
      "Does the man show active pulling behavior (e.g., tensed muscles, strained expression, firm grip)? (yes/no)",
      "Does the dog show active pulled behavior (e.g., straining forward, engaged movement, tension in body)? (yes/no)"
    ],
    "statements": [
      "The man shows active pulling behavior (e.g., tensed muscles, strained expression, firm grip)",
      "The dog shows active pulled behavior (e.g., straining, engaged movement, tension in body)"
    ]
  }}
}}


Example 9:
input: A photo of one mother kissing one baby
{{
  "detection": {{
    "weight": 5,
    "questions": [
      "Is there a mother visible in the image? (yes/no)",
      "Is there a baby visible in the image? (yes/no)"
    ],
    "statements": [
      "There is a mother in the image",
      "There is a baby in the image"
    ]
  }},
  "count": {{
    "weight": 3,
    "questions": [
      "Is there exactly one mother in the image? (yes/no)",
      "Is there exactly one baby in the image? (yes/no)"
    ],
    "statements": [
      "There is exactly one mother in the image",
      "There is exactly one baby in the image"
    ]
  }},
  "spatial": {{
    "weight": 3,
    "questions": [
      "Are the mother and baby in close physical proximity? (yes/no)",
      "Is the mother's face close to the baby's face/head? (yes/no)"
    ],
    "statements": [
      "The mother and baby are in close physical proximity",
      "The mother's face is close to the baby's face/head"
    ]
  }},
  "pose": {{
    "weight": 4,
    "questions": [
      "Is the mother's head/face positioned for kissing (e.g., leaning in, lips pursed)? (yes/no)",
      "Is the baby positioned to receive the kiss (e.g., face accessible, within reach)? (yes/no)"
    ],
    "statements": [
      "The mother's head/face is positioned for kissing",
      "The baby is positioned to receive the kiss"
    ]
  }},
  "orientation": {{
    "weight": 4,
    "questions": [
      "Is the mother's face oriented toward the baby? (yes/no)",
      "Is the baby's face/head oriented in a way that allows for the kiss? (yes/no)"
    ],
    "statements": [
      "The mother's face is oriented toward the baby",
      "The baby's face/head is oriented in a way that allows for the kiss"
    ]
  }},
  "interaction": {{
    "weight": 4,
    "questions": [
      "Does the mother show affectionate kissing behavior (e.g., gentle expression, tender approach)? (yes/no)",
      "Is there visible contact between the mother's lips and the baby? (yes/no)"
    ],
    "statements": [
      "The mother shows affectionate kissing behavior",
      "There is visible contact between the mother's lips and the baby"
    ]
  }}
}}

Example 10:

Input: A photo of one grandpa holding one teddy bear

{{
  "detection": {{
    "weight": 5,
    "questions": [
      "Is there a grandpa visible in the image? (yes/no)",
      "Is there a teddy bear visible in the image? (yes/no)"
    ],
    "statements": [
      "There is a grandpa in the image",
      "There is a teddy bear in the image"
    ]
  }},
  "count": {{
    "weight": 3,
    "questions": [
      "Is there exactly one grandpa in the image? (yes/no)",
      "Is there exactly one teddy bear in the image? (yes/no)"
    ],
    "statements": [
      "There is exactly one grandpa in the image",
      "There is exactly one teddy bear in the image"
    ]
  }},
  "spatial": {{
    "weight": 4,
    "questions": [
      "Is the teddy bear positioned in the arms of the grandpa? (yes/no)"
    ],
    "statements": [
      "The teddy bear is positioned in the arms of the grandpa"
    ]
  }},
  "pose": {{
    "weight": 2,
    "questions": [
      "Is the grandpa depicted in a standing or seated position while holding the teddy bear? (yes/no)"
    ],
    "statements": [
      "The grandpa is depicted in a standing or seated position while holding the teddy bear"
    ]
  }},
  "orientation": {{
    "weight": 2,
    "questions": [
      "Is the grandpa facing or oriented toward the teddy bear, indicating he is actively holding it? (yes/no)"
    ],
    "statements": [
    "The grandpa is facing or oriented toward the teddy bear, indicating he is actively holding it"
    ]
  }},
  "interaction": {{
    "weight": 4,
    "questions": [
      "Does the grandpa show active holding behavior (e.g., secure grip, cradling arms, engaged handling of the teddy bear)? (yes/no)"
    ],
    "statements": [
      "The grandpa shows active holding behavior with the teddy bear"
    ]
  }}
}}


Now, generate a similar set of weighted yes/no questions for the following relationship, ensuring each question is atomic and directly observable in an image.
Input: {prompt}
"""

    return {
        "system_prompt": "You are an expert at creating structured evaluation questions to verify if an image matches a given prompt. You will generate atomic yes/no questions organized by categories, with each category having an importance weight.",
        "user_prompt": template,
    }
