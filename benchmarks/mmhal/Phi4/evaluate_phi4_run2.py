import json
from collections import defaultdict

with open(r'c:\Users\hiep0\Downloads\phi4_mmhal_responses (1).json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Load image dimensions
with open('image_dimensions.json', 'r', encoding='utf-8') as f:
    img_dims = json.load(f)

# Evaluate each sample of phi4_mmhal_responses (1).json
# MMHal-Bench official rubric:
# 6: very informative, good analysis/reasoning, no hallucination
# 5: very informative, no hallucination
# 4: somewhat informative, no hallucination
# 3: not informative, no hallucination (or evasive/hedging without false claims)
# 2: very informative, with hallucination
# 1: somewhat informative, with hallucination
# 0: not informative, with hallucination

sample_evals = {
    0: {
        "score": 5,
        "explanation": "The model correctly identifies that the fire hydrant cap in the picture is yellow, matching ground truth exactly.",
        "hallucination": False,
        "error_type": "None"
    },
    1: {
        "score": 5,
        "explanation": "The model correctly rejects the adversarial premise, answering 'No one is sitting on the bench', exactly matching ground truth.",
        "hallucination": False,
        "error_type": "None"
    },
    2: {
        "score": 4,
        "explanation": "The model correctly answers 'Left', identifying the left wooden stool as having the vase with a red flower.",
        "hallucination": False,
        "error_type": "None"
    },
    3: {
        "score": 5,
        "explanation": "The model correctly counts four traffic lights in the image, matching ground truth.",
        "hallucination": False,
        "error_type": "None"
    },
    4: {
        "score": 6,
        "explanation": "The model accurately describes the colors and text details of traffic signs from top to bottom (top white with black text, middle blue with white text, bottom white with black text).",
        "hallucination": False,
        "error_type": "None"
    },
    5: {
        "score": 4,
        "explanation": "The model notes that weather cannot be determined with certainty, and infers it might not be raining from absence of rain gear. No false claims.",
        "hallucination": False,
        "error_type": "None"
    },
    6: {
        "score": 5,
        "explanation": "The model provides an accurate holistic description of the rainy city street, wet pavement reflection, cars, street lamps, traffic signs, sidewalk and buildings, matching the scene without hallucinating absent objects.",
        "hallucination": False,
        "error_type": "None"
    },
    7: {
        "score": 5,
        "explanation": "The model correctly extracts the parking meter rate as '$4.00', matching ground truth.",
        "hallucination": False,
        "error_type": "None"
    },
    8: {
        "score": 1,
        "explanation": "The question asks for colors from right to left (black and blue). The model answers 'blue and black', inverting the directional order. Spatial/directional attribute hallucination.",
        "hallucination": True,
        "error_type": "Attribute / Spatial Order Hallucination"
    },
    9: {
        "score": 3,
        "explanation": "Adversarial question asking gender of motorcycle rider when no one is riding. The model hedges that image does not provide enough information to determine gender. Does not fabricate gender.",
        "hallucination": False,
        "error_type": "Weak Adversarial Defense (Hedging)"
    },
    10: {
        "score": 5,
        "explanation": "The model correctly identifies that the left truck has its door open.",
        "hallucination": False,
        "error_type": "None"
    },
    11: {
        "score": 1,
        "explanation": "Ground truth states there are 4 bicycles (two front, one middle, one distant). Model says 3 bicycles, under-counting due to small scale of the distant bicycle.",
        "hallucination": True,
        "error_type": "Counting Hallucination (Under-counting)"
    },
    12: {
        "score": 6,
        "explanation": "The model correctly positions the yellow boat in front of the white yacht, adding accurate grounding detail about the man on the boat.",
        "hallucination": False,
        "error_type": "None"
    },
    13: {
        "score": 5,
        "explanation": "The model identifies the weather as cloudy, matching ground truth (sunny or cloudy).",
        "hallucination": False,
        "error_type": "None"
    },
    14: {
        "score": 6,
        "explanation": "Highly detailed and accurate holistic description of the yellow school bus ('STOP', 'ABACO CARRIERS', 'COLLINS'), cyclist with helmet/backpack, and 'TATTOOS' and 'REPAIRS' signs on background building.",
        "hallucination": False,
        "error_type": "None"
    },
    15: {
        "score": 2,
        "explanation": "GT states Virgin Airlines. Model responds 'Virgin United Airlines', conflating Virgin and United Airlines. Entity hallucination.",
        "hallucination": True,
        "error_type": "Knowledge / Entity Hallucination"
    },
    16: {
        "score": 5,
        "explanation": "The model correctly identifies shirt colors from left to right as green, blue, and red.",
        "hallucination": False,
        "error_type": "None"
    },
    17: {
        "score": 1,
        "explanation": "Adversarial question asking glasses color when the man is not wearing glasses. The model hallucinates 'The man in the image is wearing black glasses.'",
        "hallucination": True,
        "error_type": "Adversarial Susceptibility / False Object Hallucination"
    },
    18: {
        "score": 4,
        "explanation": "The model correctly answers 'woman' for who is wearing trousers.",
        "hallucination": False,
        "error_type": "None"
    },
    19: {
        "score": 1,
        "explanation": "GT states 4 people. Model answers 3 people, failing to detect one partially occluded person (under-counting).",
        "hallucination": True,
        "error_type": "Counting Hallucination (Under-counting)"
    },
    20: {
        "score": 1,
        "explanation": "GT states girl stands on right. Model asserts girl is positioned in center. Spatial relation hallucination.",
        "hallucination": True,
        "error_type": "Spatial Relation Hallucination"
    },
    21: {
        "score": 5,
        "explanation": "The model correctly identifies that the man is outdoors, sitting on a boat.",
        "hallucination": False,
        "error_type": "None"
    },
    22: {
        "score": 5,
        "explanation": "The model describes Hawkin's Bazaar storefront, green signboard, window display, fallen leaves, and signs accurately.",
        "hallucination": False,
        "error_type": "None"
    },
    23: {
        "score": 5,
        "explanation": "The model reads runner number '1097' correctly.",
        "hallucination": False,
        "error_type": "None"
    },
    24: {
        "score": 3,
        "explanation": "Question asks for colors of dogs from right to left (brown and black). Model only answers 'brown', omitting the second dog. Under-informative but not fabricating.",
        "hallucination": False,
        "error_type": "Incomplete Response"
    },
    25: {
        "score": 6,
        "explanation": "The model robustly rejects adversarial premise of vicuna in sand, and accurately identifies the true subject as a toucan with colorful beak on a branch.",
        "hallucination": False,
        "error_type": "None"
    },
    26: {
        "score": 4,
        "explanation": "The model correctly answers 'right' for the cat opening its mouth.",
        "hallucination": False,
        "error_type": "None"
    },
    27: {
        "score": 1,
        "explanation": "GT states 6 zebras. Model counts 4 zebras, under-counting due to overlapping patterns.",
        "hallucination": True,
        "error_type": "Counting Hallucination (Under-counting)"
    },
    28: {
        "score": 4,
        "explanation": "The model describes foreground sheep as 'the one with a white face and a black body', accurately distinguishing it.",
        "hallucination": False,
        "error_type": "None"
    },
    29: {
        "score": 1,
        "explanation": "Question asks for weather (GT: dark night race). Model claims sunny day due to green field and outdoor track, completely hallucinating daylight for a night scene.",
        "hallucination": True,
        "error_type": "Environmental Attribute Hallucination"
    },
    30: {
        "score": 4,
        "explanation": "The model describes a zoo enclosure with three giraffes and three zebras under thatched structures with rocky background and reddish dirt. GT states 3 zebras and 4 giraffes. Very close count (3 vs 4 giraffes) and richly descriptive.",
        "hallucination": False,
        "error_type": "None"
    },
    31: {
        "score": 1,
        "explanation": "GT states 2 rhinos, 1 giraffe, 1 wild horse. Model claims no giraffes and describes 2 rhinos and a zebra (misidentifying horse as zebra, missing giraffe).",
        "hallucination": True,
        "error_type": "Object Misidentification / False Negative"
    },
    32: {
        "score": 5,
        "explanation": "The model correctly identifies the kiwi in front is dried.",
        "hallucination": False,
        "error_type": "None"
    },
    33: {
        "score": 3,
        "explanation": "Adversarial question asking if jam is strawberry (GT: only bread, no jam). Model hedges that detail is insufficient to determine jam type. Fails to reject jam but avoids false claim.",
        "hallucination": False,
        "error_type": "Weak Adversarial Defense (Hedging)"
    },
    34: {
        "score": 5,
        "explanation": "The model compares tomatoes accurately: leftmost red/ripe, middle yellow/red ripening, rightmost mostly yellow unripe. Matches GT perfectly.",
        "hallucination": False,
        "error_type": "None"
    },
    35: {
        "score": 1,
        "explanation": "GT states 5 oysters visible. Model counts 4 oysters, under-counting by one due to low resolution of small shell.",
        "hallucination": True,
        "error_type": "Counting Hallucination (Under-counting)"
    },
    36: {
        "score": 1,
        "explanation": "GT states broccoli bowl is next to meatball bowl. Model claims broccoli bowl is on top of meatball bowl. Spatial relation hallucination.",
        "hallucination": True,
        "error_type": "Spatial Relation Hallucination"
    },
    37: {
        "score": 1,
        "explanation": "Photo taken outdoors on beach grill. Model claims photo is taken at a restaurant. Scene hallucination.",
        "hallucination": True,
        "error_type": "Scene / Environment Hallucination"
    },
    38: {
        "score": 5,
        "explanation": "The model describes two women/girls and a man in aprons preparing food in kitchen with bowls, spoon, knife, table. Well aligned with GT.",
        "hallucination": False,
        "error_type": "None"
    },
    39: {
        "score": 5,
        "explanation": "The model extracts price tag '£5.00' correctly.",
        "hallucination": False,
        "error_type": "None"
    },
    40: {
        "score": 5,
        "explanation": "The model correctly identifies parachute colors as blue, green, and orange.",
        "hallucination": False,
        "error_type": "None"
    },
    41: {
        "score": 6,
        "explanation": "Adversarial question asking to describe person playing tennis. Model clearly rejects premise: no person playing tennis, only tennis racket and ball on court.",
        "hallucination": False,
        "error_type": "None"
    },
    42: {
        "score": 4,
        "explanation": "Model identifies white and red surfboards matching GT colors.",
        "hallucination": False,
        "error_type": "None"
    },
    43: {
        "score": 5,
        "explanation": "The model correctly counts 3 horses.",
        "hallucination": False,
        "error_type": "None"
    },
    44: {
        "score": 5,
        "explanation": "The model correctly identifies ball on left as orange.",
        "hallucination": False,
        "error_type": "None"
    },
    45: {
        "score": 5,
        "explanation": "The model correctly identifies indoor ice hockey rink.",
        "hallucination": False,
        "error_type": "None"
    },
    46: {
        "score": 2,
        "explanation": "The model describes players in white and blue jerseys. However, it hallucinates that 'The field is surrounded by a crowd of spectators, who are watching the match intently', which are not present in this close view.",
        "hallucination": True,
        "error_type": "Background Object / Crowd Hallucination"
    },
    47: {
        "score": 5,
        "explanation": "The model correctly extracts tournament name 'INDIAN WELLS'.",
        "hallucination": False,
        "error_type": "None"
    },
    48: {
        "score": 5,
        "explanation": "The model correctly identifies silver microwave.",
        "hallucination": False,
        "error_type": "None"
    },
    49: {
        "score": 5,
        "explanation": "Adversarial question: how many people eating in kitchen. Model answers '0'. Correct.",
        "hallucination": False,
        "error_type": "None"
    },
    50: {
        "score": 5,
        "explanation": "The model describes metal forks, spoons, knives with plastic handles designed like building blocks (Lego-like). Matches GT.",
        "hallucination": False,
        "error_type": "None"
    },
    51: {
        "score": 4,
        "explanation": "The model correctly counts 2 forks.",
        "hallucination": False,
        "error_type": "None"
    },
    52: {
        "score": 3,
        "explanation": "Question asks for objects right to left (lid, spoon, cup). Model only says 'spoon'. Incomplete response.",
        "hallucination": False,
        "error_type": "Incomplete Response"
    },
    53: {
        "score": 5,
        "explanation": "The model correctly identifies daytime based on natural window sunlight.",
        "hallucination": False,
        "error_type": "None"
    },
    54: {
        "score": 2,
        "explanation": "GT describes wooden table with 4 chairs, 2 coasters, 2 cups. Model asserts green chair with green cushion, glass of water, white plate, radiator. Fabricated objects.",
        "hallucination": True,
        "error_type": "Object Hallucination"
    },
    55: {
        "score": 3,
        "explanation": "Question asks which buttons control stove above. Model gives generic manual disclaimer. Uninformative.",
        "hallucination": False,
        "error_type": "Evasive / Uninformative"
    },
    56: {
        "score": 5,
        "explanation": "The model correctly identifies umbrella colors (purple, yellow, green, orange).",
        "hallucination": False,
        "error_type": "None"
    },
    57: {
        "score": 1,
        "explanation": "Adversarial question asking to describe hands wearing gloves when no hands exist. Model states 'The hands are not visible as they are covered by the gloves', presupposing hands inside.",
        "hallucination": True,
        "error_type": "Adversarial Presupposition Bias"
    },
    58: {
        "score": 5,
        "explanation": "Detailed comparison of interior compartment view and exterior shape/straps/logo view of backpack.",
        "hallucination": False,
        "error_type": "None"
    },
    59: {
        "score": 1,
        "explanation": "GT states 5 black hats. Model claims 6 black hats (over-counting error).",
        "hallucination": True,
        "error_type": "Counting Hallucination (Over-counting)"
    },
    60: {
        "score": 5,
        "explanation": "The model states 'The two watches are placed side by side, with the black watch on the left and the red watch on the right.' Accurate spatial description.",
        "hallucination": False,
        "error_type": "None"
    },
    61: {
        "score": 5,
        "explanation": "The model correctly determines indoor setting based on carpet.",
        "hallucination": False,
        "error_type": "None"
    },
    62: {
        "score": 5,
        "explanation": "Accurate description of bride's hand with diamond ring on groom's hand with wedding band, white dress, black suit, black and white image.",
        "hallucination": False,
        "error_type": "None"
    },
    63: {
        "score": 4,
        "explanation": "The model describes blurred reflection of surroundings and people in sunglasses. Consistent with GT.",
        "hallucination": False,
        "error_type": "None"
    },
    64: {
        "score": 5,
        "explanation": "The model correctly answers laptop is silver.",
        "hallucination": False,
        "error_type": "None"
    },
    65: {
        "score": 5,
        "explanation": "Adversarial question: what is playing on monitor (no monitor). Model answers 'nothing'. Correct rejection.",
        "hallucination": False,
        "error_type": "None"
    },
    66: {
        "score": 4,
        "explanation": "The model correctly identifies device on left as larger.",
        "hallucination": False,
        "error_type": "None"
    },
    67: {
        "score": 1,
        "explanation": "GT states 1 phone. Model claims 2 phones (over-counting).",
        "hallucination": True,
        "error_type": "Counting Hallucination (Over-counting)"
    },
    68: {
        "score": 1,
        "explanation": "Question asks if mouse connected to computer (GT: No). Model asserts mouse is connected to computer via USB cable. Relational hallucination.",
        "hallucination": True,
        "error_type": "Relational Hallucination"
    },
    69: {
        "score": 4,
        "explanation": "The model correctly answers mountain setting.",
        "hallucination": False,
        "error_type": "None"
    },
    70: {
        "score": 5,
        "explanation": "The model describes flat-screen TV showing man in suit/tie, notebook with pen below TV. Accurate grounding.",
        "hallucination": False,
        "error_type": "None"
    },
    71: {
        "score": 5,
        "explanation": "The model correctly identifies the laptop as MacBook based on keyboard layout and visible logo, matching GT.",
        "hallucination": False,
        "error_type": "None"
    },
    72: {
        "score": 1,
        "explanation": "GT states pillows are white/grey, yellow, white/grey. Model claims light blue, cream with floral embroidery, and light blue with stripes. Color/attribute hallucination.",
        "hallucination": True,
        "error_type": "Attribute / Color Hallucination"
    },
    73: {
        "score": 0,
        "explanation": "Adversarial question asking chair color when no chair exists. Model hallucinates 'The color of the chair in the image is beige.'",
        "hallucination": True,
        "error_type": "Adversarial Susceptibility / False Object Hallucination"
    },
    74: {
        "score": 5,
        "explanation": "The model correctly answers pillow on right is closer to camera.",
        "hallucination": False,
        "error_type": "None"
    },
    75: {
        "score": 5,
        "explanation": "The model correctly counts 2 chairs.",
        "hallucination": False,
        "error_type": "None"
    },
    76: {
        "score": 5,
        "explanation": "The model correctly identifies pillow in middle is in front.",
        "hallucination": False,
        "error_type": "None"
    },
    77: {
        "score": 1,
        "explanation": "Photo taken at night. Model claims sunny weather with natural sunlight through window. Environmental hallucination.",
        "hallucination": True,
        "error_type": "Environmental Attribute Hallucination"
    },
    78: {
        "score": 2,
        "explanation": "GT states two deck beds. Model claims three bunk beds. Quantity hallucination.",
        "hallucination": True,
        "error_type": "Counting / Object Quantity Hallucination"
    },
    79: {
        "score": 4,
        "explanation": "The model correctly answers 'no' to whether toilet paper was used.",
        "hallucination": False,
        "error_type": "None"
    },
    80: {
        "score": 4,
        "explanation": "Model correctly identifies red book color (GT: red and white).",
        "hallucination": False,
        "error_type": "None"
    },
    81: {
        "score": 5,
        "explanation": "Adversarial question: what in vase (empty). Model answers 'nothing'. Correct.",
        "hallucination": False,
        "error_type": "None"
    },
    82: {
        "score": 4,
        "explanation": "The model correctly answers vase on right looks larger.",
        "hallucination": False,
        "error_type": "None"
    },
    83: {
        "score": 1,
        "explanation": "GT states 6 teddy bears on stairs. Model counts 5 teddy bears (under-counting by one).",
        "hallucination": True,
        "error_type": "Counting Hallucination (Under-counting)"
    },
    84: {
        "score": 5,
        "explanation": "The model correctly states blue flowers on top, white flowers on bottom.",
        "hallucination": False,
        "error_type": "None"
    },
    85: {
        "score": 4,
        "explanation": "The model reasons weather cannot be determined because it is an indoor scene. No false claim.",
        "hallucination": False,
        "error_type": "None"
    },
    86: {
        "score": 5,
        "explanation": "The model describes baby sleeping on white mattress inside crib with 3 hanging stuffed animals above. Matches GT.",
        "hallucination": False,
        "error_type": "None"
    },
    87: {
        "score": 4,
        "explanation": "Model reads 'ROVERBIAL PHILOSOPHY', missing initial capital 'P' due to decorative font. Minor OCR truncation.",
        "hallucination": False,
        "error_type": "Minor OCR Truncation"
    },
    88: {
        "score": 5,
        "explanation": "The model correctly identifies silver pot.",
        "hallucination": False,
        "error_type": "None"
    },
    89: {
        "score": 4,
        "explanation": "Adversarial: what washed in sink (nothing). Model responds no one being washed in sink. Effective rejection.",
        "hallucination": False,
        "error_type": "None"
    },
    90: {
        "score": 4,
        "explanation": "The model correctly identifies right washing machine is taller.",
        "hallucination": False,
        "error_type": "None"
    },
    91: {
        "score": 4,
        "explanation": "The model correctly counts 1 cooked chicken.",
        "hallucination": False,
        "error_type": "None"
    },
    92: {
        "score": 5,
        "explanation": "The model correctly identifies screwdriver placed on tissue roll.",
        "hallucination": False,
        "error_type": "None"
    },
    93: {
        "score": 4,
        "explanation": "Model answers 'afternoon' for daytime photo.",
        "hallucination": False,
        "error_type": "None"
    },
    94: {
        "score": 5,
        "explanation": "The model describes stainless steel steamer with glass lid containing white steamed buns (baozi), metal tray, spatula, condensation from steam. Highly accurate.",
        "hallucination": False,
        "error_type": "None"
    },
    95: {
        "score": 5,
        "explanation": "The model correctly identifies reflection of washing machine in mirror.",
        "hallucination": False,
        "error_type": "None"
    }
}

# Metrics calculation
results = []
score_list = []
hallucination_list = []
by_type = defaultdict(list)
by_topic = defaultdict(list)
error_type_counts = defaultdict(int)

for i, d in enumerate(data):
    e = sample_evals[i]
    score = e["score"]
    hal = 1 if score < 3 else 0
    score_list.append(score)
    hallucination_list.append(hal)
    by_type[d["question_type"]].append((score, hal))
    by_topic[d["question_topic"]].append((score, hal))
    if hal == 1:
        error_type_counts[e["error_type"]] += 1
    
    img_name = d['image_src'].split('/')[-1]
    dim = img_dims.get(img_name, {})

    item_res = {
        "index": i,
        "question_type": d["question_type"],
        "question_topic": d["question_topic"],
        "image_name": img_name,
        "original_width": dim.get("width"),
        "original_height": dim.get("height"),
        "was_resized_to_1024": max(dim.get("width", 0), dim.get("height", 0)) > 1024,
        "question": d["question"],
        "gt_answer": d["gt_answer"],
        "model_answer": d["model_answer"],
        "score": score,
        "hallucination": hal,
        "error_type": e["error_type"],
        "explanation": e["explanation"]
    }
    results.append(item_res)

with open('phi4_mmhal_evaluated_run2.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Total samples: {len(results)}")
print(f"Average Score (0-6 scale): {sum(score_list)/len(score_list):.2f}")
print(f"Average Score (0-5 normalized scale): {sum(score_list)/len(score_list)/6.0*5.0:.2f}")
print(f"Hallucination Rate: {sum(hallucination_list)/len(hallucination_list)*100:.2f}%")
print(f"Total Hallucinated Samples: {sum(hallucination_list)} / {len(results)}")

print("\n--- Breakdown by Question Type ---")
for q_type, val in by_type.items():
    scs = [x[0] for x in val]
    hals = [x[1] for x in val]
    print(f"{q_type:15s} | Avg Score: {sum(scs)/len(scs):.2f} | Hal Rate: {sum(hals)/len(hals)*100:.1f}% ({sum(hals)}/{len(val)})")

print("\n--- Breakdown by Question Topic ---")
for topic, val in by_topic.items():
    scs = [x[0] for x in val]
    hals = [x[1] for x in val]
    print(f"{topic:15s} | Avg Score: {sum(scs)/len(scs):.2f} | Hal Rate: {sum(hals)/len(hals)*100:.1f}% ({sum(hals)}/{len(val)})")

print("\n--- Error Type Distribution ---")
for err, cnt in sorted(error_type_counts.items(), key=lambda x: -x[1]):
    print(f"{err:45s}: {cnt}")
