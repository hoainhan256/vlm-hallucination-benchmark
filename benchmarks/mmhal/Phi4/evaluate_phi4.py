import json
from collections import defaultdict

with open(r'c:\Users\hiep0\Downloads\phi4_mmhal_responses.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Evaluation definition following MMHal-Bench official rubric:
# Rating 6: very informative with good analysis or reasoning, no hallucination
# Rating 5: very informative, no hallucination
# Rating 4: somewhat informative, no hallucination
# Rating 3: not informative, no hallucination (or evasive/uncertain without false claims)
# Rating 2: very informative, with hallucination
# Rating 1: somewhat informative, with hallucination
# Rating 0: not informative, with hallucination (completely incorrect / blatant hallucination)
# Hallucination threshold: Score < 3 => Hallucination (1), Score >= 3 => No Hallucination (0)

sample_evals = {
    0: {
        "score": 5,
        "explanation": "The model correctly identifies that the fire hydrant cap is yellow, exactly matching the ground truth. The response is concise, accurate, and completely grounded in the visual evidence without any hallucination.",
        "hallucination": False,
        "error_type": "None"
    },
    1: {
        "score": 5,
        "explanation": "The model successfully detects the adversarial question premise and correctly asserts that no one is sitting on the bench, matching the ground truth.",
        "hallucination": False,
        "error_type": "None"
    },
    2: {
        "score": 4,
        "explanation": "The model directly answers 'Left', which correctly identifies the left wooden stool as having the vase with a red flower. It is somewhat brief but completely accurate and free of hallucinations.",
        "hallucination": False,
        "error_type": "None"
    },
    3: {
        "score": 5,
        "explanation": "The model accurately counts four traffic lights in the image, perfectly matching the ground truth without any hallucination.",
        "hallucination": False,
        "error_type": "None"
    },
    4: {
        "score": 6,
        "explanation": "The model accurately describes the colors of all three traffic signs from top to bottom (top white with black text, middle blue with white text, bottom white with black text). It provides rich, informative detail with high visual precision.",
        "hallucination": False,
        "error_type": "None"
    },
    5: {
        "score": 4,
        "explanation": "The question asks to describe the weather. The ground truth notes nighttime with a dark sky around an illuminated Ferris wheel. The model reasons that weather cannot be determined with certainty but notes the absence of rain gear. It contains no false claims about the image.",
        "hallucination": False,
        "error_type": "None"
    },
    6: {
        "score": 2,
        "explanation": "The model provides a descriptive overview of the rainy city street, trees, wet sidewalk, and cars. However, it hallucinates a 'fire hydrant on the sidewalk' which is not present in the scene/ground truth annotations.",
        "hallucination": True,
        "error_type": "Object Hallucination"
    },
    7: {
        "score": 5,
        "explanation": "The model correctly reads the parking meter rate as '$4.00', matching the ground truth of $4 per hour.",
        "hallucination": False,
        "error_type": "None"
    },
    8: {
        "score": 1,
        "explanation": "The question asks for the color of the two cars from right to left. The ground truth is 'black and blue'. The model responds 'blue and black', inverting the spatial ordering. This is a directional/ordering attribute hallucination.",
        "hallucination": True,
        "error_type": "Attribute / Spatial Order Hallucination"
    },
    9: {
        "score": 3,
        "explanation": "Adversarial question asking for the gender of the motorcycle rider when no one is riding it. The model states that the image does not provide enough information to determine the gender. While it fails to explicitly reject the presence of a rider, it hedges and does not fabricate specific gender details.",
        "hallucination": False,
        "error_type": "Weak Adversarial Defense (Hedging)"
    },
    10: {
        "score": 5,
        "explanation": "The model accurately compares the two trucks and correctly determines that the left truck (fire truck) has its door open.",
        "hallucination": False,
        "error_type": "None"
    },
    11: {
        "score": 1,
        "explanation": "The ground truth identifies four bicycles (two front, one middle, one distant). The model answers 'There are three bicycles', failing to detect the fourth distant bicycle (under-counting). Under MMHal-Bench criteria, false count claims constitute hallucination.",
        "hallucination": True,
        "error_type": "Counting Hallucination (Under-counting)"
    },
    12: {
        "score": 6,
        "explanation": "The model correctly describes that the yellow boat is positioned in front of the white yacht, and adds accurate grounding details about the man standing on the boat. Highly informative and accurate.",
        "hallucination": False,
        "error_type": "None"
    },
    13: {
        "score": 5,
        "explanation": "The model answers that the weather is cloudy, which aligns with the ground truth stating sunny or cloudy.",
        "hallucination": False,
        "error_type": "None"
    },
    14: {
        "score": 5,
        "explanation": "The model provides an accurate holistic description of the cyclist with helmet and the parked yellow school bus on the city street, fully consistent with ground truth.",
        "hallucination": False,
        "error_type": "None"
    },
    15: {
        "score": 2,
        "explanation": "The ground truth is Virgin Airlines (Virgin Atlantic). The model claims 'Virgin United Airlines', blending two distinct airlines (Virgin and United Airlines). This is an entity name conflation/hallucination.",
        "hallucination": True,
        "error_type": "Knowledge / Entity Hallucination"
    },
    16: {
        "score": 5,
        "explanation": "The model correctly identifies the shirt colors of the three men from left to right as green, blue, and red, perfectly matching ground truth.",
        "hallucination": False,
        "error_type": "None"
    },
    17: {
        "score": 1,
        "explanation": "Adversarial question asking what color glasses the man is wearing. The man is not wearing glasses. The model falls for the trap and hallucinates that 'The man in the image is wearing black glasses.'",
        "hallucination": True,
        "error_type": "Adversarial Susceptibility / False Object Hallucination"
    },
    18: {
        "score": 4,
        "explanation": "The model answers 'woman', correctly answering who is wearing trousers/pants in the image.",
        "hallucination": False,
        "error_type": "None"
    },
    19: {
        "score": 1,
        "explanation": "Ground truth states there are 4 people in the image. The model claims there are 3 people, failing to detect one person (under-counting error).",
        "hallucination": True,
        "error_type": "Counting Hallucination (Under-counting)"
    },
    20: {
        "score": 1,
        "explanation": "The girl stands on the right among the three people. The model claims the girl is positioned in the center, which is spatially false.",
        "hallucination": True,
        "error_type": "Spatial Relation Hallucination"
    },
    21: {
        "score": 5,
        "explanation": "The model correctly identifies that the man is outdoors, sitting on a boat, matching ground truth.",
        "hallucination": False,
        "error_type": "None"
    },
    22: {
        "score": 5,
        "explanation": "The model describes Hawkin's Bazaar storefront, green awning, window display, and 'Mid-Season Madness' sale sign accurately.",
        "hallucination": False,
        "error_type": "None"
    },
    23: {
        "score": 5,
        "explanation": "The model accurately reads the runner bib number as '1097', matching ground truth.",
        "hallucination": False,
        "error_type": "None"
    },
    24: {
        "score": 3,
        "explanation": "The question asks for the colors of the dogs from right to left (brown and black). The model only answers 'brown', omitting the second dog. It does not fabricate false colors but is under-informative.",
        "hallucination": False,
        "error_type": "Incomplete Response"
    },
    25: {
        "score": 6,
        "explanation": "Adversarial question asking what a vicuna is looking at. The model completely and robustly rejects the premise, explaining that there is no vicuna or animal in sand, and accurately identifies the true subject as a toucan perched on a branch.",
        "hallucination": False,
        "error_type": "None"
    },
    26: {
        "score": 4,
        "explanation": "The model correctly answers 'right', matching the ground truth that the cat on the right opens its mouth.",
        "hallucination": False,
        "error_type": "None"
    },
    27: {
        "score": 1,
        "explanation": "Ground truth states there are 6 zebras. The model claims there are 4 zebras, under-counting by two.",
        "hallucination": True,
        "error_type": "Counting Hallucination (Under-counting)"
    },
    28: {
        "score": 4,
        "explanation": "Ground truth states the sheep in front is a white sheep. The model describes it as 'the one with a white face and a black body'. It correctly isolates the foreground animal with visual detail.",
        "hallucination": False,
        "error_type": "None"
    },
    29: {
        "score": 1,
        "explanation": "Question asks how the weather was when the picture was captured (nighttime horse race). The model claims the weather was 'rainy, as the horses and jockeys are racing on a wet track'. The track or lighting was misinterpreted as rain.",
        "hallucination": True,
        "error_type": "Environmental Attribute Hallucination"
    },
    30: {
        "score": 2,
        "explanation": "Holistic description of zebras and giraffes in enclosure. Ground truth states 3 zebras and 4 giraffes. The model hallucinates 'two giraffes and two zebras', severely miscounting both groups despite getting the setting right.",
        "hallucination": True,
        "error_type": "Counting / Quantifier Hallucination"
    },
    31: {
        "score": 1,
        "explanation": "Question asks how many giraffes out of four animals. GT states 2 rhinos, 1 giraffe, 1 wild horse. The model erroneously claims there are no giraffes and describes the animals as two rhinos and a zebra.",
        "hallucination": True,
        "error_type": "Object Misidentification / False Negative"
    },
    32: {
        "score": 5,
        "explanation": "The model correctly identifies that the kiwi in front is dried, matching ground truth.",
        "hallucination": False,
        "error_type": "None"
    },
    33: {
        "score": 3,
        "explanation": "Adversarial question asking if the jam on bread is strawberry. GT notes there is only toasted bread and no jam. The model hedges: 'The image does not provide enough detail to determine the exact type of jam.' It fails to reject the jam premise but makes no affirmative claim about strawberry.",
        "hallucination": False,
        "error_type": "Weak Adversarial Defense (Hedging)"
    },
    34: {
        "score": 5,
        "explanation": "The model compares the tomatoes by color from red to yellow and infers stages of ripeness, matching the ground truth comparison.",
        "hallucination": False,
        "error_type": "None"
    },
    35: {
        "score": 1,
        "explanation": "Ground truth identifies 5 oysters visible. The model counts 4 oysters, under-counting by one.",
        "hallucination": True,
        "error_type": "Counting Hallucination (Under-counting)"
    },
    36: {
        "score": 1,
        "explanation": "Question asks if the bowl with broccoli is on top of the bowl with meatballs. GT states they are side by side. The model erroneously answers 'Yes, the bowl with broccoli is on top of the bowl with meatballs.'",
        "hallucination": True,
        "error_type": "Spatial Relation Hallucination"
    },
    37: {
        "score": 1,
        "explanation": "Question asks if photo is taken in a restaurant, kitchen, or anywhere else. GT notes outdoors on a beach grill. The model claims it was taken 'at a restaurant', hallucinating an indoor/commercial dining setting.",
        "hallucination": True,
        "error_type": "Scene / Environment Hallucination"
    },
    38: {
        "score": 5,
        "explanation": "The model accurately describes a man and two young girls in aprons preparing food at a table with kitchen utensils, well-aligned with ground truth.",
        "hallucination": False,
        "error_type": "None"
    },
    39: {
        "score": 5,
        "explanation": "The model correctly reads the price tag '£5.00' for each jerk chicken dumpling, matching ground truth.",
        "hallucination": False,
        "error_type": "None"
    },
    40: {
        "score": 5,
        "explanation": "The model correctly identifies the colors of the parachutes as blue, green, and orange, matching the ground truth colors.",
        "hallucination": False,
        "error_type": "None"
    },
    41: {
        "score": 6,
        "explanation": "Adversarial question asking to describe the person playing tennis. The model correctly and clearly rejects the premise, stating no person is playing tennis and describing the racket and ball on court.",
        "hallucination": False,
        "error_type": "None"
    },
    42: {
        "score": 4,
        "explanation": "The model correctly identifies one surfboard as white and the other as red, matching the colors in GT, though it mentions left/right instead of top/bottom.",
        "hallucination": False,
        "error_type": "None"
    },
    43: {
        "score": 5,
        "explanation": "The model accurately counts 3 horses in the photo, matching ground truth.",
        "hallucination": False,
        "error_type": "None"
    },
    44: {
        "score": 5,
        "explanation": "The model correctly answers that the ball on the left is orange, matching ground truth (red or orange).",
        "hallucination": False,
        "error_type": "None"
    },
    45: {
        "score": 5,
        "explanation": "The model correctly determines that the photo is taken indoors at an ice hockey rink, matching ground truth.",
        "hallucination": False,
        "error_type": "None"
    },
    46: {
        "score": 2,
        "explanation": "The model describes two soccer players (white/red and blue). However, it hallucinates that 'In the background, there are several chairs arranged in rows... mostly white, with a few blue ones', which are not present in the field scene.",
        "hallucination": True,
        "error_type": "Background Object Hallucination"
    },
    47: {
        "score": 5,
        "explanation": "The model correctly extracts the tournament name 'INDIAN WELLS' from the court surface text.",
        "hallucination": False,
        "error_type": "None"
    },
    48: {
        "score": 5,
        "explanation": "The model correctly answers that the microwave is silver, matching ground truth.",
        "hallucination": False,
        "error_type": "None"
    },
    49: {
        "score": 5,
        "explanation": "Adversarial question asking how many people are eating in the kitchen. The model answers '0', correctly identifying that no one is eating.",
        "hallucination": False,
        "error_type": "None"
    },
    50: {
        "score": 5,
        "explanation": "The model compares the utensils, noting they are metal with colorful Lego-like handles (forks, spoons, knives), consistent with ground truth.",
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
        "explanation": "Question asks for objects from right to left (lid, spoon, coffee cup). The model only answers 'spoon', omitting the lid and coffee cup. Incomplete, though not an active fabrication.",
        "hallucination": False,
        "error_type": "Incomplete Response"
    },
    53: {
        "score": 5,
        "explanation": "The model correctly states the photo was taken during the day, citing natural sunlight from the kitchen window.",
        "hallucination": False,
        "error_type": "None"
    },
    54: {
        "score": 2,
        "explanation": "The GT describes a wooden dining table with 4 chairs, 2 coasters, and 2 glass cups. The model asserts there is a 'green chair', 'glass of water and a white plate', and a 'radiator' in background, introducing ungrounded/inaccurate objects.",
        "hallucination": True,
        "error_type": "Object Hallucination"
    },
    55: {
        "score": 3,
        "explanation": "Question asks which buttons to use to control the stove above (buttons on the right). The model gives a generic disclaimer about referring to the user manual without answering which buttons based on visual evidence.",
        "hallucination": False,
        "error_type": "Evasive / Uninformative"
    },
    56: {
        "score": 5,
        "explanation": "The model describes the umbrella as rainbow-colored with purple, yellow, green, and orange, closely matching the GT colors (purple, yellow, teal, orange).",
        "hallucination": False,
        "error_type": "None"
    },
    57: {
        "score": 1,
        "explanation": "Adversarial question asking to describe the hands wearing the gloves when there are no hands in the image (gloves only). The model states 'The hands are not visible as they are covered by the gloves', fabricating the existence of hands inside.",
        "hallucination": True,
        "error_type": "Adversarial Susceptibility / False Object Presupposition"
    },
    58: {
        "score": 5,
        "explanation": "The model compares the interior/compartment view and exterior structure view of the backpack, mentioning the camera and organizing features, accurately reflecting the image.",
        "hallucination": False,
        "error_type": "None"
    },
    59: {
        "score": 1,
        "explanation": "Ground truth states there are 5 black hats. The model claims there are 6 black hats, an over-counting error.",
        "hallucination": True,
        "error_type": "Counting Hallucination (Over-counting)"
    },
    60: {
        "score": 5,
        "explanation": "The model accurately describes how the two watches are placed, noting the black watch on top of the red watch and strap overlap.",
        "hallucination": False,
        "error_type": "None"
    },
    61: {
        "score": 5,
        "explanation": "The model correctly determines the photo is taken indoors due to carpeted floor, matching ground truth.",
        "hallucination": False,
        "error_type": "None"
    },
    62: {
        "score": 5,
        "explanation": "The model gives an accurate holistic description of the bride's hand resting on the groom's hand with wedding rings in black and white.",
        "hallucination": False,
        "error_type": "None"
    },
    63: {
        "score": 4,
        "explanation": "The model describes a blurred reflection of a person wearing a hat, which is consistent with the vague reflections in the sunglasses described in GT.",
        "hallucination": False,
        "error_type": "None"
    },
    64: {
        "score": 5,
        "explanation": "The model correctly answers that the laptop is silver in color.",
        "hallucination": False,
        "error_type": "None"
    },
    65: {
        "score": 5,
        "explanation": "Adversarial question asking what is playing on the monitor when there is no monitor. The model responds 'nothing', successfully rejecting the false object.",
        "hallucination": False,
        "error_type": "None"
    },
    66: {
        "score": 4,
        "explanation": "The model correctly identifies the device on the left as larger.",
        "hallucination": False,
        "error_type": "None"
    },
    67: {
        "score": 1,
        "explanation": "Ground truth states there is 1 mobile phone. The model claims there are 2 mobile phones (over-counting error).",
        "hallucination": True,
        "error_type": "Counting Hallucination (Over-counting)"
    },
    68: {
        "score": 1,
        "explanation": "Question asks if the mouse is connected to a computer. GT states No. The model asserts that 'The image shows a white computer mouse connected to a computer via a USB cable... indicating that it is indeed connected to a computer'. False relational assertion.",
        "hallucination": True,
        "error_type": "Relational Hallucination"
    },
    69: {
        "score": 4,
        "explanation": "The model correctly identifies the setting as 'mountain', matching ground truth outdoor mountain scene.",
        "hallucination": False,
        "error_type": "None"
    },
    70: {
        "score": 4,
        "explanation": "The model identifies the television showing a man in suit and tie with a book and pen nearby. It describes it as mounted on a wall rather than resting on a table, but core content is grounded.",
        "hallucination": False,
        "error_type": "None"
    },
    71: {
        "score": 3,
        "explanation": "Question asks whether laptop is Macbook or Windows. GT notes Apple logo on keyboard. The model cannot identify it and states it cannot be definitively determined. Uninformative/hedging without false claims.",
        "hallucination": False,
        "error_type": "Recognition Failure (Hedging)"
    },
    72: {
        "score": 1,
        "explanation": "Question asks for colors of pillows from left to right (white/grey, yellow, white/grey). The model claims 'light blue, yellow, and light blue', hallucinating light blue for the white/grey pillows.",
        "hallucination": True,
        "error_type": "Attribute / Color Hallucination"
    },
    73: {
        "score": 0,
        "explanation": "Adversarial question asking the color of the chair when there is no chair in the image. The model hallucinates 'The color of the chair in the image is beige.' Direct ungrounded hallucination.",
        "hallucination": True,
        "error_type": "Adversarial Susceptibility / False Object Hallucination"
    },
    74: {
        "score": 5,
        "explanation": "The model correctly determines that the pillow on the right is closer to the camera.",
        "hallucination": False,
        "error_type": "None"
    },
    75: {
        "score": 5,
        "explanation": "The model correctly counts 2 chairs in the image, matching ground truth.",
        "hallucination": False,
        "error_type": "None"
    },
    76: {
        "score": 5,
        "explanation": "The model correctly identifies that the pillow in the middle is in front of the other pillows.",
        "hallucination": False,
        "error_type": "None"
    },
    77: {
        "score": 1,
        "explanation": "Question asks what weather was like when photo was taken. GT states it was night and sky dark. The model claims weather was 'sunny, as the living room is well-lit with natural light coming through windows'. Complete hallucination of daytime/sunlight.",
        "hallucination": True,
        "error_type": "Environmental Attribute Hallucination"
    },
    78: {
        "score": 2,
        "explanation": "Ground truth states there are two deck beds. The model hallucinates 'three bunk beds', though it correctly notes blue and yellow bedding.",
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
        "explanation": "Question asks what color the books are (GT: red and white). The model answers 'red'. It captures the primary color accurately, though omits white.",
        "hallucination": False,
        "error_type": "None"
    },
    81: {
        "score": 5,
        "explanation": "Adversarial question asking what is in the vase when it is empty. The model correctly answers 'nothing'.",
        "hallucination": False,
        "error_type": "None"
    },
    82: {
        "score": 4,
        "explanation": "The model correctly compares the vases and states 'The one on the right' looks larger.",
        "hallucination": False,
        "error_type": "None"
    },
    83: {
        "score": 1,
        "explanation": "Ground truth states there are 6 teddy bears on the stairs. The model counts 5 teddy bears, under-counting by one.",
        "hallucination": True,
        "error_type": "Counting Hallucination (Under-counting)"
    },
    84: {
        "score": 5,
        "explanation": "The model correctly states that blue flowers are on top and white flowers are on the bottom.",
        "hallucination": False,
        "error_type": "None"
    },
    85: {
        "score": 4,
        "explanation": "Question asks for weather in an indoor scene. The model reasons that weather cannot be determined because it is an indoor scene. Legitimate reasoning without false claims.",
        "hallucination": False,
        "error_type": "None"
    },
    86: {
        "score": 5,
        "explanation": "The model describes a baby sleeping with 3 stuffed animals hanging above, matching ground truth.",
        "hallucination": False,
        "error_type": "None"
    },
    87: {
        "score": 4,
        "explanation": "Question asks for book title ('Proverbial Philosophy'). The model reads 'ROVERBIAL PHILOSOPHY', missing the first letter 'P' due to decorative initial cap font. Highly accurate visual reading without hallucination.",
        "hallucination": False,
        "error_type": "Minor OCR Truncation"
    },
    88: {
        "score": 5,
        "explanation": "The model correctly identifies the pot color as silver.",
        "hallucination": False,
        "error_type": "None"
    },
    89: {
        "score": 4,
        "explanation": "Adversarial question asking what is being washed in the sink. GT states nothing is being washed. The model responds that there is no one being washed in the sink. Effectively rejects the premise.",
        "hallucination": False,
        "error_type": "None"
    },
    90: {
        "score": 4,
        "explanation": "The model correctly determines that the washing machine on the right is taller.",
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
        "explanation": "The model correctly identifies that a screwdriver is placed on the tissue paper roll.",
        "hallucination": False,
        "error_type": "None"
    },
    93: {
        "score": 4,
        "explanation": "Question asks what time photo was taken (GT: daytime). The model answers 'afternoon', which is accurate and grounded.",
        "hallucination": False,
        "error_type": "None"
    },
    94: {
        "score": 5,
        "explanation": "The model accurately describes white steamed buns (baozi) on paper in a container/tray.",
        "hallucination": False,
        "error_type": "None"
    },
    95: {
        "score": 5,
        "explanation": "The model correctly identifies that the reflection of the washing machine is shown in the mirror.",
        "hallucination": False,
        "error_type": "None"
    }
}

# Combine and compute statistics
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
    
    item_res = {
        "index": i,
        "question_type": d["question_type"],
        "question_topic": d["question_topic"],
        "question": d["question"],
        "gt_answer": d["gt_answer"],
        "model_answer": d["model_answer"],
        "score": score,
        "hallucination": hal,
        "error_type": e["error_type"],
        "explanation": e["explanation"]
    }
    results.append(item_res)

with open('phi4_mmhal_evaluated.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Total samples: {len(results)}")
print(f"Average Score (0-6 scale): {sum(score_list)/len(score_list):.2f}")
print(f"Average Score (0-5 normalized scale): {sum(score_list)/len(score_list) / 6.0 * 5.0:.2f}")
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
