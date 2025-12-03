from pathlib import Path
import re
from typing import Tuple, List, Optional, Union

import matplotlib.pyplot as plt
import pyvips

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.openai import OpenAIResponsesModelSettings, OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

import logging

logger = logging.getLogger(__name__)

# === Configurable values ===
large_image_path = Path('IMG') / 'cars.jpg'  # change to your large image file
MAX_PATCH_SIZE = 256  # largest side of extracted patch in pixels
# Question templates to ask the model while searching
class QueryBack:
    __L_INIT_DELIMITER = '{'
    __R_INIT_DELIMITER = '}'
    __L_DELIMITER = '{'
    __R_DELIMITER = '}'
    def __init__(self, question_id: str,*args):
        query_template = f'{self.__L_INIT_DELIMITER}MQ_{question_id}|optional id keyword{self.__R_INIT_DELIMITER}'
        query_regex_str = rf"{self.__L_INIT_DELIMITER}MQ_{question_id}(?:\|([^{self.__R_INIT_DELIMITER}]*))?{self.__R_INIT_DELIMITER}"
        query_regex_group_mapping = []
        for i, value in enumerate(args):
            query_template += f"{self.__L_DELIMITER}{value}{self.__R_DELIMITER}"
            query_regex_str += rf"{self.__L_DELIMITER}([^{self.__R_DELIMITER}]*){self.__R_DELIMITER}"
            query_regex_group_mapping.append(value)
        self.query_template = query_template
        self.query_regex = re.compile(query_regex_str, re.IGNORECASE)
        self.query_regex_group_mapping = query_regex_group_mapping
    
    def __repr__(self) -> str:
        return f"Question(template={self.query_template})"
    
    def __str__(self) -> str:
        return self.query_template
    
    def __call__(self, text: str) -> tuple[dict, list]:
        match = re.finditer(self.query_regex, text)
        result = {}
        duplicate_questions = []
        for m in match:
            key = m.group(1)
            if key in result:
                duplicate_questions.append(key)
            else:
                result[key] = {}
            for i, group_name in enumerate(self.query_regex_group_mapping, start=2):
                result[key][group_name] = m.group(i)
        return result, duplicate_questions




    

# Tokens/formats the model should use
# # Request a patch with relative bounding box (fractions): REGION_REQUEST_REL:<x>,<y>,<w>,<h> (values from 0.0 to 1.0)
# RE_REGION_REQUEST_REL = re.compile(r'REGION_REQUEST_REL[:\s]*([0-9]*\.?[0-9]+),\s*([0-9]*\.?[0-9]+),\s*([0-9]*\.?[0-9]+),\s*([0-9]*\.?[0-9]+)')
# # Final Waldo answer must be: WALDO_POS:<x_rel>,<y_rel> where x_rel,y_rel are floats in [0,1] representing center
# RE_WALDO_POS = re.compile(r'WALDO_POS[:\s]*([0-9]*\.?[0-9]+),\s*([0-9]*\.?[0-9]+)')
# # Answers for auxiliary questions: ANSWER[<key>]: <text>
# RE_ANSWER = re.compile(r'ANSWER\[(?P<q>[^\]]+)\]:(?P<text>.*)', re.IGNORECASE)
Q_REGION_REQUEST = QueryBack("REGION_REQUEST", "x", "y", "w", "h")
Q_ANSWER_POS = QueryBack("WALDO_POS", "x", "y")


# === Helpers: image & region handling ===

def validate_rel_box(image: pyvips.Image, rel_box: Tuple[float, float, float, float]) -> Optional[str]:
    """Validate rel_box (x_frac, y_frac, w_frac, h_frac).

    Returns None if valid, otherwise an error message string.
    """
    orig_w, orig_h = image.width, image.height
    try:
        x_frac, y_frac, w_frac, h_frac = rel_box
    except Exception:
        return 'rel_box must be a 4-tuple of floats (x,y,w,h)'

    # Validate fractions are numbers and in [0,1]
    for name, val in (('x', x_frac), ('y', y_frac), ('w', w_frac), ('h', h_frac)):
        try:
            f = float(val)
        except Exception:
            return f'Value for {name} is not a float: {val!r}'
        if not (0.0 <= f <= 1.0):
            return f'Value for {name} ({f}) is out of range [0.0, 1.0]'

    x_frac, y_frac, w_frac, h_frac = float(x_frac), float(y_frac), float(w_frac), float(h_frac)

    # Ensure box is not empty and does not exceed image bounds
    if w_frac <= 0.0 or h_frac <= 0.0:
        return 'Width and height fractions must be greater than 0'
    if x_frac < 0.0 or y_frac < 0.0:
        return 'x and y fractions must be >= 0.0'
    if x_frac + w_frac > 1.0 + 1e-9 or y_frac + h_frac > 1.0 + 1e-9:
        return 'Requested region exceeds image bounds (x+w>1.0 or y+h>1.0)'

    # Convert to pixel bbox and final pixel checks
    left = int(round(x_frac * orig_w))
    top = int(round(y_frac * orig_h))
    width = int(round(w_frac * orig_w))
    height = int(round(h_frac * orig_h))

    if width <= 0 or height <= 0:
        return 'Requested region maps to an empty pixel area'
    if left < 0 or top < 0 or left >= orig_w or top >= orig_h:
        return 'Requested region top-left is outside image bounds'
    if left + width > orig_w or top + height > orig_h:
        return 'Requested pixel region exceeds image bounds'

    return None


def extract_patch_relative(image: pyvips.Image, rel_box: Tuple[float, float, float, float], max_side: int = 512) -> tuple[pyvips.Image, float]:
    """Extract a patch assuming rel_box is already validated.

    Returns (patch_image, scale). Caller should run `validate_rel_box` first.
    """
    orig_w, orig_h = image.width, image.height
    x_frac, y_frac, w_frac, h_frac = map(float, rel_box)

    # convert to pixel bbox
    left = int(round(x_frac * orig_w))
    top = int(round(y_frac * orig_h))
    width = int(round(w_frac * orig_w))
    height = int(round(h_frac * orig_h))

    patch = image.crop(left, top, width, height)

    # resize if needed
    larger = max(width, height)
    scale = float(min(larger, max_side)) / float(larger) if larger > 0 else 1.0
    if scale != 1.0:
        patch = patch.resize(scale)

    return patch, scale




# === Iterative loop that evaluates against a true pixel position ===
def iterative_search_in_image(
        agent: Agent,
        image_to_search: pyvips.Image,
        search_target_image: BinaryContent, 
        true_pos_px: Tuple[int, int], 
        max_iters: int = 10, 
        verbose: bool = True
) -> Union[tuple[list[tuple[float, float]], list[tuple[float, float, float, float]]], tuple[None, str]]:
    # true_pos_px: (x_px, y_px) in original image pixel coordinates. If not provided, loop will run until model returns WALDO_POS.
    orig_w, orig_h = image_to_search.width, image_to_search.height
    if verbose: print(f'Large image size: {orig_w}x{orig_h}')
    answer_guesses: list[tuple[float, float]] = []
    tile_requests: List[tuple[float, float, float, float]] = []  # list of (x_frac, y_frac, w_frac, h_frac)
    
    # prepare initial prompt
    initial_prompt = [
        f'You are searching for clues in a large image. You can only request to see parts of the image by specifying their relative bounding boxes.',
        f'To request a patch, use request in the form {Q_REGION_REQUEST}. Values are fractions (0.0-1.0) of the full image dimensions. x,y are measured from TOP-LEFT corner and w,h are width and height of the requested patch.',
        f'Each value in your requests must be enclosed in curly braces.',
        f'I will respond with image patches scaled so their larger side is at most {MAX_PATCH_SIZE} px, so you have to request smaller regions to see more detail. In your response, describe what you see in each patch. Try to ask for only a few patches at a time to keep the context size manageable.',
        f'When you are confident you know the answer, reply with: {Q_ANSWER_POS} giving the RELATIVE CENTER (0-1).',
        f"Your goal is to find a yellow car which is being stolen. If you answer incorrectly, I will tell you how far your answer is from the true position in pixels.",
        'Start by asking for the image patches covering larger areas and then ask for smaller, more detailed patches to inspect parts of the larger regions in more detail. Adhere strictly to the request templates.',
        'No further instructions will be provided.'
    ]
    print(initial_prompt)
    model_settings = OpenAIResponsesModelSettings(openai_previous_response_id='auto')
    try:
        result = agent.run_sync(initial_prompt, model_settings=model_settings)
    except Exception as e:
        return None, f'Initial agent run failed: {e}'
    reply = result.output
    it = 0
    while it < max_iters:
        it += 1
        if verbose: print(f'\n=== Iteration {it} ===\n{reply}')
        next_prompt = []

        # check if WALDO_POS is in reply
        waldo_pos_answers, _ = Q_ANSWER_POS(reply)
        if len(waldo_pos_answers.keys()) > 1:
            next_prompt.append('You have provided multiple ANSWER_POS answers. Please provide only one final answer with the relative position.')
        elif len(waldo_pos_answers.keys()) == 1:
            # extract the single answer
            for key, answer in waldo_pos_answers.items():
                x_rel = float(answer['x'])
                y_rel = float(answer['y'])
                answer_guesses.append((x_rel, y_rel))
                # check against true position
                true_x, true_y = true_pos_px
                dist_px = ((x_rel * orig_w - true_x) ** 2 + (y_rel * orig_h - true_y) ** 2) ** 0.5
                if dist_px < 100:  # within 100 pixels is considered correct
                    break  # exit the loop
                else:
                    if verbose:
                        print(f'Answer guess ({x_rel:.4f}, {y_rel:.4f}) is {dist_px:.1f} pixels away from true position.')
                    next_prompt.append(f'Your ANSWER_POS answer ({x_rel:.4f}, {y_rel:.4f}) is {dist_px:.1f} pixels away from the true position. Please try again.')
            # continue to next iteration if not returned


        # Determine which patch to request
        queries, duplicates = Q_REGION_REQUEST(reply)
        if len(queries) == 0:
            next_prompt.append(f'No REGION_REQUEST found in your reply. Include all special characters in the form {Q_REGION_REQUEST}, if you want to request image patches.')
        else:
            for dup in duplicates:
                next_prompt.append(f'You have duplicate REGION_REQUEST for id "{dup}". Keep identifiers for questions unique.')
            for key, query in queries.items():
                rel_box = (float(query['x']), float(query['y']), float(query['w']), float(query['h']))
                # validate first
                v_err = validate_rel_box(image_to_search, rel_box)
                if v_err is not None:
                    if verbose:
                        print(f'Invalid REGION_REQUEST for {key}: {v_err}')
                    next_prompt.append(f'Invalid REGION_REQUEST for {key}: {v_err}')
                    continue

                # extraction assumes rel_box is valid; catch unexpected runtime errors
                try:
                    patch, scale = extract_patch_relative(image_to_search, rel_box, max_side=MAX_PATCH_SIZE)
                except Exception as e:
                    err = f'Failed to extract patch for {key}: {e}'
                    if verbose:
                        print(err)
                    next_prompt.append(f'Invalid REGION_REQUEST for {key}: {err}')
                    continue

                # safe attempt to display patch
                if verbose:
                    try:
                        print(f'Requested patch {key}: (x={query["x"]}, y={query["y"]}, w={query["w"]}, h={query["h"]}), scale={scale:.4f}')
                        arr = patch.write_to_memory()
                        try:
                            import numpy as _np
                            bands = patch.bands
                            h = patch.height
                            w = patch.width
                            img = _np.frombuffer(arr, dtype=_np.uint8)
                            img = img.reshape((h, w, bands))
                            plt.imshow(img)
                            plt.axis('off')
                            plt.show()
                        except Exception:
                            pass
                    except Exception:
                        pass

                tile_requests.append((
                    float(query['x']),
                    float(query['y']),
                    float(query['w']),
                    float(query['h'])
                ))
                next_prompt.append(f'Patch {key}:')
                try:
                    next_prompt.append(BinaryContent(data=patch.write_to_buffer('.png'), media_type='image/png'))
                except Exception as e:
                    # If writing buffer fails, report and continue
                    err = f'Failed to serialize patch for {key}: {e}'
                    if verbose:
                        print(err)
                    next_prompt.append(f'Invalid REGION_REQUEST for {key}: {err}')
                    continue
        
        if verbose: print(f'...advancing to iteration {it+1}')
        print(next_prompt)
        # Send prompt with patches
        try:
            result = agent.run_sync(next_prompt, model_settings=model_settings, message_history=result.new_messages())
        except Exception as e:
            return None, f'Agent run failed during iteration {it}: {e}'
        reply = result.output
    return answer_guesses, tile_requests
        



# large_image = pyvips.Image.new_from_file(large_image_path, access='sequential')
# search_target_image = BinaryContent(data=Path('IMG/waldo_small.png').read_bytes(), media_type='image/png')
# guesses, ROIs = iterative_search_for_waldo(image_to_search=large_image, search_target_image=search_target_image, true_pos_px=(473, 1550), max_iters=4, verbose=True)



