## This script reweights the system modality
import os
import json
import argparse
import shortuuid
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

import torch
from torch.utils.data import Dataset, DataLoader

import math
import shutil
import random
from PIL import Image
from transformers import set_seed
from pycocotools.coco import COCO

from io import BytesIO ##

##
import csv
import torch.nn.functional as F
import numpy as np


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

###
def safe_to_float(x):
    """Convert objects that are possibly GPU tensors or arrays to floats."""
    if torch.is_tensor(x):
        if x.numel() == 1:
            return x.detach().cpu().item()
        return float(x.mean().detach().cpu())  # fallback for vectors
    return float(x)

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        ## Handle image_file as a path or PIL image
        if isinstance(image_file, str):
            image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        elif isinstance(image_file, Image.Image):
            image = image_file
        else:
            raise TypeError(f"Unknown type for image_file: {type(image_file)}")  # Defensive
        
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):

    ################FUNCTIONS FOR ATTN MATRICES################

    MODALITY_LABELS = ["system", "image", "text"]

    dataset_type = "reduced" if args.use_reduced else "full"

    # Output file naming
    attn_file_parts = [
        f"y25_target_{args.dataset}_attn",
        f"_sys_{args.system_alpha}" if args.reweight_system else "",
        f"_to_{args.redistribute_to}" if args.redistribute_to in ['img', 'text', 'both', 'prop'] else "",
        f'_k_{args.top_k}',
        f'_heads_{args.attention_head_path.split("/")[-1].replace(".json","")}',
        f'{dataset_type}', ##
        f'_seed_{args.seed}',
    ]
    attn_file_name = "".join(attn_file_parts)

    def compute_attention_entropy_per_head(attns):
        """Compute entropy for each attention head separately."""
        # attn_probs shape: (heads, queries, keys)
        # Calculate entropy per head (averaging across queries)
        attn_probs = F.softmax(attns, dim=-1)
        entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-9), dim=-1)  # (heads, queries)
        return entropy.mean(dim=1)  # Mean over queries only, keeping head dimension (heads,)

    def normalise(text, dataset, category=None):
        """Normalise text based on dataset-specific logic."""
        import re
        
        if isinstance(text, int):
            return text
        if text is None:
            return text

        # Strip leading/trailing whitespace but preserve original for first-token check
        text_stripped = text.strip()
        
        if dataset in ['pope', 'mme', 'sugarcrepe', 'hallusionbench', 'beaf', 'winoground']:  # Binary datasets
            # Check if first token (ignoring punctuation) is yes/no
            match = re.match(r'^(yes|no)[\s,\.!?;:]*', text_stripped, re.IGNORECASE)
            if match:
                answer = match.group(1).lower()
                # For all binary datasets convert yes→1, no→0
                return 1 if answer == 'yes' else 0
            # If first token is not yes/no, try to convert to int if it's a digit
            elif text_stripped.isdigit():
                return int(text_stripped)
            else:
                return text_stripped.lower().rstrip(".")  # Return as is if not yes/no

        elif dataset == 'naturalbench':  # NaturalBench-specific logic
            if category == 'yes_no':  # Yes/No questions
                # Check if first token (ignoring punctuation) is yes/no
                match = re.match(r'^(yes|no)[\s,\.!?;:]*', text_stripped, re.IGNORECASE)
                if match:
                    return match.group(1).lower()
                else:
                    return text_stripped.lower().rstrip(".")
            elif category == 'multiple_choice':  # A/B questions
                # Check if first token (ignoring punctuation) is A/B
                match = re.match(r'^([AB])[\s,\.!?;:]*', text_stripped, re.IGNORECASE)
                if match:
                    choice = match.group(1).upper()
                    return 'a' if choice == 'A' else 'b'
                else:
                    return text_stripped.lower().rstrip(".")
    
        return text_stripped.lower().rstrip(".")  # Default: return the text as is

    #################################################

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    if args.dataset == 'coco':
        caption_file_path = args.caption_file_path
        coco = COCO(caption_file_path)
        img_ids = coco.getImgIds()
        sampled_img_ids = random.sample(img_ids, args.num_samples)

        questions = []
        dest_image_folder = os.path.join(os.path.split(os.path.split(os.path.dirname(args.answers_file))[0])[0], 'images', f'seed{args.seed}_{args.num_samples}')
        os.makedirs(dest_image_folder, exist_ok=True)
        for sampled_img_id in sampled_img_ids:
            image_file = coco.loadImgs(sampled_img_id)[0]["file_name"]
            question = {
                "question_id": sampled_img_id,
                "image": image_file,
                "text": "Please describe this image in detail.",
            }
            shutil.copyfile(os.path.join(args.image_folder, image_file), os.path.join(dest_image_folder, image_file))
            questions.append(question)

    elif args.dataset == 'nocaps':  
        caption_file_path = args.caption_file_path
        val_caps = json.load(open(caption_file_path))
        image_infos = val_caps["images"]
        out_image_infos = [image_info for image_info in image_infos if image_info['domain'] == 'out-domain']
        sampled_img_infos = random.sample(out_image_infos, args.num_samples)

        questions = []
        dest_image_folder = os.path.join(os.path.split(os.path.split(os.path.dirname(args.answers_file))[0])[0], 'images', f'seed{args.seed}_{args.num_samples}')
        os.makedirs(dest_image_folder, exist_ok=True)
        for sampled_img_info in sampled_img_infos:
            image_file = sampled_img_info['file_name']
            image_id = sampled_img_info['id']
            question = {
                "question_id": sampled_img_info['id'],
                "image": sampled_img_info['file_name'],
                "text": "Please describe this image in detail.",
            }
            shutil.copyfile(os.path.join(args.image_folder, image_file), os.path.join(dest_image_folder, f'{image_id}_{image_file}'))
            questions.append(question)

    elif args.dataset == 'mme':
        from datasets import load_dataset ##
        import requests ##

        questions = []
        if args.use_reduced:
            print("Using reduced MME subset for faster evaluation.")
            hf_ds = load_dataset("tsch00001/mme_reduced", split='train') # Reduced & balanced, 560 samples
        else:
            print("Using full MME dataset for evaluation.")
            # Full dataset
            hf_ds = load_dataset("lmms-lab/MME", split='test')

        for question in hf_ds:
            if isinstance(question['image'], str):
                response = requests.get(question["image"])
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = question["image"].convert("RGB")

            query = question['question'] # Original prompt already requests for y/n answers. #+ " Answer only 'yes' or 'no'." 
            label = 1 if question["answer"].strip().lower() == "yes" else 0
            category = question['category']

            question = {
                "question_id": f'{question['question_id']}_{label}', ####
                'image': image,
                "text": query,
                'label': label, ##
                'category': category ##           
            }
            questions.append(question)

    elif args.dataset == 'pope':
        from datasets import load_dataset ##
        import requests ##

        questions = []
        if args.use_reduced:
            print("Using reduced POPE subset for faster evaluation.")
            hf_ds = load_dataset("tsch00001/pope_300_per_cat", split='train') # Reduced & balanced, 900 samples
        else:
            print("Using full POPE dataset for evaluation.")
            # Full dataset
            hf_ds = load_dataset("lmms-lab/POPE", split='test')

        for question in hf_ds:
            if isinstance(question['image'], str):
                response = requests.get(question["image"])
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = question["image"].convert("RGB")

            query = question['question'] + " Answer only 'yes' or 'no'." # Explicitly request yes/ no responses
            label = 1 if question["answer"].strip().lower() == "yes" else 0
            category = question['category']

            question = {
                "question_id": question['id'],
                "image": image, ##
                "text": query,
                'label': label, ##
                'category': category, ##
                'image_source': question['image_source'] ## Keep original image URL    
            }
            questions.append(question)

    elif args.dataset == 'hallusionbench':
        from datasets import load_dataset ##
        import requests ##

        questions = []
        # Use full dataset as only 951 samples
        print("Using full HallusionBench dataset for evaluation.")
        hf_ds = load_dataset("lmms-lab/HallusionBench", split='image')

        for question in hf_ds:
            if isinstance(question['image'], str):
                response = requests.get(question["image"])
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = question["image"].convert("RGB")

            query = question['question'] +  " Answer only 'yes' or 'no'." # Explicitly request yes/ no responses
            label = question["gt_answer"]
            category = question['subcategory']

            question = {
                "question_id": f'{question['question_id']}_{question['filename']}', ####
                "image": image, ##
                "text": query,
                'label': label, ##
                'category': category, ##
                'image_source': question['filename'] ##
            }
            questions.append(question)

    elif args.dataset == 'naturalbench':
        from datasets import load_dataset ##
        import requests ##

        questions = []
        if args.use_reduced:
            print("Using reduced NaturalBench subset for faster evaluation.")
            hf_ds = load_dataset("tsch00001/naturalbench_450_per_cat", split='train') # Reduced & balanced, 900 samples
        else:
            print("Using full NaturalBench dataset for evaluation.")
            # Full dataset
            hf_ds = load_dataset("BaiqiL/NaturalBench-lmms-eval", split='test')
        
        for question in hf_ds:
            if isinstance(question['Image'], str):
                response = requests.get(question["Image"])
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = question["Image"].convert("RGB")

            q = question['Question']
            q_type = question['Question_Type']

            if q_type == "yes_no":
                suffix = " Answer only 'yes' or 'no'."
            elif q_type == "multiple_choice":
                suffix = " Answer with either A or B."

            query = q + suffix # Explicitly request one-token responses
            label = question['Answer'].strip().lower()

            question = {
                "question_id": question['Index'],
                "image": image, ##
                "text": query,
                'label': label, ##
                'category': q_type, ##
            }
            questions.append(question)

    elif args.dataset == 'sugarcrepe':
        from datasets import load_dataset ##
        import requests ##

        questions = []
        if args.use_reduced:
            print("Using reduced SugarCrepe subset for faster evaluation.")
            hf_ds = load_dataset("tsch00001/sugarcrepe_130_per_cat", split='train') # Reduced & balanced, 910 samples
        else:
            print("Using full SugarCrepe dataset for evaluation.")
            # Full dataset
            hf_ds = load_dataset("yjkimstats/SUGARCREPE_fmt", split='train')

        for item in hf_ds:
            # Process prompts from both the caption and negative_caption columns
            captions_data = [
                {'text': item['caption'], 'ground_truth': 'yes', 'type': 'positive'},
                {'text': item['negative_caption'], 'ground_truth': 'no', 'type': 'negative'}  # If text is from caption column
            ]
            if isinstance(item['images'], str):
                response = requests.get(item["images"])
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = item["images"].convert("RGB")

            for caption_info in captions_data:
                query = 'Does this image show: ' + caption_info['text'] + " Answer only 'yes' or 'no'." # Explicitly request yes/ no responses
                label = 1 if caption_info['type'] == "positive" else 0
                category = item['Negative_type']

                question = {
                    "question_id": f"{item['filename']}_{caption_info['type']}",
                    "image": image, ##
                    "text": query,
                    'label': caption_info['ground_truth'], ##
                    'category': category, ##
                    'image_source': caption_info['type'] ##    
                }
                questions.append(question)

    elif args.dataset == 'beaf':
        from datasets import load_dataset ##
        import requests ##
        import os ##

        questions = []
        if args.use_reduced:
            print("Using reduced BEAF subset for evaluation.")
            hf_ds = load_dataset("tsch00001/kopie_beaf", split='train') # Reduced, 9000 samples
            hf_ds = hf_ds.select(range(9000))
        else:
            print("Using full BEAF dataset for evaluation.")
            # Full dataset
            hf_ds = load_dataset("tsch00001/kopie_beaf", split='train')

        for question in hf_ds:
            if isinstance(question['image_data'], str):
                response = requests.get(question["image_data"])
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = question["image_data"].convert("RGB")

            query = question['question'] + " Answer only 'yes' or 'no'." # Explicitly request yes/ no responses
            label = 1 if question["gt"].strip().lower() == "yes" else 0
            category = f"{question['orig_img']}_{question['removed_q']}"

            question = {
                "question_id": question['id'],
                "image": image, ##
                "text": query,
                'label': label, ##
                'category': category, ##
                'image_source': question['image']   
            }
            questions.append(question)        

    elif args.dataset == 'whoops':
        from datasets import load_dataset ##
        import requests ##

        questions = []
        print("Using full Whoops! dataset for evaluation.") 
        hf_ds = load_dataset("nlphuji/whoops", split='test')

        for question in hf_ds:
            
            if isinstance(question['image'], str):
                response = requests.get(question["image"])
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = question["image"].convert("RGB")

            query = "Does this image show a normal scene? Explain." # 'Weird' used in paper - hence also try 'Is this image weird?'

            label = question['designer_explanation'] # All images are weird
            category = question['commonsense_category']

            question = {
                "question_id": question['image_id'],
                "image": image, ##
                "text": query,
                'label': label, ##
                'category': category
            }
            questions.append(question)
    
    elif args.dataset == 'winoground':
        from datasets import load_dataset
        import requests

        questions = []
        print("Using full Winoground dataset for evaluation.") 
        hf_ds = load_dataset("facebook/winoground", split="test", token=os.environ["HUGGINGFACE_HUB_TOKEN"]) 
        
        for item in hf_ds:

            # Define four passes
            passes = [
                ("image_0", "caption_0", True),
                ("image_0", "caption_1", False),
                ("image_1", "caption_0", False),
                ("image_1", "caption_1", True),
            ]

            for image_field, caption_field, gt in passes:
                # Retrieve image
                image_data = item[image_field]

                # Images are already PIL
                if isinstance(image_data, Image.Image):
                    image = image_data.convert("RGB")

                caption_text = item[caption_field]

                query = (
                    "Does this image show: "
                    + caption_text
                    + "? Answer only 'yes' or 'no'."
                )

                # image_source: textual indicator of ground-truth correctness
                q_type = "true" if gt else "false"

                question = {
                    "question_id": f"{item['id']}_{image_field}_{caption_field}",
                    "image": image,
                    "text": query,
                    "label": gt,
                    "category": f"{item['collapsed_tag']}_{q_type}",
                    "image_source": f"{item['tag']}_{item['num_main_preds']}",
                }

                questions.append(question)

    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented yet.")

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    if not args.random_heads:
        with open(args.attention_head_path, 'r') as file:
            data_loaded = json.load(file)
        hal_attention_heads = data_loaded['hal_heads'][:args.top_k]
    else:
        all_heads = [[i,j] for i in range(32) for j in range(32)]
        hal_attention_heads = random.sample(all_heads, args.top_k)

    #############ATTN MATRIX OUTPUT#############
    with open(f"{attn_file_name}.csv", mode="w", newline="") as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow([
            "Sample", 'Subset', "Layer", "Head", "To", "Entropy",
            "Mean_Token_Attn_Weight", 'Attention_Weight', 'Num_To_Tokens',
            "Model_Output", "Ground_Truth", "Correct"
        ])
        #############ATTN MATRIX OUTPUT#############

        count = 0
        for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
            count += 1
            question_id = line["question_id"]
            cur_prompt = line["text"]
            image_file = line["image"]

            input_ids = input_ids.to(device='cuda', non_blocking=True)
            image_tensor = image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)
            
            model.config.reweight_system = args.reweight_system ##

            model.config.system_alpha = args.system_alpha ##
            #model.config.reweight_alpha = args.reweight_alpha
            model.config.hal_attention_heads = hal_attention_heads
            model.config.img_start_pos = 35
            model.config.img_length = 576
            model.config.redistribute_to = args.redistribute_to

            with torch.inference_mode():
                output_dict = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    output_attentions=True,
                    return_dict_in_generate=True)

            output_ids = output_dict['sequences']
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print(question_id, outputs)

            ## Answer dict
            answer_dict = dict(line)

            # Add model output
            answer_dict['prompt'] = cur_prompt
            answer_dict['text'] = outputs
            answer_dict['model_id'] = model_name
            if isinstance(answer_dict['image'], str):
                pass  # COCO/Nocaps: already a filename
            elif 'image_source' in answer_dict:
                answer_dict['image'] = answer_dict['image_source']  # POPE: use URL
            else:
                answer_dict['image'] = ''  # fallback for MME

            if args.dataset in ['coco', 'nocaps']:
                answer_dict['answer_id'] = shortuuid.uuid()
                    
            ans_file.write(json.dumps(answer_dict) + "\n")
            ans_file.flush()

            #############ANALYSE ATTN WEIGHTS BEFORE & AFTER REWEIGHTING#############
            all_stats = []
            for layer in model.model.layers:
                if hasattr(layer.self_attn, "collected_stats"):
                    all_stats.append(layer.self_attn.collected_stats)

            for stage in ["pre_alpha", "post_alpha"]:
                all_layer_stats = []
                for stat_dict in all_stats:
                    all_layer_stats.extend(stat_dict[stage])
                # Compute means and variances
                if not all_layer_stats:  # skip if no stats
                    continue
                # Convert everything to CPU floats
                means_system = np.mean([safe_to_float(s["system_mean"]) for s in all_layer_stats])
                means_img = np.mean([safe_to_float(s["img_mean"]) for s in all_layer_stats])
                means_text = np.mean([safe_to_float(s["text_mean"]) for s in all_layer_stats])

                props_img = np.mean([safe_to_float(s["img_prop"]) for s in all_layer_stats])
                props_sys = np.mean([safe_to_float(s["system_prop"]) for s in all_layer_stats])
                props_txt = np.mean([safe_to_float(s["text_prop"]) for s in all_layer_stats])

                var_img = np.var([safe_to_float(s["img_mean"]) for s in all_layer_stats])
                var_sys = np.var([safe_to_float(s["system_mean"]) for s in all_layer_stats])
                var_txt = np.var([safe_to_float(s["text_mean"]) for s in all_layer_stats])

                print(f"== {stage} ==")
                print(f"Mean system: {means_system:.4f}, img: {means_img:.4f}, txt: {means_text:.4f}")
                print(f"Mean proportions: sys={props_sys:.2%}, img={props_img:.2%}, txt={props_txt:.2%}")
                print(f"Variance (means): sys={var_sys:.5f}, img={var_img:.5f}, txt={var_txt:.5f}")
                print("")

            ################GET ATTN MATRICES################
            attentions = output_dict.get('attentions', None)
            if attentions is not None and len(attentions) > 0:
                # Establish modality indices (system, image, text)
                # Assumes batch size of 1
                original_len = input_ids.shape[1]
                total_len = attentions[0][0].shape[-1]
                img_start_idx = model.config.img_start_pos # Unchanged by expansion
                num_patches = total_len - original_len + 1 # No. of image patches after expansion
                system_indices = list(range(img_start_idx))
                img_end_idx = img_start_idx + num_patches - 1
                image_indices = list(range(img_start_idx, img_end_idx + 1))
                text_indices = list(range(img_end_idx + 1, total_len))

                # Add safety check
                if not text_indices:
                    print(f"[WARNING] No text tokens found for sample {line['question_id']}")

                print(f"[DEBUG] Original sequence length: {original_len}")
                print(f"[DEBUG] Expanded sequence length: {total_len}")
                print(f"[DEBUG] System indices: {len(system_indices)} tokens")
                print(f"[DEBUG] Image indices: {len(image_indices)} tokens") 
                print(f"[DEBUG] Text indices: {len(text_indices)} tokens")

                modality_indices = {
                    'system': system_indices,
                    'image': image_indices,
                    'text': text_indices
                }

                ## For multi-step generation
                ## attentions is a list of attention tensors contained in tuples, one tuple per layer
                ## All grouped by generation step
                # if attentions is not None:
                #     for step, attn_tuple in enumerate(attentions):
                #         for layer, attn_tensor in enumerate(attn_tuple):
                #             print(f'Generation step {step}, Layer {layer}, Shape: {attn_tensor.shape}')

                # Only extract attention matrices for first generation step (attentions[0])
                # For binary datasets like POPE
                attn_list = attentions[0]  # Get first generation step
                for layer_idx, attn in enumerate(attn_list):
                    if layer_idx == 0 or layer_idx == len(attn_list) - 1:
                        print(f'First generation step, Layer {layer_idx}, Shape: {attn.shape}')
                        #print(f"[DEBUG-pope_eval] Layer {layer_idx+1} attn to image tokens (last query):", attn[0, :, -1, img_start_idx:img_end_idx])
                        # Process attn tensors
                        # attn: (1, heads, query_len, key_len)

                    # For each target modality
                    for to_modality in MODALITY_LABELS:
                        to_idxs = modality_indices[to_modality]
                        if not to_idxs:
                            # If target modality is empty, skip
                            continue

                        # Create empty tensor to accumulate attention weights across all sources
                        # (heads, queries) where queries is the total input sequence length
                        to_idxs_tensor = torch.tensor(to_idxs, device=attn.device)
                        
                        # For each head, get all attention directed TO this modality
                        # Note: dim 2 = query positions, dim 3 = key positions
                        # We select all query positions (from all sources) that attend to keys in our target modality
                        selected_attn = attn[0, :, :, to_idxs_tensor]  # (heads, all_queries, to_tokens)
                        
                        # Compute mean per-key attention weights for each head (average over all queries and target tokens)
                        mean_attn_weights = selected_attn.mean(dim=1).mean(dim=1)  # (heads,)

                        # No. of tokens in target modality
                        num_tokens = len(to_idxs)

                        # Compute sum of attention weights to target modality per head
                        sum_attn_weights = mean_attn_weights * num_tokens

                        # Compute entropy per head
                        head_entropies = compute_attention_entropy_per_head(selected_attn)

                        model_output = outputs
                        ground_truth = line.get('label', '')  # Use empty string if not available (e.g. COCO, Nocaps)
                        
                        # Record metrics for each head separately
                        correct_pred = normalise(model_output, args.dataset, line.get('category', '')) == normalise(ground_truth, args.dataset, line.get('category', ''))
                        for head_idx, (head_entropy, mean_attn, sum_attn) in enumerate(zip(head_entropies, mean_attn_weights, sum_attn_weights)):
                            csv_writer.writerow([
                                line["question_id"],
                                line.get('category', ''),  # Insert empty string if category not available
                                layer_idx + 1,
                                head_idx + 1,
                                to_modality,
                                head_entropy.item(),
                                mean_attn.item(),
                                sum_attn.item(), ##
                                num_tokens, ##
                                model_output,
                                ground_truth,
                                correct_pred
                            ])

            #################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--caption_file_path", type=str, default="")
    parser.add_argument("--question-file", type=str, default="question.jsonl")
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=0) ## Changed from 500 to 0 to activate logic below
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k", type=int, default=0) ## Changed from 10 to 0 to activate logic below
    parser.add_argument("--attention_head_path", type=str, default="")
    parser.add_argument("--reweight_system", action='store_true', default=False) ##
    parser.add_argument("--redistribute_to", type=str, default='img') ##
    parser.add_argument("--system_alpha", type=float, default=1.0) ##
    parser.add_argument("--random_heads", action='store_true', default=False)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--use_reduced", action="store_true", default=False, 
                    help="Use the reduced dataset if available. Defaults to the full dataset.") ##

    args = parser.parse_args()

    ## Make top_k ALL hallucination heads identified in attention_head_path.json if not specified
    with open(args.attention_head_path, 'r') as file:
        data_loaded = json.load(file)
    hal_heads_list = data_loaded['hal_heads']
    if not args.top_k or args.top_k <= 0:
        args.top_k = len(hal_heads_list)
    hal_attention_heads = hal_heads_list[:args.top_k]

    ## Dynamic file name
    reweight_parts = []

    if args.reweight_system:
        reweight_parts.append(f'sys{args.system_alpha}')
    if args.redistribute_to in ['img', 'text', 'both', 'prop']:
        reweight_parts.append(f'to_{args.redistribute_to}')

    reweight_str = '_'.join(reweight_parts) if reweight_parts else 'none'
    dataset_type = "reduced" if args.use_reduced else "full"
    answers = f'./results/{args.dataset}/llava_3000/{args.dataset}_redistr_{reweight_str}_k_{args.top_k}_heads_{args.attention_head_path.split("/")[-1].replace(".json","")}_{dataset_type}_seed{args.seed}.jsonl'

    ## If user did not specify --answers-file, use dynamic name
    if args.answers_file is None:
        args.answers_file = answers
    
    ## Ensure ans_file directory exists before writing
    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)

    ## Dynamically set num_samples to dataset length if not specified
    if args.num_samples <= 0:
        if args.dataset == 'coco':
            coco = COCO(args.caption_file_path)
            args.num_samples = len(coco.getImgIds())
        elif args.dataset in ['mme', 'pope', 'hallusionbench', 'naturalbench', 'sugarcrepe',
                              'beaf', 'whoops', 'winoground']:
            from datasets import load_dataset
            if args.use_reduced:
                # Load the reduced dataset
                reduced_datasets = {
                    'mme': "tsch00001/mme_reduced",
                    'pope': "tsch00001/pope_300_per_cat",
                    'hallusionbench': "lmms-lab/HallusionBench",  # No reduced version
                    'naturalbench': "tsch00001/naturalbench_450_per_cat",
                    'sugarcrepe': "tsch00001/sugarcrepe_130_per_cat",
                    'beaf': "tsch00001/kopie_beaf", # Reduced version first 9000 samples
                    'whoops': "nlphuji/whoops", # No reduced version
                    'winoground': "facebook/winoground" # No reduced version         
                }
                if args.dataset not in reduced_datasets:
                    raise ValueError(f"No reduced dataset available for '{args.dataset}'.")
                hf_ds = load_dataset(
                    reduced_datasets[args.dataset], 
                    split={
                        'mme': 'train',
                        'pope': 'train',
                        'hallusionbench': 'image',
                        'naturalbench': 'train',
                        'sugarcrepe': 'train',
                        'beaf': 'train',
                        'whoops': 'test',
                        'winoground': 'test'
                    }[args.dataset])
                dataset_type = "reduced"
            else:
                # Load the full dataset
                full_datasets = {
                    'mme': "lmms-lab/MME",
                    'pope': "lmms-lab/POPE",
                    'hallusionbench': "lmms-lab/HallusionBench",
                    'naturalbench': "BaiqiL/NaturalBench-lmms-eval",
                    'sugarcrepe': "yjkimstats/SUGARCREPE_fmt",
                    'beaf': "tsch00001/kopie_beaf",
                    'whoops': "nlphuji/whoops",
                    'winoground': "facebook/winoground"
                }
                hf_ds = load_dataset(full_datasets[args.dataset], split={
                    'mme': 'test',
                    'pope': 'test',
                    'hallusionbench': 'image',
                    'naturalbench': 'test',
                    'sugarcrepe': 'train',
                    'beaf': 'train',
                    'whoops': 'test',
                    'winoground': 'test'
                }[args.dataset])
                dataset_type = "full"

    set_seed(args.seed)
    eval_model(args)
