import argparse
import random
from functools import lru_cache

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from utils import open_config, create_model
from detector.attn import AttentionDetector

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    set_seed(args.seed)

    model_config_path = f"./configs/model_configs/{args.model_name}_config.json"
    model_config = open_config(config_path=model_config_path)
    model_id = model_config["model_info"]["model_id"]

    model = create_model(config=model_config)
    model.print_model_info()

    templates_and_rules = pd.read_csv("templates_and_rules.csv", na_values=[], keep_default_na=False)
    template_rows_by_rule = {
        row["Rule"]: row for _, row in templates_and_rules.iterrows()
    }
    attacks = pd.read_csv("attacks.csv")
    attacks["Template"] = attacks["Template"].apply(lambda s: s.replace("\\n", "\n"))
    attacks_by_name = {
        row["Name"]: row["Template"] for _, row in attacks.iterrows()
    }

    df = pd.read_parquet(args.df)
    if "attention_tracker" not in df:
        df["attention_tracker"] = 0.0
    df_slice = df[df["model"] == model_id]

    detector = AttentionDetector(model)

    @lru_cache(maxsize=100)
    def focus_score(instruction_, prompt_):
        _, _, attention_maps, _, input_range, _ = model.inference(
            instruction_, prompt_, max_output_tokens=1
        )
        return detector.attn2score(attention_maps, input_range)

    for index, data in tqdm(df_slice.iterrows(), total=len(df_slice)):
        template_row = template_rows_by_rule[data["rule"]]
        instruction = template_row["System Template"].format(rule=data["rule"], refusal="Unable")

        prompt = template_row["User Template"].format(user=data["prompt"])
        attack = attacks_by_name.get(data["attack"], "{prompt}")
        formatted_attack = attack.format(prompt=prompt, refusal="Unable")

        df.loc[index, "attention_tracker"] = focus_score(instruction, formatted_attack)

    df.to_parquet(args.df, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt Injection Detection Script")
    
    parser.add_argument("--model_name", type=str, default="qwen2-attn",
                        help="Path to the model configuration file.")
    parser.add_argument("--df", type=str, default="prompt_injections.parquet",
                        help="Path to the dataframe.")
    parser.add_argument("--seed", type=int, default=0)
    
    args_ = parser.parse_args()

    main(args_)