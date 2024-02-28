import numpy as np

import sys
import os
import subprocess
import argparse
from warnings import warn

ALLOWED_ROLES = [
    "archeologist",
    "barbarian",
    "cavewoman",
    "caveman",
    "healer",
    "knight",
    "monk",
    "priest",
    "priestess",
    "ranger",
    "rogue",
    "samurai",
    "tourist",
    "valkyrie",
    "wizard"
]

AGENT_GUIDE = """Your task is to win the game. Tips:
- Stay on the first 2 levels (depth=1/2, dungeon number=0) until you reached level 8.
- You can kick open locked doors, but that usually requires multiple tries. Note WHAMM means its still closed.
- When your health gets low, run away before healing.
- Corpses in your inventory will rot, do not eat them. Only eat freshly killed corpses. Keep eating corpses until satiated, but do not keep eating.
- Do not eat kobolds, they are poisonous.
- Only pray every 500 turns to fix bad situations, for example low health or when you are fainting.
- Use the keys "ykulnjbh" for specifying the directions "n ne e se s sw w nw". For example press the key "y" to indicate north.
- The skill "type_text" can be used to press multiple alphanumeric keys, use it to navigate menus faster.
- To move onto a tile occupied by a friendly monster stand next to it and press "m". Use this for shopkeepers.
- Dip long swords into fountains as a lawful character at level 5 or higher to get excalibur.
- Sacrifice corpses at altars for a chance to get powerful artifacts.
- You will timeout if you do not make progress for a while. So if something doesn't work on the first try, shift your focus onto something else."""

CREATIVE_AGENT_GUIDE = """You are a creative agent, that is interested in exploring the game as much as possible."
Your goals are to find interesting interactions and to discover new areas to explore.
You despise dying, as it will reset your progress.
Here are some survival tips:
- You can kick open locked doors, but that usually requires multiple tries. Note WHAMM means its still closed.
- When your health gets low, run away before healing.
- Corpses in your inventory will rot, do not eat them. Only eat freshly killed corpses. Keep eating corpses until satiated, but avoid overeating.
- Do not eat kobolds, they are poisonous.
- Only pray every 500 turns to fix bad situations, for example low health or when you are fainting.
- The keys "ykulnjbh" correspond to the directions "n ne e se s sw w nw".
- The skill "type_text" can be used to press multiple alphanumeric keys, use it to navigate menus faster.
- To move onto a tile occupied by a friendly monster stand next to it and press "m". Use this for shopkeepers.
- You will timeout if you do not make progress for a while. So if something doesn't work on the first try, shift your focus onto something else.
"""

RUN_PY = "/workspaces/nethack_llm/run.py"
LOG_FOLDER = "/workspaces/nethack_llm/runs"

if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser(
        prog = 'Runs the LLM agent on NetHack with a specific role.'
    )
    parser.add_argument('agent', type=str, default="llm", help="Choose which agent to run (handcrafted or llm).")
    parser.add_argument('-agent_name', type=str, default="agent", help="A name for the agent for logging purposes.")
    parser.add_argument('-seeds', nargs="+", type=int, default=None, help="Specify a list of integer seeds for each run.")
    parser.add_argument('-num_runs', nargs="+", type=int, default=30, help="Specify how many runs to run. Will be ignored if -seeds is set.")
    parser.add_argument('-role', type=str, default="valkyrie", choices=ALLOWED_ROLES, help="The role the agent will use.")
    parser.add_argument('-model', type=str, default="gpt-4-1106-preview", help="Choose the OpenAI language model to use.")
    parser.add_argument('--censor_nethack_context', action='store_true', help="Censors any mentions of the word 'NetHack' before passing prompts to the LLM.")
    parser.add_argument('--render', default=False, action='store_true', help="Renders each run in an window.")
    parser.add_argument('--use_guide', default=False, action='store_true', help="When set the agent gets access to a guide for playing the game.")
    parser.add_argument('--use_creative_guide', default=False, action='store_true', help="When set the agent will be tasked to act creative.")
    parser.add_argument('--update_hidden_objects', action='store_true', help="Enable to fix a bug where removed objects would still show up in the environment description.")
    args = parser.parse_args()
    if args.seeds:
        print("-seeds was set, ignoring -num_runs.")
        seeds = args.seeds
        args.num_runs = len(args.seeds)
    else:
        seeds = np.random.randint(1000000, size=args.num_runs)

    if args.role.lower() not in ALLOWED_ROLES:
        print(f"The role {args.role} is not valid. Make sure to write the full role name and lookout for typos.")
        exit()

    if args.use_guide and args.use_creative_guide:
        print("Both flags --use_guide and --use_creative_guide have been set. Only use one.")
        exit()

    for seed in seeds:
        log_folder = os.path.join(LOG_FOLDER, args.agent_name, args.role, str(seed))
        if os.path.exists(log_folder):
            print(f"Skipping seed {seed} because the folder {log_folder} already exists. Delete it to re-run this seed.")
            continue

        print(f"Running {args.agent_name} with seed {seed}.")
        try:
            os.makedirs(log_folder, exist_ok=True) 

            process_args = [
                args.agent,
                "-task", AGENT_GUIDE if args.use_guide else CREATIVE_AGENT_GUIDE if args.use_creative_guide else "Win the game.",
                "-log_folder", log_folder,
                "-seed", str(seed),
                "-character", args.role,
                "-model", args.model,
                "--keep_log_folder",
                "--disable_finish_task_skill"
            ]
            if args.censor_nethack_context:
                process_args.append("--censor_nethack_context")
            if args.update_hidden_objects:
                process_args.append("--update_hidden_objects")
            if args.render:
                process_args.append("--render")

            with open(os.path.join(log_folder, "out.txt"), "w") as out_file:
                subprocess.run(["python", RUN_PY, *process_args], stdout=out_file)
        except Exception as e:
            warn(f"Running {args.agent_name} with seed {seed} failed:\n{e}")
    