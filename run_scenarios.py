import os
import subprocess
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from warnings import warn
from typing import List

# Randomly generated, idk what random generator NetHack uses so this is to avoid correlation between seeds
DEFAULT_SEEDS = [
    779726,
    474862,
    151437,
    10518,
    380261
]

RUN_PY = "/workspaces/nethack_llm/run.py"
SCENARIO_FOLDER = "/workspaces/nethack_llm/scenarios"
LOG_FOLDER = "/workspaces/nethack_llm/runs/scenarios"

@dataclass
class ScenarioConfig:
    name: str
    des_file: str
    task: str
    additional_args: List[str] = field(default_factory=lambda: [])

SCENARIO_CONFIGS = [
    ScenarioConfig(
        name="alternative",
        des_file="instructions/alternative.des",
        task="Drink from a fountain or drink a potion.",
        additional_args=["-max_steps", "200"]
    ),
    ScenarioConfig(
        name="conditional",
        des_file="instructions/conditional.des",
        task="Drink from a fountain, if there is no fountain drink a potion.",
        additional_args=["-max_steps", "200"]
    ),
    ScenarioConfig(
        name="ordered",
        des_file="instructions/ordered.des",
        task="Pickup both wands, then the scroll, finally use the scroll to identify the wands.",
        additional_args=["-max_steps", "200"]
    ),
    ScenarioConfig(
        name="unordered", 
        des_file="instructions/unordered.des",
        task="Fulfill all your tasks in any order: drink from the fountain, open both doors, kill the grid bug.",
        additional_args=["-max_steps", "200"]
    ),
    ScenarioConfig(
        name="bag_of_holding", 
        des_file="game_mechanics/bag_of_holding.des",
        task="Stuff all objects in this room into a bag.",
        additional_args=["-max_steps", "200"]
    ),
    ScenarioConfig(
        name="guided_bag_of_holding",
        des_file="game_mechanics/bag_of_holding.des",
        task="Stuff all objects in this room into a bag. Only use 'pickup x y' to pickup items, without move_to. Use Auto-Select every type when using the bag.",
        additional_args=["-max_steps", "200"]
    ),
    ScenarioConfig(
        name="multipickup",
        des_file="game_mechanics/multipickup.des",
        task="Pickup all objects in the current room.",
        additional_args=["-max_steps", "200"]
    ),
    ScenarioConfig(
        name="wand",
        des_file="game_mechanics/wand.des",
        task="Hit the statue with a wand.",
        additional_args=["-max_steps", "200"]
    ),
    ScenarioConfig(
        name="guided_wand",
        des_file="game_mechanics/wand.des",
        task="Hit the statue with a wand, which will usually cause your spell to bounce. Make sure to stand right next to the statue, not on top of it. Also make sure to fire in the right direction.",
        additional_args=["-max_steps", "200"]
    ),
    # Seems like my tip doesn't even work most of the time (just results in game message "pardon me, <shopkeeper_name>" but no movement), idk how to resolve this
    #ScenarioConfig(
    #    name="shop", 
    #    des_file="game_mechanics/shop.des",
    #    task="Buy something from a shop. To move onto a tile occupied by the shopkeeper next to him and press 'm' followed by a direction. Use the keys 'ykulnjbh' for specifying the directions 'n ne e se s sw w nw'. For example use 'type_text my' to move north without attacking.",
    #    additional_args=["-max_steps", "200"]
    #),
    ScenarioConfig(
        name="carry", 
        des_file="creativity/carry.des",
        task="Drop a plate mail and a iron ball in the room containing the statue.",
        additional_args=["-max_steps", "500"]
    ),
    ScenarioConfig(
        name="guided_carry", 
        des_file="creativity/carry.des",
        task="Drop a plate mail and a iron ball in the room containing the statue. Make sure to first defeat any monster that you encounter. The items are very heavy, only pickup one at a time and use a teleportation wand to teleport yourself with the item.",
        additional_args=["-max_steps", "500"]
    ),
    ScenarioConfig(
        name="boulder", 
        des_file="creativity/boulder.des",
        task="Find a new room.",
        additional_args=["-max_steps", "500"]
    ),
    ScenarioConfig(
        name="focused_boulder", 
        des_file="creativity/boulder.des",
        task="Find a new room. If a boulder blocks your path, find a way to remove it.",
        additional_args=["-max_steps", "500"]
    ),
    ScenarioConfig(
        name="guided_boulder", 
        des_file="creativity/boulder.des",
        task="Find a new room. If a boulder blocks you path remove it using teleportation wands, striking wands, or pickaxes. Note to hit with the wands you have to stand next to the boulder. To use pickaxes you can apply them and specify the direction. (h=west, l=east).",
        additional_args=["-max_steps", "500"]
    ),
    ScenarioConfig(
        name="escape", 
        des_file="creativity/escape.des",
        task="Escape room 0 without using a staircase.",
        additional_args=["-max_steps", "500"]
    ),
    ScenarioConfig(
        name="hint_escape", 
        des_file="creativity/hint_escape.des",
        task="Escape room 0 without using a staircase.", #  If you want to find a wall, move using 'type_text 3l' to move 3 times east or 'type_text 3h' to move 3 times west.
        additional_args=["-max_steps", "500"]
    ),
]
SCENARIO_CONFIGS = {config.name : config for config in SCENARIO_CONFIGS}

def run_scenario(config: ScenarioConfig, seed: int, render: bool):
    print(f"Running scenario '{config.name}' with seed {seed}.")
    try:
        log_folder = os.path.join(LOG_FOLDER, Path(config.des_file).parent.name, config.name, str(seed))
        os.makedirs(log_folder, exist_ok=True)

        process_args = [
            "llm",
            "-task", config.task,
            "-log_folder", log_folder,
            "-des_file", os.path.join(SCENARIO_FOLDER, config.des_file),
            "-seed", str(seed),
            "-model", "gpt-4-1106-preview",
            "--censor_nethack_context",
            "--keep_log_folder",
            "--update_hidden_objects",
            *config.additional_args
        ]
        if render:
            process_args.append("--render")

        with open(os.path.join(log_folder, "out.txt"), "w") as out_file:
            subprocess.run(["python", RUN_PY, *process_args], stdout=out_file)
    except Exception as e:
        warn(f"Scenario '{config.name}' with seed {seed} failed:\n{e}")

def run_scenario_seeds(config: ScenarioConfig, seeds: List[int], render: bool):
    for seed in seeds:
        run_scenario(config=config, seed=seed, render=render)

if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser(
        prog = 'Runs the LLM agent on scenarios in the scenarios folder.'
    )
    parser.add_argument('-scenario_name', default=None, help="Specify a scenario to run, otherwise all scenarios will be executed in order.")
    parser.add_argument('-seeds', nargs="+", type=int, default=DEFAULT_SEEDS, help="Specify a list of integer seeds for each run, defining the randomization for each scenario. Also determines often a scenario will be repeated.")
    parser.add_argument('--render', action='store_true', help="Renders each run in an window.")
    args = parser.parse_args()

    if args.scenario_name:
        config = SCENARIO_CONFIGS.get(args.scenario_name)
        if not config:
            print(f"Scenario {args.scenario_name} does not exist. Available scenarios:")
            print("\n".join([f"- {name}" for name in SCENARIO_CONFIGS.keys()]))
            exit()

        run_scenario_seeds(config=config, seeds=args.seeds, render=args.render)
    else:
        for config in SCENARIO_CONFIGS.values():
            run_scenario_seeds(config=config, seeds=args.seeds, render=args.render)