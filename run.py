from netplay import create_llm_agent
from netplay.core.agent_base import NethackBaseAgent
from netplay.nethack_agent.agent import NetHackAgent
from netplay.handcrafted_agent.agent import HandcraftedAgent
from netplay.nethack_utils.nle_wrapper import NethackGymnasiumWrapper
from netplay.logging.nethack_monitor import NethackH5PYMonitor
from netplay.nethack_agent.describe import describe_glyph

from nle_language_wrapper import NLELanguageWrapper

from langchain.chat_models import ChatOpenAI

from termcolor import colored
import argparse
import os
import datetime
import time

def run_agent(agent: NethackBaseAgent, args):
    agent.init()

    # Print the role
    role_glyph = agent.data.current_level.glyphs[agent.blstats.y,agent.blstats.x]
    print(f"Agent is playing as a {describe_glyph(role_glyph)}.")

    nethack_agent: NetHackAgent = agent # Ugly but useful for type hinting
    if args.task:
        nethack_agent.set_task(args.task)

    while True:
        if args.interactive and args.agent == "llm" and nethack_agent.task is None:
            task = input("Input task: ")
            nethack_agent.set_task(task)

        try:
            for i, step in enumerate(agent.run()):
                if step.executed_action():
                    action_description = NLELanguageWrapper.all_nle_action_map[step.step_data.action][0]
                    parts = [f"Executed action '{action_description}'.", f"Thoughts: {step.thoughts}" if step.thoughts else None]
                    print(" ".join([p for p in parts if p]))
                elif step.thoughts:
                    print(colored(f"Thinking: {step.thoughts}", "blue"))

                if args.max_steps != -1 and agent.timestep >= args.max_steps:
                    print("Reached maximum steps. Aborting...")
                    break

                if args.render:
                    time.sleep(0.1)

            if not args.interactive:
                print("Agent is done.")
                break
        except KeyboardInterrupt:
            if args.interactive:
                print(f"Aborted task, provide a new task or press CTRL+C again to end the program.")
            else:
                print(f"Aborting run.")
                break

if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser(
        prog = 'Run NetHack agent',
        description='Runs the specified NetHack agent.'
    )
    parser.add_argument('agent', type=str, default="handcrafted", help="Choose which agent to run (handcrafted or llm).")
    parser.add_argument('-task', type=str, default=None, help="The task that will be passed to the agent at the beginning of the run.")
    parser.add_argument('-des_file', type=str, default=None, help="Specify an optional des-file.")
    parser.add_argument('-character', type=str, default="@", help="Specify which character the agent will spawn with. See https://nethackwiki.com/wiki/Options#role")
    parser.add_argument('-max_steps', type=int, default=-1, help="The maximum amount of steps until the run will be aborted. Default=-1 indicating no timelimit.")
    parser.add_argument('-seed', type=int, help="The random seed used for this run.")
    parser.add_argument('-log_folder', type=str, default="./runs", help="Folder for storing the run log. Note each run will create a new subfolder in the log folder.")
    parser.add_argument('-model', type=str, default="gpt-4-1106-preview", help="Choose the OpenAI language model to use. Only used for the llm agent.")
    parser.add_argument('-max_memory_tokens', type=int, default=500, help="Specify number of tokens the agents memory can hold. Only used for the llm agent.")
    parser.add_argument('--censor_nethack_context', action='store_true', help="Censors any mentions of the word 'NetHack' before passing prompts to the LLM.")
    parser.add_argument('--disable_finish_task_skill', action='store_true', help="Disables the ability of the LLM to finish tasks on its own. ONLY disable this flag for tasks that focus on ending the game.")
    parser.add_argument('--update_hidden_objects', action='store_true', help="Enable to fix a bug where removed objects would still show up in the environment description.")
    parser.add_argument('--interactive', default=False, action='store_true', help="Allows to pass tasks to the agent and to abort active tasks using CTRL+C.")
    parser.add_argument('--render', default=False, action='store_true', help="Render the game in an window. Always active when --interactive is set.")
    parser.add_argument('--keep_log_folder', default=False, action='store_true', help="Set to ensure the passed log folder is not modified.")
    args = parser.parse_args()

    # Arg processing
    print(f"Using {args.agent} agent.")
    if args.max_steps > 0:
        print(f"Limiting agent to {args.max_steps} steps")
    if args.agent == "handcrafted":
        if args.interactive:
            print("Ignoring --interactive due to using the handcrafted agent.")
            args.interactive = False
        if args.task:
            print("Ignoring task due to using the handcrafted agent.")
            args.task = None
    if args.interactive and not args.render:
        print("The flag --interactive was set, setting --render as well.")
        args.render = True
    if args.interactive and args.max_steps != -1:
        print("The flag --interactive was set, ignoring max_steps.")
        args.max_steps = -1

    # Setup logging
    log_folder = args.log_folder
    if not args.keep_log_folder:
        log_folder = os.path.join(log_folder, datetime.datetime.now().strftime("%Y-%m-%d--%H%M%S"))
    os.makedirs(log_folder, exist_ok=args.keep_log_folder)
    print(f"Logging in '{log_folder}'.")

    # Setup environment
    env = NethackGymnasiumWrapper(render_mode="human", des_file=args.des_file, autopickup=False, character=args.character)
    env = NethackH5PYMonitor(env, os.path.join(log_folder, "trajectories.h5py"))
    if args.seed:
        # Seed once after that any reset with seed=None will be deterministic
        env.reset(seed=args.seed)
    
    # Init agent
    if args.agent == "llm":
        response_format = { "type": "json_object" } if args.model == "gpt-4-1106-preview" else None
        llm = ChatOpenAI(model=args.model, temperature=0, response_format=response_format, max_retries=0)
        agent = create_llm_agent(
            env=env,
            llm=llm,
            memory_tokens=args.max_memory_tokens,
            log_folder=log_folder,
            render=args.render,
            censor_nethack_context=args.censor_nethack_context,
            enable_finish_task_skill=not args.disable_finish_task_skill,
            update_hidden_objects=args.update_hidden_objects
        )
    elif args.agent == "handcrafted":
        agent = HandcraftedAgent(
            env=env,
            log_folder=log_folder,
            render=args.render
        )
    else:
        print("No valid agent was specified. Use 'run llm' or 'run handcrafted'")
        exit(0)

    # Run
    try:
        run_agent(agent, args)
    except KeyboardInterrupt as e:
        pass

    agent.close()