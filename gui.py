from langchain.chat_models import ChatOpenAI

from netplay import create_llm_agent
from netplay.nethack_utils.nle_wrapper import NethackGymnasiumWrapper
from netplay.logging.nethack_monitor import NethackH5PYMonitor
from netplay.core.agent_base import AgentRenderer
from netplay.nethack_agent.agent import Step

import logging
import datetime
import os
import argparse

import gradio as gr
import os
import signal

class DummyAgentRenderer(AgentRenderer):
    def init(self):
        return []
    def update(self):
        return []

class PlaytestingChatbot:
    def __init__(self, agent):
        self.agent = agent
        self.agent_renderer = self.agent.get_renderer()
        if self.agent_renderer is None:
            self.agent_renderer = DummyAgentRenderer()
        
        self.task = None
        self.thoughts = []
        self.actions = []
        self.keys = []
        self.stop = False
        self.step_execution = False
        self.execute_next_step = False
        
        # Setup
        self.agent.init()
        
        css = """
        .nowrap textarea {
            white-space: pre;
            text-overflow: ellipsis;
        }
        """

        # Define gradio interface
        with gr.Blocks(css=css) as self.interface:
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        thoughts = gr.Text(lines=5, max_lines=5, interactive=False, label="Thoughts", elem_classes="nowrap", scale=4)
                        actions = gr.Text(lines=5, max_lines=5, interactive=False, label="Actions", elem_classes="nowrap", scale=1)
                        keys = gr.Text(lines=5, max_lines=5, interactive=False, label="Keys", elem_classes="nowrap", scale=1)

                    initial_img = self.agent.env.render()
                    initial_observation = self.agent.describe_current_state()
                    env_img = gr.Image(value=initial_img, height=initial_img.shape[0], width=initial_img.shape[1], show_download_button=False, show_label=False)
                    with gr.Row():
                        step_checkbox = gr.Checkbox(label="Activate Step")
                        next_step_button = gr.Button("Step")
                        stop_button = gr.Button("Stop")
                    with gr.Row():
                        manual_actions = gr.Checkbox(label="Manual")
                        msg = gr.Textbox(scale=3)
                    env_description = gr.TextArea(value=initial_observation, interactive=False)
                with gr.Column():
                    agent_components = self.agent_renderer.init()

            step_checkbox.change(self._on_step_change, inputs=step_checkbox, queue=False)
            next_step_button.click(self._on_press_next_step, queue=False)
            stop_button.click(self.on_click_stop, queue=False)

            msg.submit(self.clear_task, inputs=[msg], outputs=[msg]).then(self._fullfill_task,
                inputs=[env_description, manual_actions],
                outputs=[env_img, env_description, thoughts, actions, keys, *agent_components]
            )

        # Handle CTRL+C while we are doing something
        self.closing = False
        self.original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        print("Keyboard Interrupt - Stopping agent")
        self.closing = True
        self.stop = True
        signal.signal(signal.SIGINT, self.original_handler)

    def on_click_stop(self):
        self.stop = True
     
    def clear_task(self, task):
        self.task = task
        return ""

    def run(self, server_port):
        self.interface.queue()
        self.interface.launch(server_port=server_port)
    
    def _on_step_change(self, value):
        self.execute_next_step = False
        self.step_execution = value

    def _on_press_next_step(self):
        self.execute_next_step = self.step_execution

    def _fullfill_task(self, env_description, use_manual_actions):
        # Prevent a loading image
        yield self.agent.env.render(), env_description, *self._render_steps(), *self.agent_renderer.update()

        if use_manual_actions:
            for step in self.agent.solve_manual_task(self.task):
                if self.stop:
                    break

                while self.step_execution:
                    if self.execute_next_step or self.stop:
                        self.execute_next_step = False
                        break
                if step.executed_action():
                    action_description = self.agent.describe_action(step.step_data.action)
                    key = repr(chr(int(step.step_data.action)))
                else:
                    action_description = ""
                    key = ""
                
                self.thoughts.append(step.thoughts if step.thoughts else "")
                self.actions.append(action_description)
                self.keys.append(key)

                observation_description = self.agent.describe_current_state()
                yield self.agent.env.render(), observation_description, *self._render_steps(), *self.agent_renderer.update()
        else:
            step: Step = None
            self.agent.set_task(self.task)
            for step in self.agent.run():
                if self.stop:
                    break

                if step.executed_action():
                    action_description = self.agent.describe_action(step.step_data.action)
                    key = repr(chr(int(step.step_data.action)))
                else:
                    action_description = ""
                    key = ""
                observation_description = self.agent.describe_current_state()
                
                self.thoughts.append(step.thoughts if step.thoughts else "")
                self.actions.append(action_description)
                self.keys.append(key)

                yield self.agent.env.render(), observation_description, *self._render_steps(), *self.agent_renderer.update()

                while self.step_execution:
                    if self.execute_next_step or self.stop:
                        self.execute_next_step = False
                        break
            
        self.stop = False
        yield self.agent.env.render(), self.agent.describe_current_state(), *self._render_steps(), *self.agent_renderer.update()

    def _render_steps(self):
        thoughts = "\n".join(reversed(self.thoughts))
        actions = "\n".join(reversed(self.actions))
        keys = "\n".join(reversed(self.keys))
        return thoughts, actions, keys

def initialize_default_logging(log_name = "log.txt"):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # File logging
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_name)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'Playtesting Chatbot',
        description='Playtests games for you.'
    )

    #parser.add_argument('-model', metavar='MODEL_NAME', type=str, default="gpt-4-1106-preview", help="Choose the OpenAI language model to use.")
    parser.add_argument('-des_file', metavar='MODEL_NAME', type=str, default=None, help="Specify an optional des-file.")
    #parser.add_argument('-des_file', metavar='MODEL_NAME', type=str, default="/workspaces/nethack_llm/scenarios/two_objects_one_spot.des", help="Specify an optional des-file.")
    parser.add_argument('-model', metavar='MODEL_NAME', type=str, default="gpt-4-1106-preview", help="Choose the OpenAI language model to use.")
    parser.add_argument('-port', metavar='SERVER_PORT', type=int, default=7861, help="Set the port number for the server that will be created.")
    parser.add_argument('-seed', metavar="SEED", type=int, help="Set the random seed to get the same results everytime.")
    parser.add_argument('-max_memory_tokens', metavar="MAX_MEMORY_TOKENS", type=int, default=500, help="Specify how many tokens the agent can store until it starts deleting old memories.")
    args = parser.parse_args()
    
    initialize_default_logging()

    log_folder = os.path.join("./runs/", datetime.datetime.now().strftime("%Y-%m-%d--%H%M%S"))
    os.makedirs(log_folder)
    
    # Initialize llm
    llm = ChatOpenAI(model=args.model, temperature=0.1, response_format={ "type": "json_object" }, max_retries=0)

    # Initialize env
    env = NethackGymnasiumWrapper(render_mode="rgb_array", des_file=args.des_file, autopickup=False)
    env = NethackH5PYMonitor(env, os.path.join(log_folder, "trajectories.h5py"))
    if args.seed:
        # Seed once after that any reset with seed=None will be deterministic
        env.reset(seed=args.seed)

    # Initialize agent
    agent = create_llm_agent(
        env=env,
        llm=llm,
        memory_tokens=args.max_memory_tokens,
        log_folder=log_folder,
        update_hidden_objects=True
    )
    
    # Create and run GUI
    chatbot = PlaytestingChatbot(agent)
    chatbot.run(args.port)

    agent.close()
