# NetPlay - A LLM powered agent for NetHack
https://github.com/CommanderCero/NetPlay/assets/10519507/b920116a-a288-4165-9c30-548a5ce0f1bc


This repo contains the source code for NetPlay, the first LLM-powered
zero-shot agent for the extremely challenging roguelike NetHack. 

## Installation
NetPlay was tested using VSCode, the devcontainer plugin, and Python 3.8.5. We recommend running NetPlay using the provided Dockerfile as it installs all the required dependencies. Alternatively, we provide a setup.py to install the project using pip. Note that some files, such as [run_scenarios.py](run_scenarios.py), use hardcoded paths and must be adapted if the project is installed in any other way.

```
git clone https://github.com/CommanderCero/NetPlay
cd netplay
pip install -e .
```

Netplay requires the environment variable OPENAI_API_KEY to be set to a valid key for the [OpenAI API](https://openai.com/blog/openai-api). Using the devcontainer plugin, this can be done by adding a devcontainer.env file to the .devcontainer folder. **Other LLMs** can be used by modifying the [run.py](run.py). However, we had no success when using any LLM besides GPT-4. If you manage to get it to work, let us know!


## Interactive
https://github.com/CommanderCero/NetPlay/assets/10519507/9198c08b-c799-47dd-8d2e-60f7c5350b62

NetPlay can be run in interactive mode, where you can continually pass new tasks to the agent. Note that although possible, we do not support guiding the agent in real-time. Instead, you have to interrupt a task to provide feedback. 

#### Run
The recommended approach is to run NetPlay in interactive mode using the [run.py](run.py).
```
python run.py --interactive --render --update_hidden_objects
```
This will open a window where you can observe what the agent is doing while you pass tasks to the agent using the command line. You can also interrupt a task using CTRL+C. If you want more control or to run skills manually, use the developer GUI described below.

#### Developer GUI
[gui.py](gui.py) launches a [Gradio](https://www.gradio.app/) app, which can then be accessed in your browser by navigating to http://127.0.0.1:7861. The app allows you to run skills manually "move_to x y" or to pass a task to the LLM "Pickup an item." The app also allows you to execute skills step-by-step and interrupt them whenever possible. **Note**: The Gradio app was primarily designed for development purposes and should be used carefully. The app also does not support multiple connections, so do not open it multiple times.




## Full Runs
NetPlay can also play the game fully autonomously using the following command.
```
python run.py llm -task "Win the game" --disable_finish_task_skill --update_hidden_objects --render
```
The *disable_finish_task_skill* flag will prevent the agent from declaring the task as finished prematurely. By setting this flag, the agent will only stop playing once the game has ended.

## Scenarios
NetHack uses [description files](https://nethackwiki.com/wiki/Des-file_format) **(des-files)** to define special levels, such as the [oracle level](https://nethackwiki.com/wiki/The_Oracle). We provide a set of predefined scenarios consisting of a des-file plus an instruction to test NetPlays abilities in isolation. To run all scenarios, use the following command:
```
python run_scenarios.py --render
```
This command will run each scenario with a different seed five times. Alternatively, use the following command to run a single scenario for a given set of seeds:
```
python run_scenarios.py -scenario_name ordered -seeds 1,2,3,4,5 --render
```
Finally, to run your own scenarios, you can use [run.py](run.py):
```
python run.py -task "<Your Task>" -des_file "<Path to des-file>" --update_hidden_objects --render
```

## Experiments
Recordings of NetPlay, autoascend, and the handcrafted agent are available on [Google Drive](https://drive.google.com/file/d/1Lkidie9UTlTm8bpfaHYIO4dxsA53Iofs/view?usp=sharing). The [experiments folder](experiments) contains the code used to analyze the recordings and the results reported in our paper.

The files [run_guided_agent.sh](experiments\run_guided_agent.sh), [run_unguided_agent.sh](experiments\run_guided_agent.sh), and [run_handcrafted_agent.sh](experiments\run_handcrafted_agent.sh) contain the command used to run the corresponding agent. This repo also contains a copy of autoascend, modified to record its playthroughs similar to ours. To get autoascend working, follow the instructions in its [README](autoascend\README.md). The [run_valkyrie.sh](autoascend\run_valkyrie.sh) contains the command we used to run autoascend.

## Citation
```bibtex
@INPROCEEDINGS{jeurissen2024playing,
  author={Jeurissen, Dominik and Perez-Liebana, Diego and Gow, Jeremy and Cakmak, Duygu and Kwan, James},
  booktitle={2024 IEEE Conference on Games (CoG)}, 
  title={Playing NetHack with LLMs: Potential \& Limitations as Zero-Shot Agents}, 
  year={2024},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/CoG60054.2024.10645630}
}
```
