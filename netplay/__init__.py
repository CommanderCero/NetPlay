import netplay.nethack_agent.skills as skills
import netplay.nethack_agent.descriptors as descriptors
import netplay.nethack_agent.skill_selection as skill_selection
from netplay.nethack_agent.agent import NetHackAgent
from netplay.core.skill_repository import SkillRepository
from netplay.core.descriptor import TitleValueDescriptor

def create_llm_agent(env, llm, memory_tokens, log_folder, render=False, censor_nethack_context=False, enable_finish_task_skill=True, update_hidden_objects=False):
    skill_repo = SkillRepository([
        *skills.ALL_COMMAND_SKILLS,
        skills.set_avoid_monster_flag,
        skills.melee_attack,
        skills.explore_level,
        skills.move_to,
        skills.go_to,
        skills.press_key,
        skills.type_text,
    ])
    state_descriptor = TitleValueDescriptor({
        "Context": descriptors.GeneralContextDescriptor() if censor_nethack_context else descriptors.NetHackContextDescriptor(),
        "Agent Information": descriptors.AgentInformationDescriptor(),
        "Rooms": descriptors.RoomsObjectFeatureDescriptor(),
        "Close Monsters": descriptors.CloseMonsterDescriptor(),
        "Distant Monsters": descriptors.DistantMonsterDescriptor(),
        #"Current Room": descriptors.CurrentRoomDescriptor(),
        #"Other Rooms": descriptors.OtherRoomsDescriptor(),
        "Exploration Status": descriptors.ExplorationStatusDescriptor(),
        "Inventory": descriptors.InventoryDescriptor(),
        "Stats": descriptors.StatsDescriptor(),
        "Task": descriptors.TaskDescriptor()
    })
    skill_selector = skill_selection.SimpleSkillSelector(
        llm=llm,
        skills=skill_repo,
        use_popup_prompt=True
    )
    agent = NetHackAgent(
        env=env,
        state_descriptor=state_descriptor,
        skill_selector=skill_selector,
        llm=llm,
        skills=skill_repo,
        max_memory_tokens=memory_tokens,
        log_folder=log_folder,
        render=render,
        censor_nethack_messages=censor_nethack_context,
        enable_finish_task_skill=enable_finish_task_skill,
        update_hidden_objects=update_hidden_objects
    )
    return agent