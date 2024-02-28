from netplay.logging.agent_log_dataset import AgentLogDataset
from netplay.logging.video_renderer import AgentVideoRenderer

import os
import argparse
import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'Generate Video',
        description='Generates a video for the given run.'
    )
    parser.add_argument('log_folder', type=str, help="Path to the log folder.")
    args = parser.parse_args()

    dataset = AgentLogDataset.from_log_folder(args.log_folder)

    video_writer = AgentVideoRenderer(os.path.join(args.log_folder, "video.mp4"))
    for step in tqdm.tqdm(dataset, desc="Generating video..."):
        video_writer.add_step(step.observation, step.action, thoughts="")
    video_writer.add_step(step.next_observation, None, thoughts="")
    video_writer.close()

    print("Done!")