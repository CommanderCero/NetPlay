import glob
import os
import tqdm
import h5py
import pickle
import numpy as np

from collections import Counter
from InstructorEmbedding import INSTRUCTOR
from nle_language_wrapper.nle_language_obsv import NLELanguageObsv

# Parameters
agent_name = "autoascend"
runs_folder = "/workspaces/nethack_llm/experiments/runs/autoascend/valkyrie"

nle_lang = NLELanguageObsv()
model = INSTRUCTOR('hkunlp/instructor-xl')

# Load all messages
trajectory_files = glob.glob(os.path.join(runs_folder, "*", "trajectories.h5py"))
print(f"Found {len(trajectory_files)} trajectories")
assert len(trajectory_files) != 0

messages = []
for path in tqdm.tqdm(trajectory_files, desc="Loading data..."):
    try:
        with h5py.File(path) as file:
            trajectory_groups = file["trajectories"]
            assert len(trajectory_groups) == 1, "Expected only one trajectory per run"
            trajectory = trajectory_groups["0"]       

            tty_chars = trajectory["observations"]["tty_chars"]
            for i in range(tty_chars.shape[0]):
                message = nle_lang.text_message(tty_chars[i]).decode("latin-1")
                if message:
                    messages.append(message)
            
    except:
        print(f"Failed to read {path}")

message_counts = Counter(messages)
instruction = "Represent the NetHack event:"
sentences = [[instruction, message] for message in message_counts.keys()]
embeddings = model.encode(sentences=sentences, show_progress_bar=True)
embeddings = [
    {"message": message, "count": count, "embedding": embedding}
    for message, count, embedding in zip(message_counts.items(), message_counts.values(), embeddings)
]

with open(os.path.join(runs_folder, "message_embeddings.pkl"), "wb") as f:
    pickle.dump(embeddings, f)