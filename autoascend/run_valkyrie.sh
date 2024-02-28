ray start --head
python main.py simulate -n 1000 --role val --seed 0 --log_folder "./runs/valkyrie" --no-plot
ray stop