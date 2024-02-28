from minihack.tiles.rendering import get_des_file_rendering
def render_des_file(des_file, **kwargs):
    image = get_des_file_rendering(des_file, **kwargs)
    image.save("test.png")

render_des_file("/workspaces/nethack_llm/scenarios/instructions/unordered.des")