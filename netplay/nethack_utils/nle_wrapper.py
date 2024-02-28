import sys
import numpy as np
import enum

from PIL import Image, ImageDraw, ImageFont
from gymnasium import Env

import nle.nethack as nh
from nle.env.base import NLE, nethack
from minihack.base import MiniHack

class RawKeyPress(enum.IntEnum):
    """
    Enumeration class (hopefully) representing all key presses used in NetHack.
    Includes uppercase letters, many symbols, numbers, and keys like Space and Escape.

    Usage:
        keypress = RawKeyPress.parse("ESC")
        keypress = RawKeyPress.parse("A")
    """
    # Uppercase Letters
    KEYPRESS_A = ord("A")
    KEYPRESS_B = ord("B")
    KEYPRESS_C = ord("C")
    KEYPRESS_D = ord("D")
    KEYPRESS_E = ord("E")
    KEYPRESS_F = ord("F")
    KEYPRESS_G = ord("G")
    KEYPRESS_H = ord("H")
    KEYPRESS_I = ord("I")
    KEYPRESS_J = ord("J")
    KEYPRESS_K = ord("K")
    KEYPRESS_L = ord("L")
    KEYPRESS_M = ord("M")
    KEYPRESS_N = ord("N")
    KEYPRESS_O = ord("O")
    KEYPRESS_P = ord("P")
    KEYPRESS_Q = ord("Q")
    KEYPRESS_R = ord("R")
    KEYPRESS_S = ord("S")
    KEYPRESS_T = ord("T")
    KEYPRESS_U = ord("U")
    KEYPRESS_V = ord("V")
    KEYPRESS_W = ord("W")
    KEYPRESS_X = ord("X")
    KEYPRESS_Y = ord("Y")
    KEYPRESS_Z = ord("Z")

    # Lowercase Letters
    KEYPRESS_a = ord("a")
    KEYPRESS_b = ord("b")
    KEYPRESS_c = ord("c")
    KEYPRESS_d = ord("d")
    KEYPRESS_e = ord("e")
    KEYPRESS_f = ord("f")
    KEYPRESS_g = ord("g")
    KEYPRESS_h = ord("h")
    KEYPRESS_i = ord("i")
    KEYPRESS_j = ord("j")
    KEYPRESS_k = ord("k")
    KEYPRESS_l = ord("l")
    KEYPRESS_m = ord("m")
    KEYPRESS_n = ord("n")
    KEYPRESS_o = ord("o")
    KEYPRESS_p = ord("p")
    KEYPRESS_q = ord("q")
    KEYPRESS_r = ord("r")
    KEYPRESS_s = ord("s")
    KEYPRESS_t = ord("t")
    KEYPRESS_u = ord("u")
    KEYPRESS_v = ord("v")
    KEYPRESS_w = ord("w")
    KEYPRESS_x = ord("x")
    KEYPRESS_y = ord("y")
    KEYPRESS_z = ord("z")

    # Numbers Letters
    KEYPRESS_0 = ord("0")
    KEYPRESS_1 = ord("1")
    KEYPRESS_2 = ord("2")
    KEYPRESS_3 = ord("3")
    KEYPRESS_4 = ord("4")
    KEYPRESS_5 = ord("5")
    KEYPRESS_6 = ord("6")
    KEYPRESS_7 = ord("7")
    KEYPRESS_8 = ord("8")
    KEYPRESS_9 = ord("9")

    # Symbols
    KEYPRESS_NUMBER_SIGN = ord("#")
    KEYPRESS_SEMICOLON = ord(";")
    KEYPRESS_DOUBLECOLON = ord(":")
    KEYPRESS_DOT = ord(".")
    KEYPRESS_COMMA = ord(",")
    KEYPRESS_SMALLER = ord("<")
    KEYPRESS_GREATER = ord(">")
    KEYPRESS_FORWARD_SLASH = ord("/")
    KEYPRESS_BACKWARD_SLASH = ord("\\")
    KEYPRESS_CARET = ord("^")
    KEYPRESS_OPEN_BRACKET = ord("(")
    KEYPRESS_CLOSE_BRACKET = ord(")")
    KEYPRESS_OPEN_SQUARE_BRACKET = ord("[")
    KEYPRESS_EQUALS = ord("=")
    KEYPRESS_STAR = ord("*")
    KEYPRESS_DOLLAR = ord("$")
    KEYPRESS_PLUS = ord("+")
    KEYPRESS_AT_SIGN = ord("@")
    KEYPRESS_QUESTION_MARK = ord("?")
    KEYPRESS_AMPERSAND = ord("&")
    KEYPRESS_EXCLAMATION_MARK = ord("!")
    KEYPRESS_UNDERSCORE = ord("_")
    KEYPRESS_DOUBLE_QUOTATION_MARK = ord("\"")
    KEYPRESS_BACKTICK = ord("`")
    KEYPRESS_PERCENT = ord("%")

    # Special Keys
    KEYPRESS_ENTER = 13
    KEYPRESS_ESC = 27
    KEYPRESS_SPACE = 32

    @staticmethod
    def parse(key: str) -> "RawKeyPress":
        special_keys = {
            "enter": RawKeyPress.KEYPRESS_ENTER,
            "space": RawKeyPress.KEYPRESS_SPACE,
            "esc": RawKeyPress.KEYPRESS_ESC
        }

        if len(key) == 1:
            return RawKeyPress(ord(key))
        elif key.lower() in special_keys:
            return special_keys[key.lower()]
        
        raise ValueError(f"Cannot parse the given key {key}.")

def render_ascii_map(map: np.array, font: ImageFont, is_ascii_map=False) -> np.array:
    if is_ascii_map:
        lines = ["".join([chr(c) for c in line]) for line in map]
    else:
        lines = ["".join([str(c) for c in line]) for line in map]
    full_text = "\n".join(lines)

    # Determine the text bounding box
    dummy_img = Image.new('RGB', (1, 1))
    left, top, width, height = ImageDraw.Draw(dummy_img).multiline_textbbox((0,0), full_text, font=font)

    # Render the image
    image = Image.new('RGB', (width, height), color='black')
    canvas = ImageDraw.Draw(image)
    canvas.multiline_text((0,0), full_text, font=font, fill="white")

    return np.array(image)

class NethackGymnasiumWrapper(Env):
    """
    NetHack Gymnasium Wrapper

    This class encapsulates the gym NLE environment.

    Additional Features:
    - **rgb_array Render Mode**: Provides RGB images of the environment for enhanced visualization.
    - **Des-file Loading**: Supports the loading of des-files, in which case a MiniHack environment will be used instead.
    """

    metadata = {'render.modes': ['rgb_array', *NLE.metadata["render.modes"]]}

    def __init__(self,
        render_mode="human",
        des_file=None,
        character="@",
        allow_all_yn_questions=True,
        allow_all_modes=True,
        autopickup=True,
        actions=nethack.ACTIONS,
        observation_keys=(
            "glyphs",
            "chars",
            "colors",
            "specials",
            "blstats",
            "message",
            "inv_glyphs",
            "inv_strs",
            "inv_letters",
            "inv_oclasses",
            "screen_descriptions",
            "tty_chars",
            "tty_colors",
            "tty_cursor",
        )
    ):
        kwargs = {
            "character": character,
            "allow_all_yn_questions": allow_all_yn_questions,
            "allow_all_modes": allow_all_modes,
            "actions": actions,
            "observation_keys": observation_keys,
            "max_episode_steps": sys.maxsize
        }
        if des_file:
            self.env = MiniHack(des_file=des_file, autopickup=autopickup, **kwargs)
        else:
            kwargs["options"] = (
                "color",  # Display color for different monsters, objects, etc
                "showexp",  # Display the experience points on the status line
                "nobones",  # Disallow saving and loading bones files
                "nolegacy",  # Not display an introductory message when starting the game
                "nocmdassist",  # No command assistance
                "disclose:+i +a +v +g +c +o",  # End of game prompt replies
                "runmode:teleport",  # Update the map after movement has finished
                "mention_walls",  # Give feedback when walking against a wall
                "nosparkle",  # Not display sparkly effect for resisted magical attacks
                "showscore",  # Shows approximate accumulated score on the bottom line
            )
            if not autopickup:
                kwargs["options"] += ("!autopickup",)
            else:
                kwargs["options"] += ("autopickup",)
            self.env = NLE(**kwargs)

        self.render_mode = render_mode
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        if render_mode == "rgb_array":
            self.font = ImageFont.truetype("TheSansMono-Plain.otf", 12)

    def step(self, action):
        # Did we receive an enum value like CompassDirection.SE? If thats the case map it to the corresponding index
        if isinstance(action, enum.Enum):
            action = self.env.actions.index(int(action))

        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # NLE / Minihack still use the old env.seed() approach
        # seed=None also generates a new random seed which is not controlled by us
        # So easiest way to prevent this is by using our own random generator
        # As our generator will be deterministic if reset was called with a seed 
        if seed is None:
            seed = self.np_random.integers(sys.maxsize)
        self.env.seed(core=seed, disp=seed, reseed=False)

        obs = self.env.reset()
        return obs, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return render_ascii_map(self.tty_chars, self.font, is_ascii_map=True)

        return self.env.render()

    @property
    def waiting_for_yn(self) -> bool:
        return bool(self.env.last_observation[self.env._internal_index][1])
    
    @property
    def waiting_for_line(self) -> bool:
        return bool(self.env.last_observation[self.env._internal_index][2])
    
    @property
    def waiting_for_space(self) -> bool:
        return bool(self.env.last_observation[self.env._internal_index][3])

    @property
    def tty_chars(self):
        return self.env.last_observation[self.env._observation_keys.index("tty_chars")]

    @property
    def gym_env(self):
        return self.env