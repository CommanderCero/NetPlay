from netplay.nethack_utils.nle_wrapper import render_ascii_map
from netplay.nethack_agent.tracking import BLStats
from netplay.nethack_agent.describe import describe_glyph

from nle_language_wrapper.nle_language_obsv import NLELanguageObsv
from minihack.tiles.glyph_mapper import GlyphMapper

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import math
import multiprocessing
from collections import deque

WINDOW_NAME = "NetHackAgent"

class VideoWriter:
    def __init__(self, path, fps, resolution=1080):
        self.path = path
        self.out = None  # lazy init
        self.fps = fps
        self.resolution = (round(resolution * 16 / 9), resolution)

    def _make_writer(self, frame):
        h, w = frame.shape[:2]
        print(f'Initializing video writer with resolution {w}x{h}: {self.path}')
        return cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))

    def write(self, frame):
        frame = cv2.resize(frame, self.resolution)
        frame = frame.astype(np.uint8)[..., ::-1]
        if self.out is None:
            self.out = self._make_writer(frame)
        self.out.write(frame)

    def close(self):
        if self.out:
            self.out.release()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, tb):
        self.close()

class AgentVideoRenderer:
    def __init__(self, video_path: str, history_length=10, font_size=12, render=False):
        self.glyph_mapper = GlyphMapper()
        self.video_writer = VideoWriter(video_path, fps=10)
        self.font = ImageFont.truetype("TheSansMono-Plain.otf", font_size)
        self.header_font = ImageFont.truetype("TheSansMono-Plain.otf", int(font_size * 1.2))
        self.header_height = int(self.header_font.size * 1.6)
        self.lang = NLELanguageObsv()
        self.last_obs = None

        self.action_history = deque(maxlen=history_length)
        self.message_history = deque(maxlen=history_length*2)
        self.thought_history = deque(maxlen=history_length)

        self.render = render
        if render:
            self._display_queue = multiprocessing.Manager().Queue()
            self._display_process = multiprocessing.Process(target=self._display_thread, daemon=False)
            self._display_process.start()

    def init(self, obs):
        self.last_obs = obs
        self._render_frame()

    def add_step(self, obs, action, thoughts, is_ai_thought=False):
        self.last_obs = obs
        self.action_history.appendleft(action)
        self.message_history.appendleft(self.lang.text_message(obs["tty_chars"]).decode("latin-1"))
        self.thought_history.appendleft((thoughts, is_ai_thought))
        self._render_frame()

    def add_thoughts(self, thoughts: str, is_ai_thought=False):
        self.action_history.appendleft(None)
        self.message_history.appendleft(None)
        self.thought_history.appendleft((thoughts, is_ai_thought))
        self._render_frame()

    def _render_frame(self):
        scene = self._render_scene(self.last_obs)
        topbar = self._render_topbar(scene.shape[1])
        bottombar = self._render_bottombar(scene.shape[1])
        frame = np.concatenate([topbar, scene, bottombar], axis=0)
        inventory = self._render_inventory(frame.shape[0] - self.header_height)
        inventory = self._render_border(self._add_header(inventory, "Inventory"))
        frame = np.concatenate([frame, inventory], axis=1)
        self.video_writer.write(frame)

        if self.render:
            self._display_queue.put(frame[..., ::-1].copy())
        
    def _render_topbar(self, width):
        action_history = self._render_action_history(math.floor(width*0.06))
        thought_history = self._render_thoughts(width - action_history.shape[1])
        #filler = np.zeros((action_history.shape[0], width-action_history.shape[1]-message_history.shape[1], 3), dtype=np.uint8)
        return np.concatenate([
            self._render_border(self._add_header(action_history, "Actions")),
            self._render_border(self._add_header(thought_history, "Agent Thoughts"))
        ], axis=1)
    
    def _render_bottombar(self, width):
        height = self.font.size * len(self.last_obs['tty_chars'])
        tty = self._render_tty(self.last_obs, int(width * 0.4), height)
        message_history = self._render_game_messages(int((width - tty.shape[1]) * 0.6), tty.shape[0])
        stats = self._render_stats(width - tty.shape[1] - message_history.shape[1], height)
        return np.concatenate([
            self._render_border(self._add_header(tty, "Terminal")),
            self._render_border(self._add_header(message_history, "Message History")),
            self._render_border(self._add_header(stats, "Stats"))
        ], axis=1)
    
    def _render_inventory(self, height):
        width = 500
        vis = np.zeros((height, width, 3), dtype=np.uint8)
        vis = Image.fromarray(vis)
        draw = ImageDraw.Draw(vis)

        inv_strs = self.last_obs["inv_strs"]
        inv_letters = self.last_obs["inv_letters"]
        num_items = len([x for x in inv_letters if x != 0])

        lines = []
        for letter, description in zip(inv_letters[:num_items], inv_strs):
            letter = chr(letter)
            description = "".join([chr(c) for c in description if c != 0])

            lines.append(f"{letter} - {description}")

        draw.multiline_text((0,0), "\n".join(lines), font=self.font, fill="white")
        return np.array(vis)
    
    def _render_tty(self, obs, width, height):
        vis = render_ascii_map(obs["tty_chars"], self.font, is_ascii_map=True)
        vis = Image.fromarray(vis).resize((width, height), Image.LANCZOS)
        return np.array(vis)

    def _render_stats(self, width, height):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        stats = BLStats(*self.last_obs["blstats"])
        agent_glyph = self.last_obs["glyphs"][stats.y, stats.x]

        hunger_lookup = {
            0: "SATIATED",
            1: "NOT_HUNGRY",
            2: "HUNGRY",
            3: "WEAK",
            4: "FAINTING",
            5: "FAINTED",
            6: "STARVED"
        }

        txt = "\n".join([
            f"Agent Glyph: {describe_glyph(agent_glyph)}",
            f'Level num: {stats.level_number}',
            f'Dung num: {stats.dungeon_number}',
            f'Time: {stats.time}',
            f'Score: {stats.score}',
            f'HP: {stats.hitpoints} / {stats.max_hitpoints}',
            f'Armor Class: {stats.armor_class}',
            f'LVL: {stats.experience_level}',
            f'ENERGY: {stats.energy} / {stats.max_energy}',
            f'Gold: {stats.gold}',
            f'Strength: {stats.strength}',
            f'Constitution: {stats.constitution}',
            f'Wisdom: {stats.wisdom}',
            f'Charisma: {stats.charisma}',
            f'Intelligence: {stats.intelligence}',
            f'Dexterity: {stats.dexterity}',
            f'Carry Capacity: {stats.carrying_capacity}',
            f"Hunger: {hunger_lookup[stats.hunger_state]}"
        ])

        draw = ImageDraw.Draw(img)
        draw.multiline_text((1, 1), txt, anchor="la", font=self.font, fill="white")

        return np.array(img)
    
    def _render_action_history(self, width):
        key_presses = ["" if a is None else repr(chr(int(a))) for a in self.action_history]
        text = "\n".join(key_presses)

        img = Image.new('RGB', (width, self.font.size * self.action_history.maxlen + (self.action_history.maxlen * 4 - 4) + 2), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)

        draw.multiline_text((img.width / 2, 1), text, font=self.font, fill="white", anchor="ma", spacing=4)
        return np.array(img)
    
    def _render_game_messages(self, width, height):
        messages = ["" if a is None else a.replace('\n', '') for a in self.message_history]
        text = "\n".join(messages)

        img = Image.new('RGB', (width, height), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)

        draw.multiline_text((1, 1), text, font=self.font, fill="white", spacing=4)
        return np.array(img)
    
    def _render_thoughts(self, width):
        messages = [("", False) if a is None else (a, is_ai) for a, is_ai in self.thought_history]

        img = Image.new('RGB', (width, self.font.size * self.thought_history.maxlen + (self.thought_history.maxlen * 4 - 4) + 2), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)

        x, y = 1, 1
        for text, is_ai_thought in messages:
            draw.text((x, y), text, font=self.font, fill="crimson" if is_ai_thought else "white")
            y += self.font.size + 4

        #draw.multiline_text((1, 1), text, font=self.font, fill="white", spacing=4)
        return np.array(img) 

    def _render_scene(self, obs):
        img = self.glyph_mapper.to_rgb(obs["glyphs"])
        return img
    
    def _add_header(self, img: np.ndarray, text: str):
        header = np.zeros((self.header_height, img.shape[1], 3), dtype=img.dtype)
        header = Image.fromarray(header)
        draw = ImageDraw.Draw(header)
        draw.text((1, 1), text, font=self.header_font, fill="white", anchor="la")
        return np.concatenate([np.array(header), img], axis=0)
    
    def _render_border(self, img: np.array, color=(90, 90, 90), thickness=1):
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        width, height = draw.im.size
        draw.rectangle([(0, 0), (width - 1, height - 1)], outline=color, width=thickness)
        return np.array(img)

    def _display_thread(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)

        last_size = (None, None)
        image = None
        while 1:
            is_new_image = False
            try:
                while 1:
                    try:
                        image = self._display_queue.get(timeout=0.03)
                        is_new_image = True
                    except:
                        break

                if image is None:
                    image = self._display_queue.get()
                    is_new_image = True

                width = cv2.getWindowImageRect(WINDOW_NAME)[2]
                height = cv2.getWindowImageRect(WINDOW_NAME)[3]
                ratio = min(width / image.shape[1], height / image.shape[0])
                width, height = round(image.shape[1] * ratio), round(image.shape[0] * ratio)

                if last_size != (width, height) or is_new_image:
                    last_size = (width, height)

                    resized_image = cv2.resize(image, (width, height), cv2.INTER_AREA)
                    cv2.imshow(WINDOW_NAME, resized_image)

                cv2.waitKey(1)
            except KeyboardInterrupt:
                pass
            except (ConnectionResetError, EOFError):
                return

        cv2.destroyWindow(WINDOW_NAME)

    def close(self):
        self.video_writer.close()
        if self.render:
            self._display_process.terminate()
            self._display_process.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, tb):
        self.close()