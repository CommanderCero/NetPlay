from netplay.nethack_agent.tracking import Level
import netplay.nethack_utils.glyphs as G

import numpy as np

import math

from typing import List, Tuple

class BFSResults:
    def __init__(self, map_shape):
        self.dist = np.full(map_shape, -1, dtype=np.int32)
        self.prev_x = np.full(map_shape, -1, dtype=np.int32)
        self.prev_y = np.full(map_shape, -1, dtype=np.int32)
        self.reachable = np.full(map_shape, False, dtype=bool)
        self.potentially_reachable = np.full(map_shape, False, dtype=bool)

    def get_reachable_tiles(self):
        for y in range(self.reachable.shape[0]):
            for x in range(self.reachable.shape[1]):
                if self.reachable[y,x]:
                    yield (x, y, self.dist[y,x])

    def distance_to(self, x, y):
        return self.dist[y,x] if self.dist[y,x] != -1 else None

    def get_path(self, x, y):
        if not self.reachable[y, x]:
            return None

        path = [(x, y)]
        while self.dist[y, x] != 0:
            nx, ny = self.prev_x[y, x], self.prev_y[y, x]
            path.append((nx, ny))
            x, y = nx, ny
        return path[::-1] # Return the reverted path, since we constructed it from goal to start

def nethack_bfs(x, y, walkable_mask: np.ndarray, diagonally_walkable_mask: np.ndarray, can_squeeze: bool) -> BFSResults:
    data = BFSResults(walkable_mask.shape)
    data.dist[y,x] = 0
    data.reachable[y,x] = True

    buf = np.zeros((walkable_mask.shape[0] * walkable_mask.shape[1], 2), dtype=np.uint32)
    index = 0
    buf[index] = (y, x)
    size = 1
    while index < size:
        y, x = buf[index]
        index += 1

        for (dx, dy) in ((0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)):
            py, px = y + dy, x + dx
            # Check bounds
            if py < 0 or py >= walkable_mask.shape[0] or px < 0 or px >= walkable_mask.shape[1]:
                continue

            # Can we walk to this tile?
            is_walkable = walkable_mask[py, px]
            can_walk_horizontally = abs(dy) + abs(dx) <= 1
            can_walk_diagonally = diagonally_walkable_mask[py, px] and diagonally_walkable_mask[y, x] and (can_squeeze or walkable_mask[py, x] or walkable_mask[y, px])
            if is_walkable and (can_walk_horizontally or can_walk_diagonally):
                # Did we reach a new tile? No need to check if we found a shorter path, because we use bfs with uniform costs 
                if data.dist[py, px] == -1:
                    data.dist[py, px] = data.dist[y, x] + 1
                    data.prev_x[py, px] = x
                    data.prev_y[py, px] = y
                    data.reachable[py, px] = True
                    buf[size] = (py, px)
                    size += 1

    return data

class LevelPath:
    def __init__(self, level: Level, positions: List[Tuple[int, int]]):
        self.level = level
        self.positions = positions

    def __getitem__(self, index):
        return self.positions[index]

    def __len__(self):
        return len(self.positions)

class LevelPathfinder:
    def __init__(self, level: Level, bfs_results: BFSResults, avoid_monsters_bfs_results: BFSResults):
        self.level = level
        self.bfs_results = bfs_results
        self.avoid_monsters_bfs_results = avoid_monsters_bfs_results

    def distance_to(self, x, y, bump_into_unwalkables=True, avoid_monsters=False) -> int:
        path = self.get_path_to(x, y, bump_into_unwalkables=bump_into_unwalkables, avoid_monsters=avoid_monsters)
        return len(path) if path is not None else math.inf

    def get_path_to(self, x, y, bump_into_unwalkables=True, avoid_monsters=False) -> LevelPath:
        path = None
        if avoid_monsters:
            path = self._get_path_to(self.avoid_monsters_bfs_results, x, y, bump_into_unwalkables)
        if path is None:
            path = self._get_path_to(self.bfs_results, x, y, bump_into_unwalkables)
        return path
    
    def get_distance_map(self, avoid_monsters=False) -> np.ndarray:
        dist = self.bfs_results.dist
        if avoid_monsters:
            can_avoid_mask = self.avoid_monsters_bfs_results.dist != -1
            dist[can_avoid_mask] = self.avoid_monsters_bfs_results.dist[can_avoid_mask]
        return dist
    
    def _get_path_to(self, bfs_results: BFSResults, x, y, bump_into_unwalkables):
        path = bfs_results.get_path(x, y)
        if path is not None:
            return LevelPath(self.level, path)
        if bump_into_unwalkables:
            # Find a path adjacent to the target tile and then bump into the target tile
            # For example this is used to open doors or to explore
            neighbors = self.level.get_neighbors(x, y)
            neighbor_paths = [bfs_results.get_path(nx, ny) for nx, ny in neighbors if bfs_results.get_path(nx, ny) is not None]
            shortest_neighbor_path = min(neighbor_paths, key=len, default=None)
            if shortest_neighbor_path is not None:
                return LevelPath(self.level, [*shortest_neighbor_path, (x,y)])
        return None

    @classmethod
    def from_level(self, x, y, level: Level, can_squeeze: bool):
        walkable_mask = level.get_walkable_mask()
        diagonally_walkable_mask = level.get_diagonal_walkable_mask()
        bfs_results = nethack_bfs(
            x=x, 
            y=y, 
            walkable_mask=walkable_mask, 
            diagonally_walkable_mask=diagonally_walkable_mask, 
            can_squeeze=can_squeeze
        )

        # Monsters are treated walkable by default
        walkable_mask = level.get_walkable_mask(treat_monster_unwalkable=True)
        diagonally_walkable_mask = level.get_diagonal_walkable_mask(treat_monster_unwalkable=True)
        avoid_monsters_bfs_results = nethack_bfs(
            x=x, 
            y=y, 
            walkable_mask=walkable_mask, 
            diagonally_walkable_mask=diagonally_walkable_mask, 
            can_squeeze=can_squeeze
        )

        return LevelPathfinder(level, bfs_results, avoid_monsters_bfs_results)