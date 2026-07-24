"""
Multi-agent stigmergic exploration/sampling simulation over the Jezero Crater DEM.

Overview
--------
This script simulates a swarm of aerial "drones" (fast, long-range scouts) and
ground "rovers" (slow, sample-collecting agents) operating on a real Digital
Elevation Model (DEM) of Jezero Crater. Coordination between agents is achieved
through stigmergy: agents read and write values on a shared 2D information
field instead of communicating directly.

Field value convention (see FIELD PARAMETERS below):
    - Negative values  -> agent-deposited trails/tracks (decaying over time)
    - EMPTY (-1.0)     -> untouched terrain
    - HQ_VAL           -> the lander / home base (frozen, never decays)
    - >= POI_MIN (0.0) -> an active or exhausted point of interest (POI)
    - POI_VAL (1.0)    -> a freshly discovered POI, full of samples

Drones explore, discover new POIs, and lay down a trail on their way back to
HQ. Rovers then follow the trail gradient out to POIs, collect samples, and
retrace their steps home to deliver them. Over time, unused trails decay
(modeling aeolian/dust burial), keeping the field responsive to the most
recently used paths.
"""

import numpy as np
import rasterio
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
import matplotlib 
matplotlib.use("pgf")
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from pyproj import Transformer
from collections import defaultdict
import random
from scipy.ndimage import maximum_filter
import pandas as pd
import os



# ───────────────────── DATA LOADING ─────────────────────────
DEM_PATH = "jezero.tif"

# Three batches of real Mars 2020 (Perseverance) sampling-campaign coordinates
# (lat, lon in Mars areocentric degrees), grouped by mission phase / region.
# These are used as realistic POI placements instead of randomly generated ones.
POI_LIST_1 = [
    (18.42769340, 77.45165066),     # M2020-164-2 Roubion
    (18.43074132, 77.44436502),     # M2020-196-4 Montagnac
    (18.43397, 77.44301),           # M2020-271-6 Coulettes
    (18.43264, 77.44133),           # M2020-337-8 Malay
    (18.44386406, 77.45242176)      # M2020-371-9 Hahonih
]

POI_LIST_2 = [
    (18.458931,   77.40617078),     # M2020-490-11 Swift Run
    (18.45863832, 77.40588685),      # M2020-516-15 Bearwallow
    (18.45068664, 77.40143554),     # M2020-579-17 Mageik
    (18.45363597, 77.39911421),     # M2020-623-19 Kukaklek
    (18.45131442, 77.40121811)     # M2020-634-20 Atmo Mountain   
]

POI_LIST_3 = [
    (18.48380149, 77.3604692),      # M2020-882-24 Pilot Mountain
    (18.48344692, 77.35065098),     # M2020-923-25 Pelican Point
    (18.48867278, 77.34704595),     # M2020-949-26 Lefroy Bay
    (18.49186664, 77.3272192),      # M2020-1088-27 Comet Geyser
    (18.49747434, 77.30514915)      # M2020-1215-28 Sapphire Canyon
]


# ───────────────────── FIELD PARAMETERS ─────────────────────
# Stigmergic field value convention: negative = agent trail/track, 0..1 = POI
# state, HQ_VAL = frozen lander cell. See module docstring for full details.
EMPTY   = -1.0          # Empty Stigmergic Pixel Value
HQ_VAL  = -5.0          # Lander Stigmergic Pixel Value

POI_VAL =  1.0          # New Point-of-Interest Stigmergic Pixel Value
POI_MIN =  0.0          # Exhausted Point-of-Interest Stigmergic Pixel Value

TRAIL_INIT = -0.8       # Drone-Trail Stigmergic Pixel Value (initial deposit)
TRACK_INIT = -0.3       # Rover-Track Stigmergic Pixel Value (initial deposit, weaker than a drone trail)
TRAIL_MIN  = -1.0       # Floor value that decay asymptotically approaches

N_POI = 0               # Number of Point-of-Interest (unused placeholder, kept for reference)


# ───────────────────── AGENT PARAMETERS ─────────────────────
N_DRONES = 3            # Number of Drones
N_ROVERS = 5            # Number of Rovers

SPEED_DRONE_MPS = 10.0  # [m/s] Drone Speed 
SPEED_ROVER_MPS = 5.0   # [m/s] Rover Speed

DRONE_BIAS = 1          # Drone determinist movement bias towards nearest POI (0 = random, 1 = deterministic)
SCAN_RADIUS = 3         # [-] Visible Pixel radius to see new POIs

ROVER_MEMORY = 10000       # Max. Memory of traveled route (recently visited cells to avoid, as a deque)

SMPL_DLV_T = 120        # [s] Time needed to deliver sample from rover to lander


DRONE_MAX_RANGE_M = 700.0   # [m] Max Range for 1 Flight before the drone must recharge
DRONE_RECHARGE_TICKS = 30   # [-] Ticks spent recharging once max range is reached
CHANCE_VAL = 1e-3           # [-] Per-tick probability of discovering a new POI while exploring

# ─────────────────── SIMULATION PARAMETERS ───────────────────
SIM_DURATION_S = 2000   # [s] Desired Simulation Time 


ANIM_INTERVAL_MS = 5    # [ms] Animation frame dt
SAVE_GIF = True        # Flag to save GIF
GIF_PATH = "sim.gif"    # GIF path



# ────────────────────── LOADING PERSEVERANCE DATA ────────────────────────

def latlon_to_dem_km(dem_path, lat, lon):
    """
    Convert a Mars areocentric (lat, lon) coordinate into the DEM's projected
    CRS, expressed in kilometers.

    Parameters
    ----------
    dem_path : str
        Path to the DEM GeoTIFF, used only to read its CRS.
    lat, lon : float
        Latitude/longitude in degrees (Mars areocentric sphere).

    Returns
    -------
    (x_km, y_km) : tuple of float
        Projected coordinates in kilometers, in the DEM's own CRS.
    """
    with rasterio.open(dem_path) as src:
        crs = src.crs
    # Mars areocentric lat/lon
    mars_geographic = "+proj=longlat +a=3396190 +b=3396190 +no_defs"
    transformer = Transformer.from_crs(mars_geographic, crs, always_xy=True)
    x, y = transformer.transform(lon, lat)                                      # attenzione: x=lon, y=lat nell'ordine always_xy
    return x/1000, y/1000  # [km]

# Convert the real Perseverance landing site and its sampling campaigns from
# lat/lon into the DEM's projected km coordinate system, used throughout the
# simulation.
hq_x, hq_y = latlon_to_dem_km("jezero.tif", 18.4447, 77.4508)
print("HQ_KM =", (hq_x, hq_y))
HQ_KM = (hq_x, hq_y)                                                                # Perseverance Landing Site
POI_LIST_KM = [latlon_to_dem_km("jezero.tif", lat, lon) for lat, lon in POI_LIST_1]   # Perseverance POIs


# ───────────────────────── CROP ─────────────────────────────
# The full DEM is much larger than needed; crop to a bounding box around HQ
# and the POIs to keep the simulation grid small and fast.
CROP_MODE = "km"          # "pixel" | "km" | None

CROP_PIXEL = (500, 800,   # row_min, row_max
              300, 600)   # col_min, col_max

all_p = POI_LIST_KM + [HQ_KM]

CROP_KM = (min(p[0] for p in all_p)-0.3, max(p[0] for p in all_p)+0.4,    # x_min, x_max
           min(p[1] for p in all_p)-1.2, max(p[1] for p in all_p)+0.2)    # y_min, y_max



# ────────────────────── AGENT STATES ────────────────────────
class State(Enum):
    """Finite-state-machine states shared by both Drone and Rover agents."""
    IDLE      = auto()   # Waiting for a trigger (rovers wait for a signal near HQ)
    EXPLORING = auto()   # Moving outward, searching for / heading to a POI
    RETURNING = auto()   # Heading back to HQ, depositing a trail/track
    SAMPLING  = auto()   # Stationary at a POI, collecting a sample
    RECHARGING = auto()  # Drone-only: grounded after reaching max flight range

STATE_LABEL = {s: s.name[:3] for s in State}                        # State label for visualization purposes



# ───────────────────────── DEM ──────────────────────────────
class DEM:
    """
    Loads a GeoTIFF elevation raster, optionally crops it to a region of
    interest, and provides helpers to convert between the raster's pixel
    grid and its projected coordinate system (in kilometers).
    """
    def __init__(self, path):
        with rasterio.open(path) as src:
            full  = src.read(1).astype("float32")
            b     = src.bounds
            H, W  = full.shape
            nodata = src.nodata
            self.px_size_x = (b.right - b.left) / W   # [m] per coloumn
            self.px_size_y = (b.top - b.bottom) / H   # [m] per row
            self.px_size = (self.px_size_x + self.px_size_y) / 2  # average, assuming square pixels


        # ── calcola finestra di crop ────────────────────────────
        if CROP_MODE == "pixel":
            r0, r1, c0, c1 = CROP_PIXEL
        elif CROP_MODE == "km":
            xmin, xmax, ymin, ymax = CROP_KM
            c0 = int(np.clip((xmin - b.left/1000)  / ((b.right - b.left)/1000) * W, 0, W-1))
            c1 = int(np.clip((xmax - b.left/1000)  / ((b.right - b.left)/1000) * W, 0, W-1))
            r0 = int(np.clip((b.top/1000 - ymax)   / ((b.top - b.bottom)/1000) * H, 0, H-1))
            r1 = int(np.clip((b.top/1000 - ymin)   / ((b.top - b.bottom)/1000) * H, 0, H-1))
        else:
            r0, r1, c0, c1 = 0, H, 0, W

        self._row0, self._col0 = r0, c0          # offset per km_to_pixel

        self.data  = full[r0:r1, c0:c1]
        self.shape = self.data.shape

        # ricalcola bounds ritagliati (in metri, come rasterio)
        pw = (b.right - b.left) / W             # larghezza pixel in m
        ph = (b.top  - b.bottom) / H            # altezza pixel in m
        self.bounds = type(b)(
            left   = b.left  + c0 * pw,
            right  = b.left  + c1 * pw,
            top    = b.top   - r0 * ph,
            bottom = b.top   - r1 * ph,
        )

        if nodata is not None:
            self.data[self.data == nodata] = np.nan
    def km_to_pixel(self, x_km, y_km):
        """Convert projected coordinates (km) to a clipped (row, col) pixel index."""

        b = self.bounds
        H, W = self.shape

        col = int((x_km - b.left/1000) / ((b.right - b.left)/1000) * W)
        row = int((b.top/1000 - y_km) / ((b.top - b.bottom)/1000) * H)

        return (
            int(np.clip(row, 0, H - 1)),
            int(np.clip(col, 0, W - 1))
        )

    def passable(self, r, c):
        """A cell is passable/traversable if it has valid (non-NaN) elevation data."""
        return not np.isnan(self.data[r, c])

dem = DEM(DEM_PATH)
DT = dem.px_size / SPEED_DRONE_MPS                                  # [s] time for drone to cross 1 pixel == Tick Time
ROVER_TICK_RATIO = round(SPEED_DRONE_MPS / SPEED_ROVER_MPS) 
SIM_STEPS = int(SIM_DURATION_S / DT)


# ───────────────────────── FIELD ────────────────────────────
class InfoField:
    """
    The shared stigmergic field: a 2D grid overlaid on the DEM that stores
    trail/track/POI values. Agents read this grid to decide where to go and
    write to it to leave markers for other agents, instead of communicating
    directly.
    """

    def __init__(self, dem, hq_km, poi_km_list=None):

        self.dem = dem
        self.bounds = dem.bounds
        self.shape = dem.shape
        self.total_delivered = 0

        # matrice campo
        self.grid = np.full(self.shape, EMPTY, dtype=np.float32)

        # frozen (HQ non modificabile)
        self._frozen = np.zeros(self.shape, dtype=bool)

        # ── HQ ─────────────────────────────
        hr, hc = dem.km_to_pixel(*hq_km)
        self.hq = (hr, hc)

        self.grid[hr, hc] = HQ_VAL
        self._frozen[hr, hc] = True

        # ── POI ────────────────────────────
        self.poi = []

        if poi_km_list:
            for x_km, y_km in poi_km_list:
                r, c = dem.km_to_pixel(x_km, y_km)

                if dem.passable(r, c):
                    self.grid[r, c] = POI_VAL
                    self.poi.append((r, c))

    # Used for Prelocated POIs
    def add_poi_km(self, x_km, y_km):
        """Manually register a POI at the given projected (km) coordinate."""
        r, c = self.dem.km_to_pixel(x_km, y_km)

        if self.dem.passable(r, c):
            self.grid[r, c] = POI_VAL
            self.poi.append((r, c))

    # Used when Drones find new POIs
    def mark_poi(self, row, col):
        """Register a newly discovered POI cell (called by exploring drones)."""
        if not self._frozen[row, col]:
            self.grid[row, col] = POI_VAL
            self.poi.append((row, col))

    def deposit_trail(self, row, col):
        """Lay down a drone trail marker, unless a stronger signal already exists there."""
        if not self._frozen[row, col] and self.grid[row, col] < TRAIL_INIT:                  # and self.grid[row, col] < TRAIL_MIN + 0.05
            self.grid[row, col] = TRAIL_INIT

    def decay(self, dt=1.0, rate=1e-5):
        """
        Erode all non-frozen, non-POI cells toward TRAIL_MIN at a fixed rate.
        Models aeolian (wind-driven) burial: unused trails fade over time so
        that the field favors recently reinforced paths.
        """
        mask = (~self._frozen) & (self.grid < POI_MIN)
        self.grid[mask] -= rate * dt
        np.clip(self.grid, TRAIL_MIN, None, out=self.grid)
        # self.grid[self.hq_pixel] = HQ_VAL
    
    def neighbors(self, row, col):
        """Return the (up to 4) orthogonal neighbor cells within grid bounds."""
        candidates = [
            (row-1, col), (row+1, col),
            (row, col-1), (row, col+1)
        ]
        H, W = self.shape
        return [(r, c) for r, c in candidates if 0 <= r < H and 0 <= c < W]
    
    def has_signal(self, radius=1):
        """
        True if at least 1 cell with POI or trail exists near HQ.
        Used to wake up idle rovers once a drone trail reaches home.
        """
        hr, hc = self.hq
        r0, r1 = max(0, hr-radius), min(self.shape[0], hr+radius+1)
        c0, c1 = max(0, hc-radius), min(self.shape[1], hc+radius+1)
        patch = self.grid[r0:r1, c0:c1]
        frozen_patch = self._frozen[r0:r1, c0:c1]
        return bool(np.any((patch > EMPTY) & (~frozen_patch)))

    def sample_poi(self, row, col):
        """Consume one unit of "sample" value from a POI cell (rover collecting a sample)."""
        if not self._frozen[row, col]:
            self.grid[row, col] -= 0.21

    def deposit_track(self, row, col):
        """Lay down a (weaker) rover track marker, only over non-POI, unmarked cells."""
        if not self._frozen[row, col] and self.grid[row, col] < POI_MIN :                  # and self.grid[row, col] < TRAIL_MIN + 0.05
            self.grid[row, col] = TRACK_INIT

    def count_signal_neighbors(self, row, col, exclude=None):
        """Count neighboring cells with field value > EMPTY, optionally excluding one cell.
        Used by rovers to detect trail branch points while retreating from a dead end."""
        n = 0
        for rc in self.neighbors(row, col):
            if exclude is not None and rc == exclude:
                continue
            if self.grid[rc[0], rc[1]] > EMPTY:
                n += 1
        return n

    def erase_pixel(self, row, col):
        """Reset a cell back to baseline (EMPTY), removing any trail/track there.
        Used to prune dead-end branches once a rover backtracks past them."""
        if not self._frozen[row, col]:
            self.grid[row, col] = EMPTY


# ── Standard Agent Modes ────────────────────────────────────────────────────────────
@dataclass
class Agent:
    """Base class shared by Drone and Rover, holding position, state, and speed bookkeeping."""
    id:    int                                  # Unique Code
    row:   int                                  # X coordinate
    col:   int                                  # Y coordinate
    hq:    tuple = (0, 0)                       # HQ Coordinates (maybe not needed)
    state: State = State.IDLE                   # State (standard set to IDLE)
    path:  list  = field(default_factory=list)  # Saves the history of the agent's position
    speed: float = 1.0                          # [m/s] Agent Speed
    _move_acc: float = 0.0                      # Pixel Accumulation to account for agent speed
    

    def pixels_this_tick(self, dem, dt):
        """How many pixels can this agent move in this tick."""
        self._move_acc += (self.speed * dt) / dem.px_size
        n_steps = int(self._move_acc)      # Integer
        self._move_acc -= n_steps          # Remainder for for next tick
        return n_steps


    #  ── Motion Function ────────────
    def move_to(self, row, col):                
        self.row, self.col = row, col           # Updates coordinates

    #  ──Checks if Agent is at HQ────────────
    def at_hq(self):
        return (self.row, self.col) == self.hq


# ── Drone ──────────────────────────────────────────────────────────────────
class Drone(Agent):
    """
    Behavior:
    EXPLORING: Moves according to a random walk biased toward the nearest point of interest (POI).
    At each step, there is a 1% probability of discovering a new POI within the surrounding area.
    If a POI is detected within the scanning radius: 
        - moves to the POI;
        - returns to the starting location while leaving a trail.
    """

    def __init__(self, field, *args, scan_radius=10, bias=DRONE_BIAS, rng=None,
             max_range_m=DRONE_MAX_RANGE_M, recharge_ticks=DRONE_RECHARGE_TICKS, **kwargs):
        super().__init__(*args, **kwargs)
        self.row         = field.hq[0]
        self.col         = field.hq[1]
        self.scan_radius = scan_radius                          # Sets Visible Radial Range
        self.bias        = bias                                 # Sets bias
        self.rng         = rng or np.random.default_rng()       # Sets Random POI generation Model
        self.target      = random.choice(field.poi) if field.poi else None
        self.hq_timer    = 0
        self._levy_remaining = 0
        self._levy_dir = (0, 0)
        self._pre_recharge_state = State.EXPLORING   # Status Memory for Charging
        self.max_range_m    = max_range_m
        self.recharge_ticks = recharge_ticks
        self.distance_m      = 0.0              # [m] Distance Traveled since last recharge
        self.recharge_timer  = 0
        
    
    # ── TICK FUNCTION ── gets called for each timestep to check agent state ───
    def tick(self, field: InfoField, dem: DEM):
        """Advance the drone by one simulation tick, dispatching to the
        behavior for its current FSM state, then updating traveled distance
        and triggering a recharge stop once max range is reached."""
        prev_pos = (self.row, self.col)

        match self.state:
            case State.EXPLORING:
                self._explore(field, dem)
            case State.RETURNING:
                self._return(field, dem)
            case State.RECHARGING:
                self._recharge(field, dem)
            case State.IDLE:
                pass

        # ── Updates Traveled distance ──
        if self.state != State.RECHARGING and self.state != State.IDLE:
            dr = self.row - prev_pos[0]
            dc = self.col - prev_pos[1]
            if dr or dc:
                self.distance_m += np.hypot(dr, dc) * dem.px_size

            # ── Check for Range ──
            if self.distance_m >= self.max_range_m:
                self._pre_recharge_state = self.state
                self.state = State.RECHARGING

    # ── EXPLORATION MODE ─────────────────────────────────────────────────────
    def _explore(self, field: InfoField, dem: DEM, chance=CHANCE_VAL):
        """
        One exploration step: take a biased/random walk step toward the
        current target, roll a small chance of discovering a brand-new POI
        within the scan radius, then re-scan the local neighborhood to
        (re)target the closest known POI. Switches to RETURNING once the
        target is reached.
        """
        H, W = dem.shape

        # 1. Biased Movement
        _ = self._biased_step(self.target, field, H, W)           # Outputs next movement tile (and Updates position)

        nr = self.row
        nc = self.col

        if (nr, nc) == self.target:
            self.state = State.RETURNING

        if dem.passable(nr, nc):
            self.path.append((self.row, self.col))      # Updates path history

        # 2. Random POI Discovery
        if self.rng.random() < chance:                  # checks if the event of finding a Target of Opportunity occurs
            
            # Sets the POI acceptable grid (considering map boundaries)
            r0 = max(0, self.row - self.scan_radius)
            r1 = min(H, self.row + self.scan_radius + 1)

            c0 = max(0, self.col - self.scan_radius)
            c1 = min(W, self.col + self.scan_radius + 1)
            
            # Selects the candidate pixels (have to be empty)
            candidates = [
                (r, c)
                for r in range(r0, r1)
                for c in range(c0, c1)
                if dem.passable(r, c) and field.grid[r, c] == EMPTY
            ]

            # Select the newly found POI
            if candidates:
                pr, pc = candidates[self.rng.integers(len(candidates))]
                field.mark_poi(pr, pc)

        # 3. Finds POI inside of scanning radius (Drone favours closeby POIs)
        r0 = max(0, self.row - self.scan_radius)
        r1 = min(H, self.row + self.scan_radius + 1)

        c0 = max(0, self.col - self.scan_radius)
        c1 = min(W, self.col + self.scan_radius + 1)

        poi_cells = np.argwhere(field.grid[r0:r1, c0:c1] >= POI_MIN)    # Finds cells with POIs within its range

        # POI sovrascription (Only ran if there is a POI within range)
        if len(poi_cells) > 0:                          

            # Chooses closest POI
            local_r, local_c = min(
                poi_cells,
                key=lambda rc: (rc[0] - self.scan_radius)**2 + (rc[1] - self.scan_radius)**2
            )

            pr = local_r + r0
            pc = local_c + c0

            # Saves Current Position
            self.path.append((self.row, self.col))

            # Updates new Target 
            self.target = (pr, pc)
    
    def _levy_step(self, dem):
        """Generate a Levy-flight-like random step: pick a random direction
        and hold it for a randomly drawn run length (mix of short and
        occasional long straight segments), used when no target is set."""
        H, W = dem.shape

        if self._levy_remaining <= 0:
            # scegli direzione a caso e lunghezza a caso (a volte corta, a volte lunga)
            self._levy_dir = tuple(self.rng.integers(-1, 2, size=2))
            self._levy_remaining = int(self.rng.choice([1, 2, 3, 5, 10, 20], p=[0.35,0.25,0.2,0.1,0.07,0.03]))

        dr, dc = self._levy_dir
        self._levy_remaining -= 1
        return dr, dc
    
    # ── WALKING FUNCTION ─────────────────────────────────────────────────────
    def _biased_step(self, target: tuple | None, field: InfoField, H: int, W: int):
        """
        Compute one movement step: if no target, perform a Levy-flight random
        walk; otherwise step toward the target with probability `bias`,
        or take a random step otherwise (noise), then apply and clip position.
        """

        # 1. No Target → random walk
        if target is None:
            dr, dc = self._levy_step(dem)

        else:
            tr, tc = target

            # 2. Target Direction
            dr = np.sign(tr - self.row)
            dc = np.sign(tc - self.col)

            # 3. Optional Noise Introduction
            if self.rng.random() >= self.bias:
                dr, dc = self.rng.integers(-1, 2, size=2)

        # 4. Update on Position
        self.row = int(np.clip(self.row + dr, 0, H - 1))
        self.col = int(np.clip(self.col + dc, 0, W - 1))

        return dr, dc

    # ── RETURN MODE ─────────────────────────────────────────────────────
    def _return(self, field: InfoField, dem: DEM):
        """
        Move one slope-aware step back toward HQ while depositing a trail
        marker on the current cell. Once at HQ, wait briefly (hq_timer) then
        switch back to EXPLORING with a freshly chosen active POI target.
        """

        H, W = dem.shape

        # Depositing Trail on Current Position
        if field.grid[self.row, self.col] < POI_MIN:
            field.deposit_trail(self.row, self.col)
        

        if self.at_hq():            # (self.row, self.col) == field.hq

            self.hq_timer += 1

            if self.hq_timer >= 20:
                    self.state = State.EXPLORING
                    valid_poi = [(r, c) for (r, c) in field.poi if field.grid[r, c] > 0]
                    self.target = random.choice(valid_poi) if valid_poi else None
                    self.hq_timer = 0
                    self.path = []    # Clears Path
        else:
            self.hq_timer = 0
            self._slope_aware_step(field.hq, field, dem)   #    # Outputs next movement tile 

    def _slope_aware_step(self, target, field, dem):
        """
        Move one orthogonal step toward `target`, preferring the neighboring
        cell with the smallest elevation gradient (gentlest slope) among the
        candidates that make progress toward the target. Falls back to any
        passable neighbor, then to an expanding-ring search, if the direct
        candidates are blocked (used to route drones/rovers around obstacles
        on the way back to HQ).
        """
        H, W = dem.shape

        if target is None:
            # anche qui, se vuoi restare ortogonali, scegli un solo asse per volta
            dr, dc = self.rng.integers(-1, 2, size=2)
            if dr != 0 and dc != 0:
                if self.rng.random() < 0.5:
                    dc = 0
                else:
                    dr = 0
            nr = int(np.clip(self.row + dr, 0, H-1))
            nc = int(np.clip(self.col + dc, 0, W-1))
            if dem.passable(nr, nc):
                self.move_to(nr, nc)
            return

        tr, tc = target
        step_r = int(np.sign(tr - self.row))
        step_c = int(np.sign(tc - self.col))

        candidates = []
        for dr in ([step_r, 0] if step_r != 0 else [0]):
            for dc in ([step_c, 0] if step_c != 0 else [0]):
                if dr == 0 and dc == 0:
                    continue
                if dr != 0 and dc != 0:      # ← nuovo: esclude la diagonale
                    continue
                nr, nc = self.row + dr, self.col + dc
                if 0 <= nr < H and 0 <= nc < W and dem.passable(nr, nc):
                    candidates.append((nr, nc))

        if not candidates:
            candidates = [
                (self.row+dr, self.col+dc)
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]   # ← solo ortogonali invece di tutte le 8 direzioni
                if 0 <= self.row+dr < H and 0 <= self.col+dc < W
                and dem.passable(self.row+dr, self.col+dc)
            ]

        if not candidates:
            # ── Bloccato: cerca la cella passabile più vicina in raggio crescente (solo anello ortogonale) ──
            for radius in range(2, 10):
                ring = [
                    (self.row+dr, self.col+dc)
                    for dr in range(-radius, radius+1)
                    for dc in range(-radius, radius+1)
                    if max(abs(dr), abs(dc)) == radius
                    and (dr == 0 or dc == 0)     # ← nuovo: solo celle sullo stesso asse, non l'intero anello
                    and 0 <= self.row+dr < H and 0 <= self.col+dc < W
                    and dem.passable(self.row+dr, self.col+dc)
                ]
                if ring:
                    best = min(ring, key=lambda c: (tr-c[0])**2 + (tc-c[1])**2)
                    self.move_to(*best)
                    return
            return

        cur_elev = dem.data[self.row, self.col]

        def slope(cell):
            """Local elevation gradient magnitude between the current cell and `cell`."""
            r, c = cell
            elev = dem.data[r, c]
            if np.isnan(elev) or np.isnan(cur_elev):
                return np.inf
            dist = np.hypot(r - self.row, c - self.col)
            return abs(elev - cur_elev) / dist

        best = min(candidates, key=lambda c: round(slope(c), 4))
        self.move_to(*best)
        
    def _recharge(self, field: InfoField, dem: DEM):
        """Count down the recharge timer; once done, reset traveled distance
        and resume whichever state (EXPLORING/RETURNING) the drone was in
        before it had to land."""
        self.recharge_timer += 1

        if self.recharge_timer >= self.recharge_ticks:
            self.distance_m = 0.0
            self.recharge_timer = 0

            if self._pre_recharge_state == State.RETURNING:
                self.state = State.RETURNING
            else:
                self.state = State.EXPLORING
                valid_poi = [(r, c) for (r, c) in field.poi if field.grid[r, c] > 0]
                # self.target = random.choice(valid_poi) if valid_poi else None

# ── Rover ──────────────────────────────────────────────────────────────────
class Rover(Agent):
    """
    Follows stigmergic field's gradient excluding last N cells visited.
    Once it reaches a POI, samples it and returns to HQ along its path.
    """        

    def __init__(self, *args, memory=ROVER_MEMORY, tick_ratio=ROVER_TICK_RATIO, dt=DT, del_time=SMPL_DLV_T, **kwargs):
        super().__init__(*args, **kwargs)
        self.recent: deque[tuple] = deque(maxlen=memory)
        self.rng = np.random.default_rng()
        self.sample_timer    = 0
        self.hq_timer        = 0
        self.moving_timer    = 0
        self.samples_carried = 0
        self.deadend_retreat = False
        self.tick_ratio = tick_ratio
        self.dt         = dt
        self.del_time   = del_time
        self._tick_count = 0
    

    def tick(self, field: InfoField, dem: DEM):
        """
        Advance the rover by one simulation tick. Rovers move slower than
        drones, so ticks are throttled by `tick_ratio` (a rover only actually
        acts once every `tick_ratio` engine ticks). Dispatches to the
        behavior for the current FSM state.
        """
        self._tick_count += 1
        if self._tick_count < self.tick_ratio:
            return               
        self._tick_count = 0     

        match self.state:
            case State.IDLE:
                if field.has_signal():
                    self.state = State.EXPLORING
            case State.EXPLORING:
                if self.moving_timer > 0:
                    self.moving_timer += 1
                else:
                    self.moving_timer = 0
                    self._follow_gradient(field)
            case State.SAMPLING:
                self._sample(field)
            case State.RETURNING:
                if self.moving_timer > 0:
                    self.moving_timer += 1
                else:
                    self.moving_timer = 0
                    self._return(field)

    def _follow_gradient(self, field: InfoField):
        """
        One exploration step for the rover: prefer any adjacent POI cell; if
        multiple non-visited signal cells (trail/track) are neighbors,
        softmax-sample among them weighted by field strength (favoring
        stronger/fresher trails); if no unvisited signal neighbor exists,
        treat this as a dead end and start retreating (RETURNING).
        Also lays down a rover track on the newly occupied cell.
        """
        self.recent.append((self.row, self.col))

        all_neighbors = field.neighbors(self.row, self.col)

        signal_cells = [
            rc for rc in all_neighbors
            if field.grid[rc[0], rc[1]] > EMPTY
        ]

        poi_cells = [
            rc for rc in all_neighbors
            if field.grid[rc[0], rc[1]] > POI_MIN
        ]

        meaningful = [rc for rc in signal_cells if rc not in self.recent]
        near_poi   = [rc for rc in poi_cells if rc not in self.recent]    

        if near_poi:
            best = near_poi[self.rng.integers(len(near_poi))]
            self.move_to(*best)
            self.state = State.SAMPLING

        elif meaningful:
            # Softmax over field strength: favors the strongest nearby signal
            # while still allowing weaker trails a (smaller) chance.
            values = np.array([field.grid[r, c] for r, c in meaningful])
            beta = 5
            weights = np.exp(beta * (values - values.max()))
            probs = weights / weights.sum()
            idx = self.rng.choice(len(meaningful), p=probs)
            best = meaningful[idx]

            if best in self.path:
                idx_in_path = self.path.index(best)
                self.path = self.path[:idx_in_path + 1]
            else:
                self.path.append((self.row, self.col))

            self.move_to(*best)

        else:
            # Deadend: No pixel with signal outside the one the rover came from
            self.state = State.RETURNING
            self.deadend_retreat = True
            self.path.append((self.row, self.col))


        if field.grid[self.row, self.col] >= POI_MIN:
            self.state = State.SAMPLING

        field.deposit_track(self.row, self.col)

    def _sample(self, field: InfoField):
        """
        Stay at the current POI cell, incrementing the sample timer each
        tick. Once enough ticks have elapsed (del_time / dt / tick_ratio),
        consume one unit of the POI's value and start returning to HQ
        carrying the sample. If the POI got exhausted by another rover in
        the meantime, abandon sampling and return empty-handed.
        """
        if field.grid[self.row, self.col] >= POI_MIN:
            self.sample_timer += 1

            if self.sample_timer >= int(self.del_time/self.dt/ROVER_TICK_RATIO):
                field.sample_poi(self.row, self.col)
                self.state = State.RETURNING
                self.samples_carried = 1
                self.sample_timer = 0
        else:
            self.sample_timer = 0
            self.state = State.RETURNING

    def _return(self, field: InfoField):
        """
        Retrace the recorded path back toward HQ one cell at a time. If the
        rover is retreating from a dead end, it also prunes the now-unused
        branch (erasing pixels) once it reaches a junction with another
        signal path or reaches HQ. On arrival at HQ, deliver any carried
        sample after 1 tick, then wait before resuming exploration.
        """
        prev_pos = (self.row, self.col)

        if self.path:
            self.move_to(*self.path.pop())
        else:
            self.recent.clear()

        if self.deadend_retreat:
            at_branch = field.count_signal_neighbors(self.row, self.col, exclude=prev_pos) >= 2
            if at_branch or self.at_hq():
                if (self.row, self.col)!=prev_pos:
                    field.erase_pixel(*prev_pos)
                    self.deadend_retreat = False
                    self.state = State.EXPLORING

        if self.at_hq():
            self.hq_timer += 1

            if self.hq_timer == 1 and self.samples_carried > 0:
                field.total_delivered += self.samples_carried
                self.samples_carried = 0

            if self.hq_timer >= int(self.del_time/self.dt/ROVER_TICK_RATIO):
                self.state = State.EXPLORING
                self.target = random.choice(field.poi) if field.poi else None
                self.hq_timer = 0
                self.path = []
        else:
            self.hq_timer = 0


# ── Engine ─────────────────────────────────────────────────────────────────
class Engine:
    """
    Owns the DEM, the shared InfoField, and all agents. Drives the
    simulation forward one tick at a time and accumulates visit-count
    heatmaps for post-hoc visualization.
    """
    def __init__(self, dem_path, hq_km):
        self.dem = DEM(dem_path)
        self.field = InfoField(self.dem, hq_km)
        self.drones = []
        self.rovers = []
        self.tick_n = 0
        self.rng = np.random.default_rng(42)
        # ── Cumulative Heatmap ──
        self.drone_visits = np.zeros(self.dem.shape, dtype=np.int32)
        self.rover_visits = np.zeros(self.dem.shape, dtype=np.int32)

    def spawn_poi(self, n):
        """Randomly place `n` POIs at uniformly sampled passable locations within the DEM extent."""
        for _ in range(n):
            x_km = random.uniform(
                self.dem.bounds.left / 1000,
                self.dem.bounds.right / 1000
            )
            y_km = random.uniform(
                self.dem.bounds.bottom / 1000,
                self.dem.bounds.top / 1000
            )
            r, c = self.dem.km_to_pixel(x_km, y_km)
            if self.dem.passable(r, c):
                self.field.mark_poi(r, c)
        print(f"[Engine] {n} POI generated.")

    def spawn_drones(self, n, scan_radius, bias):
        """Create `n` Drone agents at HQ, all starting in the EXPLORING state."""
        hq = self.field.hq
        for i in range(n):
            d = Drone(
                field=self.field,
                id=i,
                row=hq[0],
                col=hq[1],
                hq=hq,
                state=State.EXPLORING,
                scan_radius=scan_radius,
                bias=bias,
                rng=np.random.default_rng(i)
            )
            self.drones.append(d)
        print(f"[Engine] {n} Drones Created.")

    def spawn_rovers(self, n, memory=ROVER_MEMORY):
        """Create `n` Rover agents at HQ, all starting in the IDLE state (waiting for a drone trail)."""
        hq = self.field.hq

        for i in range(n):

            r = Rover(
                id=i,
                row=hq[0],
                col=hq[1],
                hq=hq,
                state=State.IDLE,
                memory=memory
            )

            self.rovers.append(r)

        print(f"[Engine] {n} Rovers Created (memory={memory}).")

    def step(self, dt):
        """Advance every agent by one tick, update visit heatmaps, then decay the field."""
        for d in self.drones:
            d.tick(self.field, self.dem)
            self.drone_visits[d.row, d.col] += 1
        for r in self.rovers:
            r.tick(self.field, self.dem)
            self.rover_visits[r.row, r.col] += 1
        self.tick_n += 1
        self.field.decay(dt=dt)
    # ── Plot Functions ─────────────────────────────────────────────────────────

# ─────────────── ANIMATION ───────────────

def animate(engine: Engine, steps: int, interval_ms: int, save_path: str | None):
    """
    Run and animate the simulation live using matplotlib's FuncAnimation:
    DEM as a base map, the stigmergic field as a semi-transparent overlay,
    and drone/rover markers with state labels updated every frame.
    Optionally saves the animation as a GIF if `save_path` is given.
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    dem = engine.dem
    b = dem.bounds

    # ── Extent in km per assi in scala reale ──────────────────
    extent_km = [
        b.left / 1000, b.right / 1000,
        b.bottom / 1000, b.top / 1000
    ]

    # DEM base (immobile) — palette topografica
    dem_im = ax.imshow(
        dem.data,
        cmap="gist_earth",
        vmin=np.nanmin(dem.data),
        vmax=np.nanmax(dem.data),
        interpolation="nearest",
        extent=extent_km,
        origin="upper"
    )

    # Colorbar elevazione, agganciata all'immagine DEM
    cbar = fig.colorbar(dem_im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Elevation [m]", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    # Heatmap stigmergica sovrapposta (aggiornata ogni frame)
    im = ax.imshow(
        engine.field.grid,
        cmap="viridis",
        vmin=EMPTY,
        vmax=POI_VAL,
        alpha=0.5,
        interpolation="nearest",
        extent=extent_km,
        origin="upper"
    )

    # ── Conversione pixel → km per i marker (HQ, droni, rover) ──
    def px_to_km(row, col):
        x_km = b.left/1000 + (col + 0.5) * dem.px_size_x/1000
        y_km = b.top/1000   - (row + 0.5) * dem.px_size_y/1000
        return x_km, y_km

    hq_x, hq_y = px_to_km(*engine.field.hq)
    ax.plot(hq_x, hq_y, "w*", ms=18, markeredgecolor="black")

    drone_markers = []
    drone_labels = []
    trail_lines = []
    for d in engine.drones:
        x, y = px_to_km(d.row, d.col)
        mk, = ax.plot(x, y, "r^", ms=8)
        lbl = ax.text(x, y, STATE_LABEL[d.state], color="red", fontsize=6, ha="center", va="bottom")
        trail, = ax.plot([], [], color="cyan", lw=0.5, alpha=0.5)
        drone_markers.append(mk)
        drone_labels.append(lbl)
        trail_lines.append(trail)

    rover_markers = []
    rover_labels = []
    for r in engine.rovers:
        x, y = px_to_km(r.row, r.col)
        mk, = ax.plot(x, y, "co", ms=7)
        lbl = ax.text(x, y, STATE_LABEL[r.state], color="green", fontsize=6, ha="center", va="bottom")
        rover_markers.append(mk)
        rover_labels.append(lbl)

    title = ax.set_title("", color="white")
    ax.set_xlabel("Longitude [km]", color="white")
    ax.set_ylabel("Latitude [km]", color="white")
    ax.tick_params(colors="white")

    def update(frame):
        """Advance the simulation by one step and refresh all plotted artists."""
        engine.step(DT)

        im.set_data(engine.field.grid)

        for mk, lbl, trail, d in zip(drone_markers, drone_labels, trail_lines, engine.drones):
            x, y = px_to_km(d.row, d.col)
            mk.set_data([x], [y])
            lbl.set_position((x, y))
            lbl.set_text(STATE_LABEL[d.state])
            if d.path:
                xs, ys = zip(*[px_to_km(p[0], p[1]) for p in d.path])
                trail.set_data(xs, ys)

        for mk, lbl, r in zip(rover_markers, rover_labels, engine.rovers):
            x, y = px_to_km(r.row, r.col)
            mk.set_data([x], [y])
            if r.samples_carried:
                mk.set_color("brown")
            else:
                mk.set_color("cyan")
            lbl.set_position((x, y))
            lbl.set_text(STATE_LABEL[r.state])

        title.set_text(
            f"Time {engine.tick_n*DT:.0f} s | "
            f"Drones: {len(engine.drones)} | "
            f"Rovers: {len(engine.rovers)} | "
            f"Delivered Samples: {engine.field.total_delivered}"
        )
        return [im, *drone_markers, *drone_labels, *trail_lines, *rover_markers, title]

    anim = FuncAnimation(
        fig,
        update,
        frames=steps,
        interval=interval_ms,
        blit=False,
        repeat=False
    )

    if save_path:
        anim.save(save_path, writer="pillow")

    plt.tight_layout()
    plt.show()
    return anim


# if __name__ == "__main__":
#     engine = Engine(DEM_PATH, HQ_KM)
#     engine.field = InfoField(engine.dem, HQ_KM, poi_km_list=POI_LIST_KM)
#     engine.spawn_drones(N_DRONES, scan_radius=SCAN_RADIUS, bias=DRONE_BIAS)
#     engine.spawn_rovers(N_ROVERS)

#     print(
#         f"INIT | POI={len(engine.field.poi)} | "
#         f"drones={len(engine.drones)} | "
#         f"rovers={len(engine.rovers)}"
#     )

#     animate(
#         engine,
#         steps=SIM_STEPS,
#         interval_ms=ANIM_INTERVAL_MS,
#         save_path=GIF_PATH if SAVE_GIF else None
#     )

#     print(f"Simulation finished, Samples Delivered:{engine.field.total_delivered}")



# ─────────────── FINAL STATE MULTIPLOT ───────────────
def plot_final_state(engine: Engine, dem: DEM, poi_batch_name: str = "POI_LIST",
                      output_dir: str = ".", dpi: int = 150):
    """
    Generates and saves three separate figures (light theme), with text sized
    to remain readable even when the figure is later scaled down to ~0.3
    \\linewidth in the LaTeX document. Uses manual margins instead of
    constrained_layout so fontsize is never silently compressed, and disables
    scientific-notation axis offsets (which don't scale with tick fontsize).

    Produces (and saves to `output_dir`):
      1. final_state_<name>.pdf     -- DEM + final POI states + HQ
      2. drone_coverage_<name>.pdf  -- DEM + drone visit-count heatmap
      3. rover_coverage_<name>.pdf  -- DEM + rover visit-count heatmap
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # ── Font/marker sizing ──────────────────────────────────────────
    TITLE_SIZE      = 85
    LABEL_SIZE      = 75
    TICK_SIZE       = 65
    LEGEND_SIZE     = 40
    HQ_MARKER_SIZE  = 55
    POI_MARKER_SIZE = 30
    LINEWIDTH       = 3.0

    def style_axes(ax):
        """Applies consistent large-font styling and disables sci-notation offset."""
        ax.tick_params(colors="black", labelsize=TICK_SIZE, width=LINEWIDTH, length=10)
        for spine in ax.spines.values():
            spine.set_linewidth(LINEWIDTH)
        ax.ticklabel_format(useOffset=False, style="plain", axis="both")

    b = dem.bounds
    extent_km = [b.left / 1000, b.right / 1000, b.bottom / 1000, b.top / 1000]

    map_width_km = extent_km[1] - extent_km[0]
    map_height_km = extent_km[3] - extent_km[2]
    map_aspect = map_height_km / map_width_km

    fig_width = 14.0
    fig_height = max(3.5, fig_width * map_aspect + 1.5)

    def px_to_km(row, col):
        x_km = b.left / 1000 + (col + 0.5) * dem.px_size_x / 1000
        y_km = b.top / 1000 - (row + 0.5) * dem.px_size_y / 1000
        return x_km, y_km

    hq_x, hq_y = px_to_km(*engine.field.hq)
    gray_vmin = np.nanmin(dem.data)
    gray_vmax = np.nanmax(dem.data) - 0.3 * (np.nanmax(dem.data) - np.nanmin(dem.data))

    # ── FIGURE 1: Final state (POIs + DEM) ──────────────────────────
    fig1, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig1.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig1.subplots_adjust(top=0.88, bottom=0.16, left=0.18, right=0.97)

    ax.imshow(
        dem.data, cmap="gist_earth",
        vmin=np.nanmin(dem.data), vmax=np.nanmax(dem.data),
        interpolation="nearest", extent=extent_km, origin="upper"
    )

    hq_handle, = ax.plot(
        hq_x, hq_y, "*", color="black", ms=HQ_MARKER_SIZE, markeredgecolor="white",
        markeredgewidth=2.0, zorder=5, label="HQ"
    )

    poi_active_handle = None
    poi_done_handle = None
    for (r, c) in engine.field.poi:
        x, y = px_to_km(r, c)
        val = engine.field.grid[r, c]
        if val <= POI_MIN:
            h = ax.plot(x, y, "o", color="white", ms=POI_MARKER_SIZE,
                        markeredgecolor="black", markeredgewidth=1.5,
                        zorder=4, label="Sampled POI")[0]
            poi_done_handle = h
        else:
            h = ax.plot(x, y, "o", color="darkorange", ms=POI_MARKER_SIZE,
                        markeredgecolor="black", markeredgewidth=1.5,
                        zorder=4, label="Active POI")[0]
            poi_active_handle = h

    handles = [h for h in [hq_handle, poi_active_handle, poi_done_handle] if h is not None]
    ax.legend(
        handles=handles, loc="upper right", facecolor="white",
        edgecolor="black", labelcolor="black", fontsize=LEGEND_SIZE,
        markerscale=1.0, framealpha=0.95
    )

    ax.set_title(
        f"Final State | Delivered: {engine.field.total_delivered} | "
        f"POIs: {len(engine.field.poi)}",
        color="black", fontsize=TITLE_SIZE, fontweight="bold"
    )
    ax.set_xlabel("Longitude [km]", color="black", fontsize=LABEL_SIZE)
    ax.set_ylabel("Latitude [km]", color="black", fontsize=LABEL_SIZE)
    style_axes(ax)

    fig1_path = os.path.join(output_dir, f"final_state_{poi_batch_name}.pdf")
    fig1.savefig(fig1_path, dpi=dpi, facecolor=fig1.get_facecolor())
    plt.close(fig1)

    # ── FIGURE 2: Drone coverage heatmap ─────────────────────────────
    fig2, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig2.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig2.subplots_adjust(top=0.88, bottom=0.16, left=0.18, right=0.97)

    ax.imshow(
        dem.data, cmap="gray_r",
        vmin=gray_vmin, vmax=gray_vmax,
        alpha=0.6, interpolation="nearest", extent=extent_km, origin="upper"
    )

    drone_visits_masked = engine.drone_visits.copy()
    drone_visits_masked[engine.field.hq] = 0
    for (r, c) in engine.field.poi:
        drone_visits_masked[r, c] = 0
    drone_hm = np.where(drone_visits_masked > 0, drone_visits_masked, np.nan)

    ax.imshow(
        drone_hm, cmap="inferno", interpolation="nearest",
        extent=extent_km, origin="upper", alpha=0.85
    )
    ax.plot(hq_x, hq_y, "*", color="black", ms=HQ_MARKER_SIZE - 8, markeredgecolor="white",
            markeredgewidth=2.0, zorder=5)

    ax.set_title("Drone Coverage Heatmap", color="black", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("Longitude [km]", color="black", fontsize=LABEL_SIZE)
    ax.set_ylabel("Latitude [km]", color="black", fontsize=LABEL_SIZE)
    style_axes(ax)

    fig2_path = os.path.join(output_dir, f"drone_coverage_{poi_batch_name}.pdf")
    fig2.savefig(fig2_path, dpi=dpi, facecolor=fig2.get_facecolor())
    plt.close(fig2)

    # ── FIGURE 3: Rover coverage heatmap ─────────────────────────────
    fig3, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig3.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig3.subplots_adjust(top=0.88, bottom=0.16, left=0.18, right=0.97)

    ax.imshow(
        dem.data, cmap="gray_r",
        vmin=gray_vmin, vmax=gray_vmax,
        alpha=0.6, interpolation="nearest", extent=extent_km, origin="upper"
    )

    rover_visits_masked = engine.rover_visits.copy()
    rover_visits_masked[engine.field.hq] = 0
    for (r, c) in engine.field.poi:
        rover_visits_masked[r, c] = 0
    rover_hm = np.where(rover_visits_masked > 0, rover_visits_masked, np.nan)

    ax.imshow(
        rover_hm, cmap="viridis", interpolation="nearest",
        extent=extent_km, origin="upper", alpha=0.85
    )
    ax.plot(hq_x, hq_y, "*", color="black", ms=HQ_MARKER_SIZE - 8, markeredgecolor="white",
            markeredgewidth=2.0, zorder=5)

    ax.set_title("Rover Coverage Heatmap", color="black", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("Longitude [km]", color="black", fontsize=LABEL_SIZE)
    ax.set_ylabel("Latitude [km]", color="black", fontsize=LABEL_SIZE)
    style_axes(ax)

    fig3_path = os.path.join(output_dir, f"rover_coverage_{poi_batch_name}.pdf")
    fig3.savefig(fig3_path, dpi=dpi, facecolor=fig3.get_facecolor())
    plt.close(fig3)

    print(f"[plot_final_state] Saved figures for {poi_batch_name}:")
    print(f"  {fig1_path}")
    print(f"  {fig2_path}")
    print(f"  {fig3_path}")

    return fig1_path, fig2_path, fig3_path

def plot_combined_overview(engine: Engine, dem: DEM, poi_batch_name: str = "POI_LIST",
                            output_dir: str = ".", dpi: int = 150):
    """
    Generates a single combined figure overlaying:
    - DEM (grayscale background)
    - Drone coverage heatmap (inferno)
    - Rover coverage heatmap (viridis)
    - POI markers (active / sampled) and HQ
    All information in one plot, with two separate colorbars for drone/rover density.
    Also auto-crops the view to the region actually explored by the agents.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # ── Font/marker sizing (tuned for ~0.3\linewidth in the final document) ──
    TITLE_SIZE      = 85
    LABEL_SIZE      = 65
    TICK_SIZE       = 65
    LEGEND_SIZE     = 30
    CBAR_LABEL_SIZE = 60
    CBAR_TICK_SIZE  = 50
    HQ_MARKER_SIZE  = 25
    POI_MARKER_SIZE = 18
    LINEWIDTH       = 3.0

    b = dem.bounds
    extent_km = [b.left / 1000, b.right / 1000, b.bottom / 1000, b.top / 1000]

    map_width_km = extent_km[1] - extent_km[0]
    map_height_km = extent_km[3] - extent_km[2]
    map_aspect = map_height_km / map_width_km

    fig_width = 16.0
    fig_height = max(4.0, fig_width * map_aspect + 2.0)

    def px_to_km(row, col):
        x_km = b.left / 1000 + (col + 0.5) * dem.px_size_x / 1000
        y_km = b.top / 1000 - (row + 0.5) * dem.px_size_y / 1000
        return x_km, y_km

    hq_x, hq_y = px_to_km(*engine.field.hq)
    gray_vmin = np.nanmin(dem.data)
    gray_vmax = np.nanmax(dem.data) - 0.3 * (np.nanmax(dem.data) - np.nanmin(dem.data))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(top=0.90, bottom=0.16, left=0.16, right=0.86)

    # ── Base layer: DEM in grayscale ──
    ax.imshow(
        dem.data, cmap="gray_r",
        vmin=gray_vmin, vmax=gray_vmax,
        alpha=0.5, interpolation="nearest", extent=extent_km, origin="upper"
    )

    # ── Drone coverage overlay ──
    drone_visits_masked = engine.drone_visits.copy()
    drone_visits_masked[engine.field.hq] = 0
    for (r, c) in engine.field.poi:
        drone_visits_masked[r, c] = 0
    drone_hm = np.where(drone_visits_masked > 0, drone_visits_masked, np.nan)

    im_drone = ax.imshow(
        drone_hm, cmap="inferno", interpolation="nearest",
        extent=extent_km, origin="upper", alpha=0.55, zorder=2
    )

    # ── Rover coverage overlay ──
    rover_visits_masked = engine.rover_visits.copy()
    rover_visits_masked[engine.field.hq] = 0
    for (r, c) in engine.field.poi:
        rover_visits_masked[r, c] = 0
    rover_hm = np.where(rover_visits_masked > 0, rover_visits_masked, np.nan)

    im_rover = ax.imshow(
        rover_hm, cmap="viridis", interpolation="nearest",
        extent=extent_km, origin="upper", alpha=0.55, zorder=3
    )

    # ── HQ marker ──
    hq_handle, = ax.plot(
        hq_x, hq_y, "*", color="black", ms=HQ_MARKER_SIZE, markeredgecolor="white",
        markeredgewidth=2.0, zorder=6, label="HQ"
    )

    # ── POI markers ──
    poi_active_handle = None
    poi_done_handle = None
    for (r, c) in engine.field.poi:
        x, y = px_to_km(r, c)
        val = engine.field.grid[r, c]
        if val <= POI_MIN:
            h = ax.plot(x, y, "o", color="white", ms=POI_MARKER_SIZE,
                        markeredgecolor="black", markeredgewidth=1.5,
                        zorder=5, label="Sampled POI")[0]
            poi_done_handle = h
        else:
            h = ax.plot(x, y, "o", color="red", ms=POI_MARKER_SIZE,
                        markeredgecolor="black", markeredgewidth=1.5,
                        zorder=5, label="Active POI")[0]
            poi_active_handle = h

    handles = [h for h in [hq_handle, poi_active_handle, poi_done_handle] if h is not None]
    ax.legend(
        handles=handles, loc="upper right", facecolor="white",
        edgecolor="black", labelcolor="black", fontsize=LEGEND_SIZE,
        markerscale=1.0, framealpha=0.95
    )

    ax.set_title(
        f"Combined Overview | Delivered: {engine.field.total_delivered} | "
        f"POIs: {len(engine.field.poi)}",
        color="black", fontsize=TITLE_SIZE, fontweight="bold"
    )
    ax.set_xlabel("Longitude [km]", color="black", fontsize=LABEL_SIZE)
    ax.set_ylabel("Latitude [km]", color="black", fontsize=LABEL_SIZE)
    ax.tick_params(colors="black", labelsize=TICK_SIZE, width=LINEWIDTH, length=10)
    for spine in ax.spines.values():
        spine.set_linewidth(LINEWIDTH)
    ax.ticklabel_format(useOffset=False, style="plain", axis="both")


    # ── Two separated colorbars: drone (inferno) + rover (viridis) ──
    # Using make_axes_locatable-style manual axes for full control over
    # spacing, so the two colorbars don't crowd each other or overlap
    # with the labels.
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax_drone = divider.append_axes("right", size="4%", pad=0.6)
    cax_rover = divider.append_axes("right", size="4%", pad=1.6)

    cbar_drone = fig.colorbar(im_drone, cax=cax_drone)
    cbar_drone.set_label("Drone visits", color="black", fontsize=CBAR_LABEL_SIZE, labelpad=20)
    cbar_drone.ax.tick_params(colors="black", labelsize=CBAR_TICK_SIZE)
    cbar_drone.outline.set_linewidth(LINEWIDTH)

    cbar_rover = fig.colorbar(im_rover, cax=cax_rover)
    cbar_rover.set_label("Rover visits", color="black", fontsize=CBAR_LABEL_SIZE, labelpad=20)
    cbar_rover.ax.tick_params(colors="black", labelsize=CBAR_TICK_SIZE)
    cbar_rover.outline.set_linewidth(LINEWIDTH)
    # ── Crop automatico sulla zona esplorata ──────────────────────
    visited = (engine.drone_visits > 0) | (engine.rover_visits > 0)
    rows, cols = np.where(visited)
    if len(rows) > 0:
        pad = 15  # margine in pixel
        r0, r1 = max(0, rows.min()-pad), min(dem.shape[0]-1, rows.max()+pad)
        c0, c1 = max(0, cols.min()-pad), min(dem.shape[1]-1, cols.max()+pad)
        x0 = b.left/1000 + c0 * dem.px_size_x/1000
        x1 = b.left/1000 + c1 * dem.px_size_x/1000
        y0 = b.top/1000  - r1 * dem.px_size_y/1000
        y1 = b.top/1000  - r0 * dem.px_size_y/1000
        ax.set_xlim(x0-0.1, x1+0.1)
        ax.set_ylim(y0-0.1, y1+0.1)

    combined_path = os.path.join(output_dir, f"combined_overview_{poi_batch_name}.pdf")
    fig.savefig(combined_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"[plot_combined_overview] Saved figure to {combined_path}")
    return combined_path

def all_samples_delivered(engine: Engine) -> bool:
    """
    True when every POI is exhausted (grid value <= POI_MIN) and no rover
    is still carrying a sample back to HQ.
    """
    all_poi_exhausted = all(
        engine.field.grid[r, c] <= POI_MIN for (r, c) in engine.field.poi
    )
    no_samples_in_transit = all(
        r.samples_carried == 0 for r in engine.rovers
    )
    return all_poi_exhausted and no_samples_in_transit

def run_and_combine_batches(dem_path, hq_km, poi_lists_km, batch_labels=None,
                             steps=SIM_STEPS, output_dir=".", dpi=150,
                             output_name="combined_overview_all"):
    """
    Runs one independent simulation per POI batch, then overlays the
    resulting drone/rover coverage heatmaps and POI states into a single
    combined figure. Each batch's engine is fully separate (its own field,
    drones, and rovers), so results are summed only for visualization.
    """
    import os
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    os.makedirs(output_dir, exist_ok=True)

    if batch_labels is None:
        batch_labels = [f"Batch {i+1}" for i in range(len(poi_lists_km))]

    # ── Run one engine per batch ──
    engines = []
    for i, poi_km_list in enumerate(poi_lists_km):
        eng = Engine(dem_path, hq_km)
        eng.field = InfoField(eng.dem, hq_km, poi_km_list=poi_km_list)
        eng.spawn_drones(N_DRONES, SCAN_RADIUS, DRONE_BIAS)
        eng.spawn_rovers(N_ROVERS)

        for tick in range(steps):
            eng.step(DT)
            if all_samples_delivered(eng):
                print(f"[run_and_combine_batches] {batch_labels[i]} delivered "
                      f"at tick {tick} (t={tick*DT:.0f}s).")
                break

        engines.append(eng)
        print(f"[run_and_combine_batches] {batch_labels[i]} finished | "
              f"Delivered: {eng.field.total_delivered}")

    dem = engines[0].dem  # same DEM/crop for all runs

    # ── Sizing ──
    TITLE_SIZE      = 20
    LABEL_SIZE      = 75
    TICK_SIZE       = 65
    LEGEND_SIZE     = 20
    CBAR_LABEL_SIZE = 50
    CBAR_TICK_SIZE  = 40
    HQ_MARKER_SIZE  = 25
    POI_MARKER_SIZE = 15
    LINEWIDTH       = 3.0

    b = dem.bounds
    extent_km = [b.left / 1000, b.right / 1000, b.bottom / 1000, b.top / 1000]
    map_width_km = extent_km[1] - extent_km[0]
    map_height_km = extent_km[3] - extent_km[2]
    map_aspect = map_height_km / map_width_km

    fig_width = 16.0
    fig_height = 6.0

    def px_to_km(row, col):
        x_km = b.left / 1000 + (col + 0.5) * dem.px_size_x / 1000
        y_km = b.top / 1000 - (row + 0.5) * dem.px_size_y / 1000
        return x_km, y_km

    hq_x, hq_y = px_to_km(*engines[0].field.hq)
    gray_vmin = np.nanmin(dem.data)
    gray_vmax = np.nanmax(dem.data) - 0.3 * (np.nanmax(dem.data) - np.nanmin(dem.data))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(top=0.90, bottom=0.16, left=0.16, right=0.78)

    # ── Base: DEM grayscale ──
    ax.imshow(
        dem.data, cmap="gray",
        vmin=gray_vmin+0.1, vmax=gray_vmax,
        alpha=0.5, interpolation="nearest", extent=extent_km, origin="upper"
    )

    # ── Sum drone/rover visits across all three engines ──
    drone_total = np.zeros(dem.shape, dtype=np.float64)
    rover_total = np.zeros(dem.shape, dtype=np.float64)

    for eng in engines:
        dv = eng.drone_visits.copy().astype(np.float64)
        rv = eng.rover_visits.copy().astype(np.float64)
        dv[eng.field.hq] = 0
        rv[eng.field.hq] = 0
        for (r, c) in eng.field.poi:
            dv[r, c] = 0
            rv[r, c] = 0
        drone_total += dv
        rover_total += rv

    drone_hm = np.where(drone_total > 0, drone_total, np.nan)
    rover_hm = np.where(rover_total > 0, rover_total, np.nan)
    
    
    # Espande ogni pixel su un intorno 3x3 (rende la heatmap più leggibile
    # a bassa risoluzione, evitando puntini isolati poco visibili)
    drone_hm = maximum_filter(drone_hm, size=11)
    rover_hm = maximum_filter(rover_hm, size=11)
    
    im_drone = ax.imshow(
        drone_hm, cmap="Reds", interpolation="nearest",
        extent=extent_km, origin="upper", alpha=0.55, zorder=2
    )

    im_rover = ax.imshow(
        rover_hm, cmap="Blues", interpolation="nearest",
        extent=extent_km, origin="upper", alpha=0.55, zorder=3
    )

    # ── HQ marker (shared across all engines) ──
    hq_handle, = ax.plot(
        hq_x, hq_y, "*", color="black", ms=HQ_MARKER_SIZE, markeredgecolor="white",
        markeredgewidth=2.0, zorder=6, label="HQ"
    )

    # ── POI markers per batch, colored, filled if sampled ──
    batch_colors = ["darkorange", "royalblue", "crimson"]
    handles = [hq_handle]

    for i, (eng, label) in enumerate(zip(engines, batch_labels)):
        color = batch_colors[i % len(batch_colors)]
        for (r, c) in eng.field.poi:
            x, y = px_to_km(r, c)
            val = eng.field.grid[r, c]
            face = "white" if val <= POI_MIN else color
            ax.plot(x, y, "o", color=face, ms=POI_MARKER_SIZE,
                    markeredgecolor=color, markeredgewidth=3.0, zorder=5)
        proxy, = ax.plot([], [], "o", color=color, ms=POI_MARKER_SIZE,
                          markeredgecolor=color, markeredgewidth=3.0, label=label)
        handles.append(proxy)

    total_delivered = sum(eng.field.total_delivered for eng in engines)
    ax.set_title(
        f"Combined Overview -- All Batches | Total Delivered: {total_delivered}",
        color="black", fontsize=TITLE_SIZE, fontweight="bold"
    )
    ax.set_xlabel("Longitude [km]", color="black", fontsize=LABEL_SIZE)
    ax.set_ylabel("Latitude [km]", color="black", fontsize=LABEL_SIZE)
    ax.tick_params(colors="black", labelsize=TICK_SIZE, width=LINEWIDTH, length=10)
    for spine in ax.spines.values():
        spine.set_linewidth(LINEWIDTH)
    ax.ticklabel_format(useOffset=False, style="plain", axis="both")
    ax.set_aspect("equal")

    # ── Colorbars ──
    divider = make_axes_locatable(ax)
    cax_drone = divider.append_axes("right", size="4%", pad=0.6)
    cax_rover = divider.append_axes("right", size="4%", pad=1.6)

    cbar_drone = fig.colorbar(im_drone, cax=cax_drone)
    cbar_drone.set_label("Drone visits", color="black", fontsize=CBAR_LABEL_SIZE, labelpad=20)
    cbar_drone.ax.tick_params(colors="black", labelsize=CBAR_TICK_SIZE)
    cbar_drone.outline.set_linewidth(LINEWIDTH)

    cbar_rover = fig.colorbar(im_rover, cax=cax_rover)
    cbar_rover.set_label("Rover visits", color="black", fontsize=CBAR_LABEL_SIZE, labelpad=20)
    cbar_rover.ax.tick_params(colors="black", labelsize=CBAR_TICK_SIZE)
    cbar_rover.outline.set_linewidth(LINEWIDTH)

    combined_path = os.path.join(output_dir, f"{output_name}.pdf")
    # fig.savefig(combined_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"[run_and_combine_batches] Saved figure to {combined_path}")
    return combined_path, engines


# if __name__ == "__main__":
#     # ── Single full simulation run using the real Perseverance POI batch 3 ──
#     engine = Engine(DEM_PATH, HQ_KM)
#     engine.field = InfoField(engine.dem, HQ_KM, poi_km_list=POI_LIST_KM)   # nessun POI iniziale
#     engine.spawn_drones(N_DRONES, scan_radius=SCAN_RADIUS, bias=DRONE_BIAS)
#     engine.spawn_rovers(N_ROVERS)

#     for tick in range(SIM_STEPS):
#         engine.step(DT)

#         if engine.field.poi and all_samples_delivered(engine):
#             print(f"[Main] All samples delivered at tick {tick} "
#                   f"(t={tick * DT:.0f} s). Stopping simulation early.")
#             break

#     print(f"Simulation finished, Samples Delivered: {engine.field.total_delivered}")
#     plot_combined_overview(engine, engine.dem, poi_batch_name="Random", output_dir="results")
    


poi_batch_1_km = [latlon_to_dem_km(DEM_PATH, lat, lon) for lat, lon in POI_LIST_1]
poi_batch_2_km = [latlon_to_dem_km(DEM_PATH, lat, lon) for lat, lon in POI_LIST_2]
poi_batch_3_km = [latlon_to_dem_km(DEM_PATH, lat, lon) for lat, lon in POI_LIST_3]

# combined_path, engines = run_and_combine_batches(
#     DEM_PATH, HQ_KM,
#     poi_lists_km=[
#         [latlon_to_dem_km(DEM_PATH, lat, lon) for lat, lon in POI_LIST_1],
#         [latlon_to_dem_km(DEM_PATH, lat, lon) for lat, lon in POI_LIST_2],
#         [latlon_to_dem_km(DEM_PATH, lat, lon) for lat, lon in POI_LIST_3],
#     ],
#     batch_labels=["Batch 1 -- Crater Floor", "Batch 2 -- Delta Front", "Batch 3 -- Margin"],
#     output_dir="results"
# )

    
# ─────────────── METRICS TRACKING ───────────────

class Metrics:
    """
    Collects per-tick statistics from an Engine over the course of a run
    (delivered samples, POIs discovered, per-state tick counts, coverage,
    dead-end events) and produces a summary dict at the end.
    """
    def __init__(self, engine):
        self.engine = engine
        self.delivered_over_time = []
        self.poi_discovered_over_time = []
        self.state_counts = defaultdict(lambda: defaultdict(int))  # {agent_type: {state: ticks}}
        self.first_delivery_tick = None
        self.deadend_count = 0
        self.visited_cells = set()   # per coverage (drones)
        self.poi_found_tick = {}     # {poi_coord: tick_scoperto}
        self.poi_delivered_tick = [] # latenza consegna

    def sample(self, tick):
        """Record one tick's worth of engine state into the running metrics."""
        eng = self.engine
        self.delivered_over_time.append(eng.field.total_delivered)
        self.poi_discovered_over_time.append(len(eng.field.poi))

        if self.first_delivery_tick is None and eng.field.total_delivered > 0:
            self.first_delivery_tick = tick

        for d in eng.drones:
            self.state_counts["drone"][d.state.name] += 1
            self.visited_cells.add((d.row, d.col))
            for poi in eng.field.poi:
                if poi not in self.poi_found_tick:
                    self.poi_found_tick[poi] = tick

        for r in eng.rovers:
            self.state_counts["rover"][r.state.name] += 1
            if getattr(r, "deadend_retreat", False):
                self.deadend_count += 1

    def summary(self, dt, dem):
        """Aggregate the collected samples into a single summary dict
        (throughput, coverage fraction, time to first delivery, per-state
        fraction of ticks spent by drones/rovers, etc.)."""
        eng = self.engine
        total_ticks = len(self.delivered_over_time)
        total_time_s = total_ticks * dt

        passable_cells = np.sum(~np.isnan(dem.data))
        coverage = len(self.visited_cells) / passable_cells

        out = {
            "total_delivered": eng.field.total_delivered,
            "throughput_per_hour": eng.field.total_delivered / (total_time_s / 3600) if total_time_s > 0 else 0,
            "time_to_first_delivery_s": self.first_delivery_tick * dt if self.first_delivery_tick else None,
            "poi_discovered": len(eng.field.poi),
            "coverage_fraction": coverage,
            "deadend_events": self.deadend_count,
        }

        for agent_type, counts in self.state_counts.items():
            total = sum(counts.values())
            out[f"{agent_type}_state_frac"] = {k: v / total for k, v in counts.items()}

        return out

def run_headless_with_metrics(seed, steps=SIM_STEPS, poi_list=POI_LIST_KM,
                               early_stop=True, verbose=True):
    """
    Run one full simulation (no plotting) with a fixed random seed,
    collecting Metrics along the way. Optionally stops early once all
    samples have been delivered. Returns (summary_dict, Metrics instance).
    """
    np.random.seed(seed)
    random.seed(seed)
    engine = Engine(DEM_PATH, HQ_KM)
    engine.field = InfoField(engine.dem, HQ_KM, poi_km_list=poi_list)   # <-- fix
    engine.spawn_drones(N_DRONES, SCAN_RADIUS, DRONE_BIAS)
    engine.spawn_rovers(N_ROVERS)

    metrics = Metrics(engine)
    stopped_early = False
    stop_tick = steps

    for tick in range(steps):
        engine.step(DT)
        metrics.sample(tick)

        if early_stop and all_samples_delivered(engine):
            stopped_early = True
            stop_tick = tick
            if verbose:
                print(f"[run_headless_with_metrics] seed={seed} | "
                      f"All samples delivered at tick {tick} "
                      f"(t={tick * DT:.0f} s). Stopping simulation early.")
            break

    summary = metrics.summary(DT, engine.dem)
    summary["stopped_early"] = stopped_early
    summary["stop_tick"] = stop_tick
    summary["stop_time_s"] = stop_tick * DT

    return summary, metrics

def multi_run(n_seeds=10, steps=SIM_STEPS, early_stop=True):
    """
    Run `n_seeds` independent headless simulations (seeds 0..n_seeds-1),
    print aggregate statistics (mean +/- std of delivered samples,
    throughput, coverage, early-stop rate), and return the list of
    per-run summary dicts.
    """
    results = []
    for seed in range(n_seeds):
        summary, _ = run_headless_with_metrics(seed, steps, early_stop=early_stop)
        results.append(summary)

    delivered = np.array([r["total_delivered"] for r in results])
    throughput = np.array([r["throughput_per_hour"] for r in results])
    coverage = np.array([r["coverage_fraction"] for r in results])
    stop_times = np.array([r["stop_time_s"] for r in results])
    n_early = sum(r["stopped_early"] for r in results)

    print(f"Delivered:   {delivered.mean():.2f} ± {delivered.std():.2f}")
    print(f"Throughput:  {throughput.mean():.2f} ± {throughput.std():.2f} campioni/h")
    print(f"Coverage:    {coverage.mean():.2%} ± {coverage.std():.2%}")
    print(f"Early stops: {n_early}/{len(results)} run "
          f"(tempo medio: {stop_times.mean():.0f}s ± {stop_times.std():.0f}s)")

    return results


# if __name__ == "__main__":
#     summary, metrics = run_headless_with_metrics(seed=0)
#     print(summary)
#     results = multi_run(n_seeds=10)


# ─────────────── BASIC MAP PLOTTING ───────────────

def plot_dem_with_batches(dem_path, hq_km, poi_batches, batch_labels=None,
                           output_path="dem_overview.pgf", dpi=150):
    """
    Plots the DEM with the HQ (lander) location and multiple POI batches,
    each shown with a distinct color/marker. Light theme.
    Used to illustrate the experimental setup before running a simulation.
    """
    dem = DEM(dem_path)
    b = dem.bounds
    extent_km = [b.left / 1000, b.right / 1000, b.bottom / 1000, b.top / 1000]

    if batch_labels is None:
        batch_labels = [f"Batch {i+1}" for i in range(len(poi_batches))]

    # ── Sizing ──
    TITLE_SIZE  = 22*1.5
    LABEL_SIZE  = 22*1.5
    TICK_SIZE   = 20*1.5
    LEGEND_SIZE = 18*1.5
    HQ_MARKER_SIZE  = 28*1.5
    POI_MARKER_SIZE = 11*1.5

    map_width_km = extent_km[1] - extent_km[0]
    map_height_km = extent_km[3] - extent_km[2]
    map_aspect = map_height_km / map_width_km
    fig_width = 12.0
    fig_height = max(3.5, fig_width * map_aspect + 1.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.imshow(
        dem.data, cmap="gist_earth",
        vmin=np.nanmin(dem.data), vmax=np.nanmax(dem.data),
        interpolation="nearest", extent=extent_km, origin="upper"
    )

    # ── HQ marker ──
    hq_x, hq_y = hq_km
    hq_handle, = ax.plot(
        hq_x, hq_y, "*", color="black", ms=HQ_MARKER_SIZE, markeredgecolor="white",
        markeredgewidth=1.5, zorder=5, label="HQ (Lander)"
    )

    # ── POI batches, one color per batch ──
    batch_colors = ["darkorange", "royalblue", "crimson", "seagreen", "purple"]
    handles = [hq_handle]

    for i, (batch, label) in enumerate(zip(poi_batches, batch_labels)):
        color = batch_colors[i % len(batch_colors)]
        xs = [p[0] for p in batch]
        ys = [p[1] for p in batch]
        h = ax.plot(
            xs, ys, "o", color=color, ms=POI_MARKER_SIZE,
            markeredgecolor="black", markeredgewidth=1,
            zorder=4, label=label
        )[0]
        handles.append(h)

    ax.legend(
        handles=handles, loc="upper right", facecolor="white",
        edgecolor="black", labelcolor="black", fontsize=LEGEND_SIZE,
        markerscale=1.2
    )

    ax.set_title("Simulation Setup: HQ and Target POI Batches",
                  color="black", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("Longitude [km]", color="black", fontsize=LABEL_SIZE)
    ax.set_ylabel("Latitude [km]", color="black", fontsize=LABEL_SIZE)
    ax.tick_params(colors="black", labelsize=TICK_SIZE)

    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[plot_dem_with_batches] Saved figure to {output_path}")

    return output_path

poi_batch_1_km = [latlon_to_dem_km(DEM_PATH, lat, lon) for lat, lon in POI_LIST_1]
poi_batch_2_km = [latlon_to_dem_km(DEM_PATH, lat, lon) for lat, lon in POI_LIST_2]
poi_batch_3_km = [latlon_to_dem_km(DEM_PATH, lat, lon) for lat, lon in POI_LIST_3]


# ─────────────── MULTI-BATCH METRICS FOR REPORT TABLE ───────────────


def run_metrics_all_batches(poi_batches_km, batch_labels, n_seeds=8,
                             steps=SIM_STEPS, early_stop=True,
                             output_dir="results", verbose=True):
    """
    Runs `n_seeds` independent simulations for each POI batch, collects
    the full Metrics.summary() dict for every run, and returns:
      - raw_df: one row per (batch, seed) run, all metrics flattened
      - agg_df: one row per batch, mean ± std of every numeric metric

    Saves both as CSV in output_dir, ready to be turned into a LaTeX table.
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_rows = []

    for batch_label, poi_list in zip(batch_labels, poi_batches_km):
        for seed in range(n_seeds):
            summary, _ = run_headless_with_metrics(
                seed=seed, steps=steps, poi_list=poi_list,
                early_stop=early_stop, verbose=verbose
            )

            row = {"batch": batch_label, "seed": seed}

            # Flatten summary dict (state fractions are nested dicts)
            for k, v in summary.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        row[f"{k}_{sub_k}"] = sub_v
                else:
                    row[k] = v

            raw_rows.append(row)

            if verbose:
                print(f"[run_metrics_all_batches] {batch_label} | seed={seed} "
                      f"| delivered={summary['total_delivered']} "
                      f"| throughput/h={summary['throughput_per_hour']:.2f}")

    raw_df = pd.DataFrame(raw_rows)

    # ── Aggregate: mean ± std per batch, only numeric columns ──
    numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "seed"]

    agg_rows = []
    for batch_label in batch_labels:
        sub = raw_df[raw_df["batch"] == batch_label]
        row = {"batch": batch_label, "n_seeds": len(sub)}

        for col in numeric_cols:
            row[f"{col}_mean"] = sub[col].mean()
            row[f"{col}_std"] = sub[col].std()

        # Fraction of runs that stopped early (bool column not in numeric_cols)
        if "stopped_early" in sub.columns:
            row["stopped_early_frac"] = sub["stopped_early"].mean()

        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)

    raw_path = os.path.join(output_dir, "metrics_raw_all_batches.csv")
    agg_path = os.path.join(output_dir, "metrics_summary_all_batches.csv")

    raw_df.to_csv(raw_path, index=False)
    agg_df.to_csv(agg_path, index=False)

    print(f"[run_metrics_all_batches] Saved raw data to {raw_path}")
    print(f"[run_metrics_all_batches] Saved aggregated summary to {agg_path}")

    return raw_df, agg_df


# if __name__ == "__main__":
#     poi_batches_km = [poi_batch_1_km, poi_batch_2_km, poi_batch_3_km]
#     batch_labels = ["Crater Floor", "Delta Front", "Margin"]

#     raw_df, agg_df = run_metrics_all_batches(
#         poi_batches_km=poi_batches_km,
#         batch_labels=batch_labels,
#         n_seeds=8,           # metti 5-10 come preferisci
#         steps=SIM_STEPS,
#         early_stop=True,
#         output_dir="results",
#         verbose=True
#     )

#     print("\n─── SUMMARY (mean ± std) ───")
#     print(agg_df.to_string(index=False))

# raw_df, agg_df = run_metrics_all_batches(
#     poi_batches_km=[[]],
#     batch_labels=["Exploration"],
#     n_seeds=8,
#     steps=SIM_STEPS,
#     early_stop=False,   
#     output_dir="results"
# )


# plot_dem_with_batches(
#     DEM_PATH,
#     HQ_KM,
#     poi_batches=[poi_batch_1_km, poi_batch_2_km, poi_batch_3_km],
#     batch_labels=["Batch 1 — Crater Floor", "Batch 2 — Delta Front", "Batch 3 — Margin"],
#     output_path="results/dem_overview.png"
# )
