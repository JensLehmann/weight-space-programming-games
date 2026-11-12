# Writing a compact neural game-playing agent

## The game: official ARC-AGI-3 game `dc22` (simulator source, redacted)

You get the game's simulator source: state, transitions, goal and loss rules, and rendering, written against the `arcengine` package. The game has 6 levels. Levels 2, 4, 6 (1-based) are withheld for evaluation and appear as `WITHHELD`: their entries are replaced, code that only they use raises `LookupError('level withheld for evaluation')`, and sprite templates that only they use are omitted. Levels 1, 3, 5 are shown in full. The actor will also be tested on states a few actions into a level. There is one canonical rendering (no renderer templates or palette variants). Identifiers were obfuscated by the game's authors. The engine excerpt is unmodified `arcengine` source, restricted to the engine classes the game imports or that the action/render loop needs; within a class, to the members the game executes, overrides or unambiguously names, the action/render loop, and the engine members those call; omitted members are listed by name.

`arcengine/enums.py` (excerpt):
```python
MAX_REASONING_BYTES = 16 * 1024  # 16 KB guard-rail


class BlockingMode(Enum):
    """Enum defining different collision detection behaviors for sprites."""

    NOT_BLOCKED = auto()  # No collision detection
    BOUNDING_BOX = auto()  # Simple rectangular collision detection
    PIXEL_PERFECT = auto()  # Precise pixel-level collision detection


class InteractionMode(Enum):
    """Enum defining how a sprite interacts with the game world visually and physically."""

    TANGIBLE = auto()  # Visible and can be collided with
    INTANGIBLE = auto()  # Visible but cannot be collided with (ghost-like)
    INVISIBLE = auto()  # Not visible but can be collided with (invisible wall)
    REMOVED = auto()  # Not visible and cannot be collided with (effectively removed)


class GameState(str, Enum):
    NOT_PLAYED = "NOT_PLAYED"
    NOT_FINISHED = "NOT_FINISHED"
    WIN = "WIN"
    GAME_OVER = "GAME_OVER"


class GameAction(Enum):
    RESET = (0, SimpleAction)
    ACTION1 = (1, SimpleAction)
    ACTION2 = (2, SimpleAction)
    ACTION3 = (3, SimpleAction)
    ACTION4 = (4, SimpleAction)
    ACTION5 = (5, SimpleAction)
    ACTION6 = (6, ComplexAction)
    ACTION7 = (7, SimpleAction)

    action_type: Union[Type[SimpleAction], Type[ComplexAction]]
    action_data: Union[SimpleAction | ComplexAction]

    def __init__(
        self,
        action_id: int,
        action_type: Union[Type[SimpleAction], Type[ComplexAction]],
    ) -> None:
        self._value_ = action_id
        self.action_type = action_type
        self.action_data = action_type()

    def is_simple(self) -> bool:
        return self.action_type is SimpleAction

    def is_complex(self) -> bool:
        return self.action_type is ComplexAction

    def validate_data(self, data: dict[str, Any]) -> bool:
        """Raise exception on invalid parse of incoming JSON data."""
        self.action_type.model_validate(data)
        return True

    def set_data(self, data: dict[str, Any]) -> Union[SimpleAction | ComplexAction]:
        self.action_data = self.action_type(**data)
        return self.action_data

    @classmethod
    def from_id(cls, action_id: int) -> GameAction:
        for action in cls:
            if action.value == action_id:
                return action
        raise ValueError(f"No GameAction with id {action_id}")

    @classmethod
    def from_name(cls, name: str) -> "GameAction":
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"No GameAction with name '{name}'")

    @classmethod
    def all_simple(cls) -> list[GameAction]:
        return [a for a in cls if a.is_simple()]

    @classmethod
    def all_complex(cls) -> list[GameAction]:
        return [a for a in cls if a.is_complex()]


class ActionInput(BaseModel):
    id: GameAction = GameAction.RESET
    data: dict[str, Any] = {}
    reasoning: Optional[Any] = Field(default=None, description="Opaque client-supplied blob; stored & echoed back verbatim.")

    # Optional size / serialisability guard
    @field_validator("reasoning")
    @classmethod
    def _check_reasoning(cls, v: Any) -> Any:
        if v is None:
            return v  # field omitted → fine
        try:
            raw = json.dumps(v, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError):
            raise ValueError("reasoning must be JSON-serialisable")
        if len(raw) > MAX_REASONING_BYTES:
            raise ValueError(f"reasoning exceeds {MAX_REASONING_BYTES} bytes")
        return v


class PlaceableArea:
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    x_scale: int = 1
    y_scale: int = 1

    def __init__(self, x: int, y: int, width: int, height: int, x_scale: int = 1, y_scale: int = 1):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.x_scale = x_scale
        self.y_scale = y_scale
```

`arcengine/sprites.py` (excerpt):
```python
def _downscale_mode(arr: np.ndarray, factor: int) -> np.ndarray:
    """
    Nearest-neighbor style down-scaling for palette images.
    For each non-overlapping block it keeps the dominant color
    (mode), breaking ties by the highest palette index.
    Transparent pixels (-1) are ignored, but if all pixels are
    transparent, the block remains transparent.

    Parameters
    ----------
    arr : 2-D np.ndarray
        Input image of dtype int8 / uint8 holding palette indices.
        Transparency is denoted by -1.
    factor : int
        The integer scale factor (e.g. 2 turns 64×64 → 32×32).

    Returns
    -------
    np.ndarray
        The down-scaled image, same dtype as the input.
    """
    H, W = arr.shape
    if H % factor != 0 or W % factor != 0:
        raise ValueError(f"Array dimensions ({H}, {W}) must be divisible by scale factor {factor}")

    # Step 1: split into blocks → shape (out_h, out_w, factor, factor)
    blocks = arr.reshape(H // factor, factor, -1, factor).swapaxes(1, 2)
    blocks = blocks.reshape(-1, factor * factor)

    # Step 2: find dominant color for each block
    out = np.empty(len(blocks), dtype=arr.dtype)

    for i, blk in enumerate(blocks):
        non_transparent = blk[blk != -1]
        transparent = blk[blk < 0]
        # Check to see if this is majority transparent
        if transparent.size > non_transparent.size:
            out[i] = -1  # if all values are -1, keep block transparent
        else:
            cnts = np.bincount(non_transparent.astype(np.int16))
            max_count = cnts.max()
            max_indices = np.where(cnts == max_count)[0]
            out[i] = max_indices[-1]  # break ties by highest index

    # Step 3: reshape to 2-D image
    return out.reshape(H // factor, W // factor)


def _interaction_mode_from(visible: bool, collidable: bool) -> InteractionMode:
    if visible and collidable:
        return InteractionMode.TANGIBLE
    elif visible and not collidable:
        return InteractionMode.INTANGIBLE
    elif not visible and collidable:
        return InteractionMode.INVISIBLE
    else:
        return InteractionMode.REMOVED


class Sprite:
    VALID_ROTATIONS = {0, 90, 180, 270}
    pixels: ndarray
    _name: str
    _x: int
    _y: int
    _layer: int
    _rotation: int
    _mirror_ud: bool
    _mirror_lr: bool
    _scale: int  # Use set_scale to validate scale factor
    _blocking: BlockingMode
    _interaction: InteractionMode
    _tags: list[str]
    def __init__(
        self,
        pixels: List[List[int]] | np.ndarray,
        name: Optional[str] = None,
        x: int = 0,
        y: int = 0,
        layer: int = 0,
        scale: int = 1,
        rotation: int = 0,
        mirror_ud: bool = False,
        mirror_lr: bool = False,
        blocking: BlockingMode = BlockingMode.PIXEL_PERFECT,
        interaction: InteractionMode | None = None,
        visible: bool = True,
        collidable: bool = True,
        tags: list[str] = [],
    ):
        """Initialize a new Sprite.

        Args:
            pixels: 2D list representing the sprite's pixels
            name: Sprite name (default: None, will generate UUID)
            x: X coordinate in pixels (default: 0)
            y: Y coordinate in pixels (default: 0)
            layer: Z-order layer for rendering (default: 0, higher values render on top)
            scale: Scale factor (default: 1)
            rotation: Rotation in degrees (default: 0)
            blocking: Collision detection method (default: NOT_BLOCKED)
            interaction: How the sprite interacts with the game world (default: TANGIBLE)

        Raises:
            ValueError: If scale is 0, pixels is not a 2D list, rotation is invalid,
                       or if downscaling factor doesn't evenly divide sprite dimensions
        """
        if isinstance(pixels, np.ndarray):
            if pixels.ndim != 2:
                raise ValueError("Pixels must be a 2D array 111")
            if pixels.dtype != np.int8:
                base = pixels.astype(np.int8, copy=False)
            else:
                base = pixels
        else:
            if not isinstance(pixels, list) or not all(isinstance(row, list) for row in pixels):
                raise ValueError("Pixels must be a 2D list or a 2D numpy array")
            base = np.array(pixels, dtype=np.int8)

        self.pixels = base
        if self.pixels.ndim != 2:
            raise ValueError("Pixels must be a 2D array 222")

        self._name = name if name is not None else str(uuid.uuid4())
        self._x = int(x)
        self._y = int(y)
        self._layer = int(layer)
        self._set_rotation(rotation)
        self._mirror_ud = mirror_ud
        self._mirror_lr = mirror_lr
        self._blocking = blocking
        self.set_scale(scale)  # Use set_scale to validate scale factor
        if interaction is None:
            self._interaction = _interaction_mode_from(visible, collidable)
        else:
            self._interaction = interaction
        self._tags = tags
    def clone(self, new_name: Optional[str] = None) -> "Sprite":
        """Create an independent copy of this sprite.

        Args:
            new_name: Optional name for the cloned sprite. If None, reuses current name.

        Returns:
            A new Sprite instance with the same properties but independent state.
        """
        # Create a deep copy of the pixels array
        pixels_copy = self.pixels.copy()

        # Create a new sprite with copied properties
        return Sprite(
            pixels=pixels_copy.tolist(),  # Convert back to list for constructor
            name=new_name if new_name is not None else self._name,  # Use new name or generate new UUID
            x=self._x,
            y=self._y,
            scale=self._scale,
            rotation=self.rotation,  # Use the public property to get normalized value
            mirror_ud=self._mirror_ud,
            mirror_lr=self._mirror_lr,
            blocking=self._blocking,
            layer=self._layer,
            interaction=self._interaction,
            tags=self._tags.copy(),  # Copy the tags list
        )
    def _set_rotation(self, rotation: int) -> None:
        """Internal method to set rotation with validation.

        Args:
            rotation: The rotation value in degrees

        Raises:
            ValueError: If rotation is not a valid 90-degree increment
        """
        normalized = rotation % 360
        if normalized not in self.VALID_ROTATIONS:
            raise ValueError(f"Rotation must be one of {self.VALID_ROTATIONS}, got {rotation}")
        self.rotation = normalized
    def set_rotation(self, rotation: int) -> "Sprite":
        """Set the sprite's rotation to a specific value.

        Args:
            rotation: The new rotation in degrees (must be 0, 90, 180, or 270)

        Raises:
            ValueError: If rotation is not a valid 90-degree increment
        """
        self._set_rotation(int(rotation))
        return self
    def set_position(self, x: int, y: int) -> "Sprite":
        """Set the sprite's position.

        Args:
            x: New X coordinate in pixels
            y: New Y coordinate in pixels
        """
        self._x = int(x)
        self._y = int(y)
        return self
    def set_scale(self, scale: int) -> "Sprite":
        """Set the sprite's scale factor.

        Args:
            scale: The new scale factor. Positive values scale up, negative values scale down.
                  Negative values indicate divisor: -1 means half size (divide by 2), -2 means one-third size, etc.

        Raises:
            ValueError: If scale is 0 or if downscaling factor doesn't evenly divide sprite dimensions
        """
        scale_int = int(scale)
        if scale_int == 0:
            raise ValueError("Scale cannot be zero")

        # For downscaling, validate dimensions are divisible by scale factor
        if scale_int < 0:
            H, W = self.pixels.shape
            factor = -scale_int + 1  # -1 -> 2, -2 -> 3, -3 -> 4, etc.
            if H % factor != 0 or W % factor != 0:
                raise ValueError(f"Array dimensions ({H}, {W}) must be divisible by scale factor {factor}")

        self._scale = scale_int
        return self
    @property
    def name(self) -> str:
        """Get the sprite's name."""
        return self._name
    @property
    def x(self) -> int:
        """Get the current X coordinate."""
        return self._x
    @property
    def y(self) -> int:
        """Get the current Y coordinate."""
        return self._y
    @property
    def blocking(self) -> BlockingMode:
        """Get the current blocking behavior."""
        return self._blocking
    @property
    def layer(self) -> int:
        """Get the current rendering layer."""
        return self._layer
    @property
    def tags(self) -> list[str]:
        """Get the current tags."""
        return self._tags
    @property
    def interaction(self) -> InteractionMode:
        """Get the current interaction mode."""
        return self._interaction
    def set_interaction(self, interaction: InteractionMode) -> "Sprite":
        """Set the sprite's interaction mode.

        Args:
            interaction: The new interaction mode

        Raises:
            ValueError: If interaction is not an InteractionMode enum value
        """
        if not isinstance(interaction, InteractionMode):
            raise ValueError("interaction must be an InteractionMode enum value")
        self._interaction = interaction
        return self
    @property
    def is_visible(self) -> bool:
        """Check if a sprite with this interaction mode should be rendered.

        Returns:
            bool: True if the sprite should be visible, False otherwise
        """
        return self._interaction == InteractionMode.TANGIBLE or self._interaction == InteractionMode.INTANGIBLE
    def set_visible(self, visible: bool) -> "Sprite":
        """Set the sprite's visibility.

        Args:
            visible: The new visibility state
        """
        self._interaction = _interaction_mode_from(visible, self.is_collidable)
        return self
    @property
    def width(self) -> int:
        """Get the sprite's width."""
        return int(self.render().shape[1])
    @property
    def height(self) -> int:
        """Get the sprite's height."""
        return int(self.render().shape[0])
    @property
    def is_collidable(self) -> bool:
        """Check if a sprite with this interaction mode should participate in collisions.

        Returns:
            bool: True if the sprite should be checked for collisions, False otherwise
        """
        return self._interaction == InteractionMode.TANGIBLE or self._interaction == InteractionMode.INVISIBLE
    def render(self) -> np.ndarray:
        """Render the sprite with current scale and rotation.

        Returns:
            np.ndarray: The rendered sprite as a 2D numpy array
        """
        # Start with the base pixels
        result = self.pixels.copy()

        # Handle rotation first (if any)
        if self.rotation != 0:
            # Convert degrees to number of 90-degree rotations (clockwise)
            k = int((-self.rotation % 360) / 90)  # Negative for clockwise rotation
            if k != 0:
                result = np.rot90(result, k=k)

        if self._mirror_ud:
            result = np.flipud(result)
        if self._mirror_lr:
            result = np.fliplr(result)

        # Handle scaling
        if self._scale != 1:
            if self._scale > 1:
                # For upscaling, repeat the array in both dimensions
                result = np.repeat(np.repeat(result, self._scale, axis=0), self._scale, axis=1)
            else:  # self._scale < 0
                # For downscaling, use mode-based approach
                # Convert negative scale to actual divisor (e.g. -1 -> 2, -2 -> 3)
                factor = -self._scale + 1  # -1 -> 2, -2 -> 3, -3 -> 4, etc.
                result = _downscale_mode(result, factor)

        return result
    def move(self, dx: int, dy: int) -> None:
        """Move the sprite by the given deltas.

        Args:
            dx: Change in x position (positive = right, negative = left)
            dy: Change in y position (positive = down, negative = up)
        """
        self._x += int(dx)
        self._y += int(dy)

    # (members not shown: rotate, adjust_scale, set_blocking, set_name, scale, mirror_ud, mirror_lr, set_mirror_ud, set_mirror_lr, set_layer, set_collidable, collides_with, color_remap, merge)
```

`arcengine/level.py` (excerpt):
```python
class Level:
    _sprites: List[Sprite]
    _sorted_sprites: List[Sprite] | None  # Sorted High to low
    _grid_size: Tuple[int, int] | None
    _data: dict[str, Any]
    _name: str
    _placeable_areas: List[PlaceableArea]
    _need_sort: bool
    def __init__(
        self,
        sprites: Optional[List[Sprite]] = None,
        grid_size: Tuple[int, int] | None = None,
        data: dict[str, Any] = {},
        name: str = "Level",
        placeable_areas: Optional[List[PlaceableArea]] = None,
    ):
        """Initialize a new level.

        Args:
            sprites: List of sprites to add to the level
            grid_size: Tuple of width and height of the grid
            data: Dictionary of data to store in the level
            name: Name of the level
            placeable_areas: List of placeable areas in the level
        """
        self._sprites = []
        self._grid_size = grid_size
        self._data = data
        self._name = name
        self._placeable_areas = placeable_areas if placeable_areas is not None else []
        self._need_sort = True

        if sprites:
            # Add first (fast path), then do one-time merge+sort.
            self._sprites.extend(sprites)
            self._merge_sys_static_pixel_perfect_on_init()
    def _merge_sys_static_pixel_perfect_on_init(self) -> None:
        """
        Merge any sprites that are:
          - PIXEL_PERFECT
          - have tag "sys_static"
        into ONE sprite per layer.

        This runs only during construction.
        """
        if not self._sprites:
            return

        # Partition sprites into merge-candidates (by layer) and others.
        by_layer: dict[int, List[Sprite]] = {}
        others: List[Sprite] = []

        for s in self._sprites:
            if s.blocking == BlockingMode.PIXEL_PERFECT and "sys_static" in s.tags:
                by_layer.setdefault(s.layer, []).append(s)
            else:
                others.append(s)

        merged: List[Sprite] = []
        for layer, group in by_layer.items():
            if not group:
                continue
            if len(group) == 1:
                merged.append(group[0])
                continue

            # Merge left-to-right; merge() returns a NEW Sprite each time.
            base = group[0]
            for nxt in group[1:]:
                base = base.merge(nxt)

            # Ensure the merged sprite stays on this layer.
            # (merge() uses max layer, but all are same layer anyway; set explicitly for safety.)
            base.set_layer(layer)

            # Ensure sys_static remains (merge unions tags, so it should already be present)
            if "sys_static" not in base.tags:
                base.tags.append("sys_static")

            merged.append(base)

        self._sprites = others + merged
    def add_sprite(self, sprite: Sprite) -> None:
        """Add a sprite to the level.

        Args:
            sprite: The sprite to add
        """
        self._sprites.append(sprite)
        self._need_sort = True
    def remove_sprite(self, sprite: Sprite) -> None:
        """Remove a sprite from the level.

        Args:
            sprite: The sprite to remove
        """
        if sprite in self._sprites:
            self._sprites.remove(sprite)
    def get_sprites(self) -> List[Sprite]:
        """Get all sprites in the level.

        Returns:
            List[Sprite]: All sprites in the level
        """
        return self._sprites.copy()  # Return copy to prevent external modification
    def get_sprites_by_tag(self, tag: str) -> List[Sprite]:
        """Get all sprites that have the given tag.

        Args:
            tag: The tag to search for

        Returns:
            List[Sprite]: All sprites that have the given tag
        """
        return [s for s in self._sprites if tag in s.tags]
    def get_data(self, key: str) -> Any:
        return self._data.get(key)
    @property
    def grid_size(self) -> Tuple[int, int] | None:
        """Get the grid size of the level.

        Returns:
            Tuple[int, int]: The grid size of the level
        """
        return self._grid_size
    def clone(self) -> "Level":
        """Create a deep copy of this level.

        Returns:
            Level: A new Level instance with cloned sprites
        """
        # Clone each sprite and create new level
        cloned_sprites = [sprite.clone() for sprite in self._sprites]
        return Level(name=self._name, sprites=cloned_sprites, grid_size=self._grid_size, data=copy.deepcopy(self._data), placeable_areas=self._placeable_areas)

    # (members not shown: remove_all_sprites, get_sprites_by_name, get_sprites_by_tags, get_sprites_by_any_tag, get_all_tags, get_sprite_at, collides_with, name, placeable_areas)
```

`arcengine/interfaces.py` (excerpt):
```python
class RenderableUserDisplay(ABC):
    @abstractmethod
    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        """Render this UI element onto the given frame.

        Args:
            frame: The 64x64 numpy array to render onto
        """
        return frame

    # (members not shown: draw_sprite)
```

`arcengine/camera.py` (excerpt):
```python
class Camera:
    MAX_DIMENSION = 64
    _x: int
    _y: int
    _width: int
    _height: int
    _background: int
    _letter_box: int
    _interfaces: list[RenderableUserDisplay]
    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        width: int = 64,
        height: int = 64,
        background: int = 5,
        letter_box: int = 5,
        interfaces: list[RenderableUserDisplay] = [],
    ):
        """Initialize a new Camera.

        Args:
            x: X coordinate in pixels (default: 0)
            y: Y coordinate in pixels (default: 0)
            width: Viewport width in pixels (default: 64, max: 64)
            height: Viewport height in pixels (default: 64, max: 64)
            background: Background color index (default: 5 - Black)
            letter_box: Letter box color index (default: 5 - Black)
            interfaces: Optional list of renderable interfaces to initialize with

        Raises:
            ValueError: If width or height exceed 64 pixels
        """
        if width > 64 or height > 64 or width < 0 or height < 0:
            raise ValueError("Camera dimensions cannot exceed 64x64 pixels and must be positive")

        self._x = x
        self._y = y
        self.width = width
        self.height = height
        self._background = background
        self._letter_box = letter_box
        self._interfaces: List[RenderableUserDisplay] = []

        if interfaces:
            for interface in interfaces:
                self._interfaces.append(interface)
    @property
    def x(self) -> int:
        """Get the camera's x position.

        Returns:
            int: The camera's x position
        """
        return self._x
    @x.setter
    def x(self, value: int) -> None:
        """Set the camera's x position.

        Args:
            value: The new x position
        """
        self._x = int(value)
    @property
    def y(self) -> int:
        """Get the camera's y position.

        Returns:
            int: The camera's y position
        """
        return self._y
    @y.setter
    def y(self, value: int) -> None:
        """Set the camera's y position.

        Args:
            value: The new y position
        """
        self._y = int(value)
    @property
    def width(self) -> int:
        """Get the camera's width.

        Returns:
            int: The camera's width
        """
        return self._width
    @width.setter
    def width(self, value: int) -> None:
        """Set the camera's width.

        Args:
            value: The new width

        Raises:
            ValueError: If width exceeds MAX_DIMENSION
        """
        width_int = int(value)
        if width_int > self.MAX_DIMENSION:
            raise ValueError(f"Width cannot exceed {self.MAX_DIMENSION} pixels")
        self._width = width_int
    @property
    def height(self) -> int:
        """Get the camera's height.

        Returns:
            int: The camera's height
        """
        return self._height
    @height.setter
    def height(self, value: int) -> None:
        """Set the camera's height.

        Args:
            value: The new height

        Raises:
            ValueError: If height exceeds MAX_DIMENSION
        """
        height_int = int(value)
        if height_int > self.MAX_DIMENSION:
            raise ValueError(f"Height cannot exceed {self.MAX_DIMENSION} pixels")
        self._height = height_int
    @property
    def background(self) -> int:
        """Get the camera's background color."""
        return self._background
    @background.setter
    def background(self, value: int) -> None:
        """Set the camera's background color."""
        self._background = value
    @property
    def letter_box(self) -> int:
        """Get the camera's letter box color."""
        return self._letter_box
    @letter_box.setter
    def letter_box(self, value: int) -> None:
        """Set the camera's letter box color."""
        self._letter_box = value
    def resize(self, width: int, height: int) -> None:
        """Resize the camera.

        Args:
            width: The new width
            height: The new height
        """
        self.width = width
        self.height = height
    def move(self, dx: int, dy: int) -> None:
        """Move the camera by the specified delta.

        Args:
            dx: The change in x position
            dy: The change in y position
        """
        self._x += int(dx)
        self._y += int(dy)
    def _calculate_scale_and_offset(self) -> Tuple[int, int, int]:
        """Calculate the scale factor and offsets for letterboxing.

        Returns:
            Tuple[int, int, int]: (scale, x_offset, y_offset)
            - scale: The uniform scale factor to fit viewport in 64x64
            - x_offset: Horizontal offset for centering
            - y_offset: Vertical offset for centering
        """
        # Calculate maximum possible scale that fits in MAX_DIMENSION
        scale_x = self.MAX_DIMENSION // self._width
        scale_y = self.MAX_DIMENSION // self._height
        scale = min(scale_x, scale_y)

        # Calculate scaled dimensions
        scaled_width = self._width * scale
        scaled_height = self._height * scale

        # Only use offsets if we can't scale up to fill the screen
        x_offset = (self.MAX_DIMENSION - scaled_width) // 2
        y_offset = (self.MAX_DIMENSION - scaled_height) // 2

        return scale, x_offset, y_offset
    def _raw_render(self, sprites: List[Sprite]) -> np.ndarray:
        """Internal method to render the camera view.

        Args:
            sprites: List of sprites to render. Sprites are rendered in order of their layer
                    value (lower layers first). Negative pixel values are treated as transparent.
                    Only visible sprites (based on their interaction mode) will be rendered.

        Returns:
            np.ndarray: The rendered view as a 2D numpy array
        """
        # Create background array filled with background color
        output = np.full((self._height, self._width), self._background, dtype=np.int8)

        if not sprites:
            return output

        # Sort sprites by layer (lower layers first) and filter out non-visible sprites
        sorted_sprites = sorted((s for s in sprites if s.is_visible), key=lambda s: s.layer)

        for sprite in sorted_sprites:
            # Get the sprite's rendered pixels (handles rotation and scaling)
            sprite_pixels = sprite.render()
            sprite_height, sprite_width = sprite_pixels.shape

            # Calculate sprite position relative to camera
            rel_x = sprite.x - self._x
            rel_y = sprite.y - self._y

            # Calculate the intersection with viewport
            dest_x_start = max(0, rel_x)
            dest_x_end = min(self._width, rel_x + sprite_width)
            dest_y_start = max(0, rel_y)
            dest_y_end = min(self._height, rel_y + sprite_height)

            # Skip if sprite is completely outside viewport
            if dest_x_end <= dest_x_start or dest_y_end <= dest_y_start:
                continue

            # Calculate source region from sprite
            sprite_x_start = max(0, -rel_x)
            sprite_x_end = sprite_width - max(0, (rel_x + sprite_width) - self._width)
            sprite_y_start = max(0, -rel_y)
            sprite_y_end = sprite_height - max(0, (rel_y + sprite_height) - self._height)

            # Get the sprite region we're going to copy
            sprite_region = sprite_pixels[sprite_y_start:sprite_y_end, sprite_x_start:sprite_x_end]

            # Create a mask for non-negative (visible) pixels
            visible_mask = sprite_region >= 0

            # Update only the non-transparent pixels
            output[dest_y_start:dest_y_end, dest_x_start:dest_x_end][visible_mask] = sprite_region[visible_mask]

        return output
    def render(self, sprites: List[Sprite]) -> np.ndarray:
        """Render the camera view.

        The rendered output is always 64x64 pixels. If the camera's viewport is smaller,
        the view will be scaled up uniformly (maintaining aspect ratio) to fit within
        64x64, and the remaining space will be filled with the letter_box color.

        Args:
            sprites: List of sprites to render (currently unused)

        Returns:
            np.ndarray: The rendered view as a 64x64 numpy array
        """
        # Start with a letter-boxed canvas
        output = np.full((self.MAX_DIMENSION, self.MAX_DIMENSION), self._letter_box, dtype=np.int8)

        # Get the raw camera view
        view = self._raw_render(sprites)

        # Calculate scaling and offsets
        scale, x_offset, y_offset = self._calculate_scale_and_offset()

        # Scale up the view using numpy's repeat
        if scale > 1:
            view = np.repeat(np.repeat(view, scale, axis=0), scale, axis=1)

        # Insert the scaled view into the letter-boxed output
        output[y_offset : y_offset + view.shape[0], x_offset : x_offset + view.shape[1]] = view

        for interface in self._interfaces:
            output = interface.render_interface(output)

        return output
    def display_to_grid(self, display_x: int, display_y: int) -> tuple[int, int] | None:
        """Convert display coordinates (64x64) to camera grid coordinates.

        This takes into account:
        - Camera scaling max(64/camera_width, 64/camera_height)
        - Letterbox padding
        - Grid boundaries

        Args:
            display_x: X coordinate in display space (0-63)
            display_y: Y coordinate in display space (0-63)

        Returns:
            tuple[int, int] | None: The corresponding grid coordinates (x, y) or None if the display coordinates are within the letterbox
        """
        # Calculate scaling factor
        scale_x = int(64 / self.width)
        scale_y = int(64 / self.height)
        scale = min(scale_x, scale_y)

        # Calculate letterbox padding
        x_padding = int((64 - (self.width * scale)) / 2)
        y_padding = int((64 - (self.height * scale)) / 2)

        # Remove padding and scale down
        grid_x = int((display_x - x_padding) / scale) if display_x - x_padding >= 0 else -1
        grid_y = int((display_y - y_padding) / scale) if display_y - y_padding >= 0 else -1

        if grid_x < 0 or grid_y < 0 or grid_x >= self.width or grid_y >= self.height:
            # Given X, Y is off the camera screen
            return None

        return grid_x + self.x, grid_y + self.y

    # (members not shown: replace_interface)
```

`arcengine/base_game.py` (excerpt):
```python
MAX_FRAME_PER_ACTION: int = 1000


class ARCBaseGame(ABC):
    _game_id: str
    _levels: list[Level]
    _clean_levels: list[Level]
    _current_level_index: int
    _camera: Camera
    _debug: bool
    _action: ActionInput
    _action_complete: bool
    _action_count: int
    _state: GameState
    _score: int
    _next_level: bool
    _full_reset: bool
    _win_score: int
    _available_actions: list[int]
    _placeable_sprite: Optional[Sprite]
    _seed: int
    def __init__(
        self,
        game_id: str,
        levels: List[Level],
        camera: Optional[Camera] = None,
        debug: bool = False,
        win_score: int = 1,
        available_actions: list[int] = [1, 2, 3, 4, 5, 6],
        seed: int = 0,
    ) -> None:
        """Initialize a new game.

        Args:
            levels: List of levels to initialize the game with. Each level will be cloned.
            camera: Optional camera to use. If not provided, a default 64x64 camera will be created.

        Raises:
            ValueError: If levels list is empty
        """
        if not levels:
            raise ValueError("Game must have at least one level")

        # Game ID should be set by subclasses
        self._game_id = game_id

        # Clone each level to prevent external modification
        self._levels = [level.clone() for level in levels]
        self._clean_levels = [level.clone() for level in levels]
        self._current_level_index = 0

        # Debug mode
        self._debug = debug

        # Camera
        # Use provided camera or create default
        self._camera = camera if camera is not None else Camera()

        # Game state
        self._state = GameState.NOT_PLAYED
        self._score = 0
        self._next_level = False
        self._action = ActionInput()
        self._action_complete = False
        self._action_count = 0
        self._full_reset = False
        self._win_score = win_score if win_score > 1 else len(levels)
        self.set_level(0)
        self._available_actions = available_actions
        self._placeable_sprite = None
        self._seed = seed
    @property
    @final
    def current_level(self) -> Level:
        """Get the current level.

        Returns:
            Level: The current level
        """
        return self._levels[self._current_level_index]
    @property
    @final
    def camera(self) -> Camera:
        """Get the game's camera.

        Returns:
            Camera: The game's camera
        """
        return self._camera
    @property
    @final
    def game_id(self) -> str:
        """Get the game's ID.

        Returns:
            str: The game's ID
        """
        return self._game_id
    @final
    def set_level(self, index: int) -> None:
        """Set the current level by index.

        Args:
            index: The index of the level to set as current

        Raises:
            IndexError: If index is out of range
        """
        if not 0 <= index < len(self._levels):
            raise IndexError(f"Level index {index} out of range [0, {len(self._levels)})")
        self._current_level_index = index
        self._action_count = 0
        level = self.current_level
        if level.grid_size:
            self.camera.resize(level.grid_size[0], level.grid_size[1])
        self.on_set_level(level)
    @property
    @final
    def level_index(self) -> int:
        """Get the current level index.

        Returns:
            int: The current level index
        """
        return self._current_level_index
    @final
    def perform_action(self, action_input: ActionInput, raw: bool = False) -> FrameData | FrameDataRaw:
        """Perform an action and return the resulting frame data.

        DO NOT OVERRIDE THIS METHOD, Your Game Logic should be in step()

        The base implementation:
        1. While the action is not complete, call step() and render frames
        2. Returns a FrameData object with the current state

        Args:
            action_input: The action to perform

        Returns:
            FrameData: The resulting frame data
        """
        self._full_reset = False
        if action_input.id == GameAction.RESET:
            self.handle_reset()
        elif self._state == GameState.GAME_OVER or self._state == GameState.WIN:
            return FrameData(
                game_id=self._game_id,
                frame=[],
                state=self._state,
                score=self._score,
                win_score=self._win_score,
                action_input=action_input,
                available_actions=self._available_actions,
            )

        self._set_action(action_input)

        frame_list: list[ndarray | list[list[int]]] = []

        count = 0

        while not self.is_action_complete():
            if count > MAX_FRAME_PER_ACTION:
                raise ValueError("Action took too many frames")
            count += 1
            if self._next_level:
                self._really_set_next_level()
            else:
                self.step()
            frame = self.camera.render(self.current_level.get_sprites())
            if raw:
                frame_list.append(frame)
            else:
                frame_list.append(frame.tolist())

        # Create and return FrameData
        if raw:
            frame_raw = FrameDataRaw()
            frame_raw.game_id = self._game_id
            frame_raw.frame = frame_list
            frame_raw.state = self._state
            frame_raw.levels_completed = self._score
            frame_raw.win_levels = self._win_score
            frame_raw.action_input = action_input
            frame_raw.full_reset = self._full_reset
            frame_raw.available_actions = self._available_actions
            return frame_raw

        return FrameData(
            game_id=self._game_id,
            frame=frame_list,
            state=self._state,
            levels_completed=self._score,
            win_levels=self._win_score,
            action_input=action_input,
            full_reset=self._full_reset,
            available_actions=self._available_actions,
        )
    @property
    @final
    def action(self) -> ActionInput:
        """Get the current action."""
        return self._action
    @final
    def _set_action(self, action_input: ActionInput) -> None:
        """Set the action to perform.

        Args:
            action_input: The action to perform
        """
        self._state = GameState.NOT_FINISHED
        self._action = action_input
        self._action_complete = False
        if action_input.id != GameAction.RESET:
            self._action_count += 1
    @final
    def complete_action(self) -> None:
        """Complete the action. Call this when the provided action is fully resolved"""
        self._action_complete = True
    @final
    def is_action_complete(self) -> bool:
        """Check if the action is complete.

        Returns:
            bool: True if the action is complete, False otherwise
        """
        return not self._next_level and self._action_complete
    @final
    def win(self) -> None:
        """Call this when the player has beaten the game."""
        self._state = GameState.WIN
    @final
    def lose(self) -> None:
        """Call this when the player has losses the game."""
        self._state = GameState.GAME_OVER
    def handle_reset(self) -> None:
        """Handle the reset of the game.

        If the action count is 0, perform a full reset.
        Otherwise, perform a level reset.
        """
        if os.getenv("ONLY_RESET_LEVELS") == "true" and self._state != GameState.WIN:
            self.level_reset()
        elif self._action_count == 0 or self._state == GameState.WIN:
            self.full_reset()
        else:
            self.level_reset()
    def full_reset(self) -> None:
        self._levels = [level.clone() for level in self._clean_levels]
        self._score = 0
        self._action_count = 0
        self._full_reset = True
        self.set_level(0)
        self._state = GameState.NOT_FINISHED
    def level_reset(self) -> None:
        self._levels[self._current_level_index] = self._clean_levels[self._current_level_index].clone()
        self.set_level(self._current_level_index)
        self._state = GameState.NOT_FINISHED
    def step(self) -> None:
        """Step the game.  This is where your game logic should be implemented.

        REQUIRED: Call complete_action() when the action is complete.
          It does not need to be called every step, but once the action is complete.
          The engine will keep calling step and rendering frames until the action is complete.
        """

        self.complete_action()
    def try_move_sprite(self, sprite: Sprite, dx: int, dy: int) -> List[Sprite]:
        """Try to move a sprite and return a list of sprites it collides with.

        This method attempts to move the sprite by the given deltas and checks for collisions.
        If any collisions are detected, the sprite is not moved and the method returns a list
        of sprite names that were collided with.

        Args:
            sprite_name: The name of the sprite to move.
            dx: The change in x position (positive = right, negative = left).
            dy: The change in y position (positive = down, negative = up).

        Returns:
            A list of sprite names that the sprite collided with. If no collisions occurred,
            the sprite is moved and an empty list is returned.

        Raises:
            ValueError: If no sprite with the given name is found.
        """  # Get the sprite to move
        # Store original position
        original_x = sprite.x
        original_y = sprite.y

        # Try the move
        sprite.move(dx, dy)

        # Check for collisions with all other sprites
        collisions = []
        for other in self.current_level.get_sprites():
            if sprite.collides_with(other):
                collisions.append(other)

        # If there were collisions, revert the move
        if collisions:
            sprite.set_position(original_x, original_y)

        return collisions
    def is_last_level(self) -> bool:
        """Check if the current level is the last level.

        Returns:
            bool: True if the current level is the last level, False otherwise
        """
        return self._current_level_index == len(self._levels) - 1
    def next_level(self) -> None:
        """Move to the next level."""
        self._score += 1
        if not self.is_last_level():
            self._next_level = True
        else:
            self.win()
    def _really_set_next_level(self) -> None:
        self.set_level(self._current_level_index + 1)
        self._next_level = False
    def on_set_level(self, level: Level) -> None:
        """Called when the level is set, use this to set level specific data."""
        pass

    # (members not shown: debug, win_score, set_level_by_name, try_move, get_pixels_at_sprite, get_pixels, set_placeable_sprite, _get_graph_location, _get_hidden_state, _get_valid_actions, _get_valid_placeble_actions, _get_valid_clickable_actions, _is_sprite_clickable_now)
```

`dc22.py`:
[GAME SOURCE NOT INCLUDED: the redacted (mechanics-only) source of the official game file `dc22.py` of official.dc22, generated by scripts/build_sections_public.py (wsp.redact_public.public_section); restore it with scripts/rebuild_public_prompts.py; sha256 45919107bff79ff16ddfa68f79f33107a436045c55d0e5d53b99ec186b62a831]


## Episodes and scoring

An episode starts at the beginning of one level (or at a state reached by a few actions into
it) and ends when that level is completed (success), on GAME_OVER, or after 3 x (shortest solution length) + 10
actions. Invalid actions count as wasted steps. RESET restarts the current level and counts as an
action. Success rate over many held-out episodes is the score.


## Example observations (support set)

Each grid is 64 rows of 64 hexadecimal colour ids.
### level 1, start
```
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
4444444444444444444444444444444400555555555555555555555555555555
4444444444444444444444444444444400555555555555555555555555555555
4444444444444444444444444444444455555555555555555555555555555555
4444444444444444444444444444444455555555555555555555555555555555
4444444444444444444444444444444400555555555555555555555555555555
4444444444444444444444444444444400555555555555555555555555555555
4444444444444444444444444444444455555555555500000000055555555555
4444444444444444444444444444444455555555555508888888055555555555
4444444444444444444444222222444400555555555508888888055555555555
4444444444444444444444222222444400555555500008888888000055555555
444444444444444444999922bb22444455555555508888888888888055555555
444444444444444444999922bb22444455555555508888888888888055555555
4444444444444444449999222222444400555555500000000000000055555555
4444444444444444449999222222444400555555555555555555555555555555
4444444444444444444444444444444455555555555555555555555555555555
4444444444444444444444444444444455555555555555555555555555555555
4444444444444444444444444444444400555555555555555555555555555555
4444444444444444444444444444444400555555555555555555555555555555
4444444444444444444444444444444455555555555555555555555555555555
4444444444444444444444444444444455555555555555555555555555555555
444444442222888888dddd444444444400555555555555555555555555555555
444444442222888888dddd444444444400555555555555555555555555555555
444444442222888888dddd444444444455555555555555555555555555555555
444444442222888888dddd444444444455555555555500000000055555555555
4444444449494444444444444444444400555555555509999999055555555555
4444444494944444444444444444444400555555555509999999055555555555
4444444449494444444444444444444455555555500009999999000055555555
4444444494944444444444444444444455555555509999999999999055555555
4444444422222244444444444444444400555555509999999999999055555555
4444444422222244444444444444444400555555500000000000000055555555
4444444422ee2244444444444444444455555555555555555555555555555555
4444444422ee2244444444444444444455555555555555555555555555555555
4444444422222244444444444444444400555555555555555555555555555555
4444444422222244444444444444444400555555555555555555555555555555
4444444444444444444444444444444455555555555555555555555555555555
4444444444444444444444444444444455555555555555555555555555555555
4444444444444444444444444444444400555555555555555555555555555555
4444444444444444444444444444444400555555555555555555555555555555
4444444444444444444444444444444455555555555555555555555555555555
4444444444444444444444444444444455555555555555555555555555555555
4444444444444444444444444444444400555555555555555555555555555555
4444444444444444444444444444444400555555555555555555555555555555
4444444444444444444444444444444455555555555555555555555555555555
4444444444444444444444444444444455555555555555555555555555555555
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
0000000000000000000000000000000000000000000000000000000000000000
```

### level 3, start
```
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
4444444444444444444444444444444444444400555555555555555555555555
4444444444444444444444444444444444444400555555555555555555555555
4444444444444444444444444444444444444455555555555555555555555555
4444444444444444444444444444444444444455555555555555555555555555
4444444444444444444444444444444444444400555555555555555555555555
4444444444444444444444444444444444444400555555555555555555555555
4444444444444444444444444444444444444455555555555555555555555555
4444444444444444444444444444444444444455555555555555555555555555
4444227644444444444444444949222244444400555555558888888555555555
4444226744444444444444449494222244444400555555558888888555555555
444422224444444444444444494922ee44444455555555558888888555555555
444422224444444444444444949422ee44444455555558888888888888555555
4444494944444444444444444444444444444400555558888888888888555555
4444949444444444444444444444444444444400555555555555555555555555
4444494944444444444444444444444444444455555555555555555555555555
4444949444444444444444444444444444444455555555555555555555555555
44442222999988888888dddd8888888844444400555555555555555555555555
44442222999988888888dddd8888888844444400555555559999999555555555
44442222999988888888dddd8888888844444455555555559999999555555555
44442222999988888888dddd8888888844444455555555559999999555555555
4444444444444444444444444444444444444400555559999999999999555555
4444444444444444444444444444444444444400555559999999999999555555
4444444444444444444444444444444444444455555555555555555555555555
4444444444444444444444444444444444444455555555555555555555555555
44444444444444444f4f44444f4f4f4f4f444400555555555555555555555555
4444444444444444f4f44444f4f4f4f4f4444400555555555555555555555555
44444444444444444f4f44444f4f4f4f4f444455555555556666666555555555
4444444444444444f4f44444f4f4f4f4f4444455555555556666666555555555
44444444444444444f4f44444444442222444400555555556666666555555555
4444444444444444f4f444444444442222444400555556666666666666555555
44444444444444444f4f444444444422bb444455555556666666666666555555
4444444444444444f4f4444444444422bb444455555555555555555555555555
4444444444444949222244444444444444444400555555555555555555555555
4444444444449494222244444444444444444400555555555555555555555555
44444444444449492ff244444444444444444455555555555555555555555555
4444444444449494ffff44444444444444444455555555555555555555555555
4444444444442222444444444444444444444400555555555555555555555555
4444444444442222444444444444444444444400555555555555555555555555
4444444444446722444444444444444444444455555555555555555555555555
4444444444447622444444444444444444444455555555555555555555555555
4444444444444444444444444444444444444400555555555555555555555555
4444444444444444444444444444444444444400555555555555555555555555
4444444444444444444444444444444444444455555555555555555555555555
4444444444444444444444444444444444444455555555555555555555555555
4444444444444444444444444444444444444400555555555555555555555555
4444444444444444444444444444444444444400555555555555555555555555
4444444444444444444444444444444444444455555555555555555555555555
4444444444444444444444444444444444444455555555555555555555555555
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
3333333333333333333333333333333333333333333333333333333333333333
0000000000000000000000000000000000000000000000000000000000000000
```

### level 5, start
```
4444444444444444444444444444444444444400555555555555555555555555
4444444444444444444444444444444444444400555555555555555555555555
4444444444444444444444444444444444444455555555555555555555555555
4444444444444444444444444444444444444455555555555555555555555555
4444444444444444444444444444444444444400555555555555555555555555
4444444444444444444444444444444444444400555555555555555555555555
4444444444444444444444227644444444444455555555555555555555555555
4444444444444444444444226744444444444455555555555555555555555555
4444444444444444444444222244444444444400555555555555555555555555
4444444444444444444444222244444444444400555555555555555555555555
4444444444444444444444cccc44444444444455555555555555555555555555
4444444444444444444444cccc44444444444455555555555555555555555555
4444444444444444444444cccc44444444444400555555555555555555555555
4444444444444444444444cccc44444444444400555555555555555555555555
44400000000000000000c4cccc44444444444455555555555555555555555555
44400000000000000000cccccc44444444444455555555555555555555555555
44400000000000000000cccccc44444444444400555555555555555555555555
44400000000000000000c4cccc44444444444400555555555555555555555555
44400004442ff244444444cccc44444444444455555555555555555555555555
4440000444ffff44444444cccc444444444444555555bbbb5555555555555555
4440000444222244444444cccc222244444444005555bbbb5555552222225555
4440000444222244444444cccc222244444444005555bbbb5555552222225555
44400004444444444444444444222244444444555555bbbbbbb5552222225555
44400004444444444444444444222244444444555555bbbbbbb5552222225555
44400888884444444444444444444444444444005555bbbbbbb5552222225555
44400804484444444444444444444444444444005555bbbbbbb5552222225555
4448888000000000000444444444444444444455555555555555555555555555
4488880000000000000444444444444444444455555555555555555555555555
4488880000000000000444444444444444444400555aaaa599995aaaa5999955
4448888000000000000444444444444444444400555aaaa599995aaaa5999955
4444484448444444444444444422224444444455555aaaa599995aaaa5999955
4444488888444444444444444422224444444455555aaaa599995aaaa5999955
4444444444444444444444444422224444444400555555555555555555555555
4444444444444444444444444422224444444400555555555555555555555555
44444444446722444444444444494922ee444455555555558888888885555555
44444444447622444444444444949422ee444455555555558888888885555555
4444444444222244444444444449492222444400555555558888888885555555
4444444444222244444444444494942222444400555555555555555555555555
4444444444999944444444444444444444444455555555555555555555555555
4444444444999944444444444444444444444455555555555999999955555555
4444444444999944444444444444444444444400555555555999999955555555
4444444444999944444444444444444444444400555555555999999955555555
4444444444222244444444444444444444444455555555999999999999955555
4444444444222244444444444444444444444455555555999999999999955555
4444444444222255555555555555554444444400555555555555555555555555
4444444444222255555555555555554444444400555555555555555555555555
444444444444444444444444444f4f4444444455555555555666666655555555
44444444444444444444444444f4f44444444455555555555666666655555555
444444444444444444444444444f4f4444444400555555555666666655555555
44444444444444444444444444f4f44444444400555555666666666666655555
4444444444bb22ffffffffff444f4f4444444455555555666666666666655555
4444444444bb22ffffffffff44f4f44444444455555555555555555555555555
44444444442222ffffffffff444f4f4444444400555555555555555555555555
44444444442222ffffffffff44f4f44444444400555555555555555555555555
4444444444444411111144444444444444444455555555555555555555555555
4444444444444411111144444444444444444455555555555555555555555555
4444444444444444444444444444444444444400555555555555555555555555
4444444444444444444444444444444444444400555555555555555555555555
4444444444444444444444444444444444444455555555555555555555555555
4444444444444444444444444444444444444455555555555555555555555555
4444444444444444444444444444444444444400555555555555555555555555
4444444444444444444444444444444444444400555555555555555555555555
4444444444444444444444444444444444444455555555555555555555555555
0000000000000000000000000000000000000000000000000000000000000000
```

## The actor network (you design its architecture and choose all its numbers)

The actor is a neural network that you define as a static graph of tensor operations in the small
language below. At every step it receives only the current 64x64 observation (colour ids 0-15), its
own memory vector (zeros at the start of every episode), a one-hot of its previous action (zeros at
episode start) and the mask of available actions. It outputs action logits, a click map and its next
memory. The actor runs alone: no other code, search, or lookup exists at deployment. Action index
order: ['RESET', 'ACTION1', 'ACTION2', 'ACTION3', 'ACTION4', 'ACTION5', 'ACTION6', 'ACTION7'] (index 6 = ACTION6 = click).

A network is `{"graph": [node, ...], "memory_size": M}` plus a dict `params` of named arrays
(cast to float32). The graph must consist of JSON-like values (dicts, lists, strings, numbers).

**Tensors and inputs.** Spatial tensors have shape (C, 64, 64); vectors have shape (n,). Inputs:
- `x0` (16, 64, 64): one-hot colour planes, x0[c, i, j] = 1 if obs[i, j] == c else 0;
- `coords` (3, 64, 64): row/63, col/63, constant 1;
- `mem` (M,): your memory, 0 <= M <= 256 chosen by you (absent if M = 0); zeros at episode start;
- `aprev` (8,): one-hot of the previous action (zeros at episode start);
- `avail` (8,): 1.0 for available actions, else 0.0.

**Nodes.** Each node is a dict `{"id": name, "op": op, "in": [ids], ...op fields}` (an optional
"comment" string is allowed; any other field is an error). Nodes run in list order and may only use
inputs and earlier nodes (no cycles). Ids match `[A-Za-z_][A-Za-z0-9_]*` and are unique across the
whole graph, including loop bodies and loop states. Parameters are referenced by name in the fields
"w" and "b" ("b" is optional: omit it for no bias). Every array in params must be used; one array may
be used by several nodes.

**Ops** (the NumPy reference below is the exact definition):
- `conv` (1 spatial input; w (O, C, k, k) with k in {1, 3, 5, 7}, optional b (O,)): cross-correlation,
  stride 1, zero padding k//2, output (O, 64, 64);
- `relu`, `tanh`, `sigmoid`, `abs` (1 spatial): elementwise;
- `add`, `sub`, `mul`, `max2`, `min2` (2 inputs, at least one spatial): elementwise with NumPy
  broadcasting, a vector (C,) acting as (C, 1, 1); channel counts must be equal or one of them 1;
- `concat` (1 or more spatial): channel concatenation; `slice` (1 spatial; integer fields "start",
  "stop"): channels start..stop-1;
- `rowmax`, `colmax`, `rowsum`, `colsum` (1 spatial): max / sum of each channel over its row (resp.
  column), broadcast back to (C, 64, 64); `rowmax` at (c, i, j) = max over j' of x[c, i, j'];
- `gmax`, `gmean` (1 spatial): max / mean over all 4096 positions, giving a vector (C,);
- `dense` (1 vector; w (m, n), optional b (m,)): w @ x + b;
- `vrelu`, `vtanh`, `vsigmoid` (1 vector); `vadd`, `vmul` (2 vectors of equal length, or one of
  length 1); `vconcat` (1 or more vectors); `vslice` (1 vector; "start", "stop");
- `loop` (fields "T", "state", "body", "next"; no "in"): runs the node list "body" T times
  (1 <= T <= 256) with the same parameters every iteration. "state" lists the carried tensors, each
  `{"id": s, "init": id}` (start from a tensor defined before the loop) or `{"id": s, "zeros": [C, 64, 64]}`
  / `{"id": s, "zeros": [n]}`. Body nodes may use the states, earlier body nodes and anything
  defined before the loop. "next" maps every state id to the id holding its value for the next
  iteration; all states update together. After the loop, `"<loop id>.<state id>"` is a state's final
  value (body ids are not visible outside the loop). Loops cannot be nested.
There are no other ops, no data-dependent control flow and no argmax inside the network.

**Outputs** (top-level nodes with these ids): `logits` (8,); `click` (1, 64, 64); `mem_next` (M,)
(omit it if M = 0), which becomes `mem` at the next step exactly as computed (nothing squashes it).

**Decoding** is deterministic and outside the network: unavailable actions get logit -inf and the
argmax over the available actions is taken (ties go to the lowest index). If ACTION6 wins, the click
target is the argmax cell of `click` (ties go to the first cell in row-major order). If any available
logit is NaN or infinite, the step is an invalid action.

**Budget and limits** (checked when build() returns; any violation makes the network invalid):
- at most 150,000 parameters (entries of all arrays in params);
- at most 1.5e9 = 1,500,000,000 multiply-accumulates (MACs) per step, counted as:
  conv O*C*k*k*4096; dense m*n; add/sub/mul/max2/min2 one per output element;
  rowmax/colmax/rowsum/colsum/gmax/gmean one per input element; vadd/vmul one per element; all other
  ops 0; a loop body counts T times;
- at most 1,000 nodes; at most 10,000 node evaluations per step (a loop body counts T times);
  at most 512 channels per spatial tensor and 4,096 channels summed over all spatial tensors
  (inputs, nodes and loop states, a loop body counted once); vectors of at most 4,096 entries;
- every parameter must be finite after the float32 cast; the network runs in exact float32.

Exact semantics (this NumPy reference is authoritative; deployment uses an equivalent batched
float32 PyTorch implementation; `ACTIONS` is the action list above and `CLICK` = 6):

```python
def np_op(nd, xs, P):
    """One node on one example. Spatial tensors are (C,64,64), vectors (n,); P = params (float32)."""
    op, x = nd["op"], xs[0]
    lift = lambda a: a[:, None, None] if a.ndim == 1 else a      # a vector (C,) acts as (C,1,1)
    if op == "conv":                                             # cross-correlation, zero padding k//2
        w = P[nd["w"]]
        k = w.shape[-1]
        if k == 1:
            y = np.einsum("oc,chw->ohw", w[:, :, 0, 0], x)
        else:
            xp = np.pad(x, ((0, 0), (k // 2, k // 2), (k // 2, k // 2)))
            win = np.lib.stride_tricks.sliding_window_view(xp, (k, k), axis=(1, 2))  # C,64,64,k,k
            y = np.einsum("chwij,ocij->ohw", win, w, optimize=True)
        return y + P[nd["b"]][:, None, None] if "b" in nd else y
    if op == "dense":
        y = P[nd["w"]] @ x
        return y + P[nd["b"]] if "b" in nd else y
    if op in ("relu", "vrelu"):
        return np.maximum(x, 0)
    if op in ("tanh", "vtanh"):
        return np.tanh(x)
    if op in ("sigmoid", "vsigmoid"):
        return 1 / (1 + np.exp(-x))
    if op == "abs":
        return np.abs(x)
    if op in ("add", "sub", "mul", "max2", "min2"):             # broadcasting; result (C,64,64)
        f = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "max2": np.maximum, "min2": np.minimum}[op]
        return f(lift(x), lift(xs[1]))
    if op == "vadd":
        return x + xs[1]
    if op == "vmul":
        return x * xs[1]
    if op in ("concat", "vconcat"):
        return np.concatenate(xs, 0)
    if op in ("slice", "vslice"):
        return x[nd["start"]:nd["stop"]]
    if op == "rowmax":                                           # (c,i,j) -> max_j' x[c,i,j']
        return np.broadcast_to(x.max(2, keepdims=True), x.shape)
    if op == "colmax":                                           # (c,i,j) -> max_i' x[c,i',j]
        return np.broadcast_to(x.max(1, keepdims=True), x.shape)
    if op == "rowsum":
        return np.broadcast_to(x.sum(2, keepdims=True), x.shape)
    if op == "colsum":
        return np.broadcast_to(x.sum(1, keepdims=True), x.shape)
    if op == "gmax":
        return x.max((1, 2))
    if op == "gmean":
        return x.mean((1, 2))
    raise ValueError(op)


def run_graph(nodes, env, P):
    """Evaluate nodes in list order into env (id -> array)."""
    for nd in nodes:
        if nd["op"] != "loop":
            env[nd["id"]] = np_op(nd, [env[i] for i in nd["in"]], P)
            continue
        state = {s["id"]: env[s["init"]] if "init" in s else np.zeros(s["zeros"], np.float32)
                 for s in nd["state"]}
        for _ in range(nd["T"]):                                 # same parameters every iteration
            local = {**env, **state}
            run_graph(nd["body"], local, P)
            state = {sid: local[src] for sid, src in nd["next"].items()}   # all states update together
        for sid, v in state.items():
            env[nd["id"] + "." + sid] = v


def reference_step(arch, params, obs, mem, aprev, avail, return_env=False):
    """Exact semantics of one step (float32). Returns (action, row, col), masked logits, mem_next."""
    P = {k: np.asarray(v, np.float32) for k, v in params.items()}
    x0 = np.zeros((16, 64, 64), np.float32)
    x0[obs, np.arange(64)[:, None], np.arange(64)[None, :]] = 1.0
    coords = np.zeros((3, 64, 64), np.float32)
    coords[0] = np.arange(64)[:, None] / 63.0
    coords[1] = np.arange(64)[None, :] / 63.0
    coords[2] = 1.0
    env = {"x0": x0, "coords": coords, "mem": np.asarray(mem, np.float32),
           "aprev": np.asarray(aprev, np.float32), "avail": np.asarray(avail, np.float32)}
    with np.errstate(all="ignore"):
        run_graph(arch["graph"], env, P)
    logits = np.where(avail, env["logits"], -np.inf)
    mem_next = env["mem_next"] if arch["memory_size"] else np.zeros(0, np.float32)
    a = int(np.argmax(logits))                                   # first maximal index
    if not np.isfinite(logits[np.asarray(avail, bool)]).all():
        move = ("NONFINITE", None, None)                         # counted as an invalid action
    elif a == CLICK:
        cell = int(np.argmax(env["click"][0]))                   # first cell in row-major order
        move = (ACTIONS[a], cell // 64, cell % 64)
    else:
        move = (ACTIONS[a], None, None)
    return (move, logits, mem_next, env) if return_env else (move, logits, mem_next)
```

**Example: the study's standard fixed actor** written in this language (57,289 parameters,
1,331,914,880 MACs per step). You may reuse it, change it, or design a different network.

```python
def fixed_actor_arch(T=128):
    """The study's fixed actor (wsp.actor, 57,289 parameters) written in this language."""
    return {"memory_size": 64, "graph": [
        {"id": "x", "op": "concat", "in": ["x0", "coords"]},              # 19 channels
        {"id": "c1", "op": "conv", "in": ["x"], "w": "W1", "b": "b1"},    # 5x5 -> 32
        {"id": "f1", "op": "relu", "in": ["c1"]},
        {"id": "c2", "op": "conv", "in": ["f1"], "w": "W2", "b": "b2"},   # 3x3 -> 32
        {"id": "f2", "op": "relu", "in": ["c2"]},
        {"id": "rm", "op": "rowmax", "in": ["f2"]},
        {"id": "cm", "op": "colmax", "in": ["f2"]},
        {"id": "f2a", "op": "concat", "in": ["f2", "rm", "cm"]},          # 96
        {"id": "c3", "op": "conv", "in": ["f2a"], "w": "W3", "b": "b3"},  # 1x1 -> 32
        {"id": "f3", "op": "relu", "in": ["c3"]},
        {"id": "drive", "op": "conv", "in": ["f3"], "w": "Uh", "b": "bh"},  # 1x1 -> 16
        {"id": "H", "op": "loop", "T": T,                                 # H = relu(conv3x3(H) + drive)
         "state": [{"id": "h", "zeros": [16, 64, 64]}],
         "body": [{"id": "hc", "op": "conv", "in": ["h"], "w": "Wh"},     # 3x3, no bias
                  {"id": "hs", "op": "add", "in": ["hc", "drive"]},
                  {"id": "hn", "op": "relu", "in": ["hs"]}],
         "next": {"h": "hn"}},
        {"id": "g1", "op": "gmax", "in": ["H.h"]},
        {"id": "g2", "op": "gmean", "in": ["H.h"]},
        {"id": "g3", "op": "gmax", "in": ["f3"]},
        {"id": "g4", "op": "gmean", "in": ["f3"]},
        {"id": "g", "op": "vconcat", "in": ["g1", "g2", "g3", "g4"]},     # 96
        {"id": "zl", "op": "dense", "in": ["g"], "w": "Wz", "b": "bz"},
        {"id": "z", "op": "vrelu", "in": ["zl"]},                         # 128
        {"id": "m1", "op": "dense", "in": ["mem"], "w": "Wm"},
        {"id": "m2", "op": "dense", "in": ["z"], "w": "Wmz"},
        {"id": "m3", "op": "dense", "in": ["aprev"], "w": "Wma", "b": "bm"},
        {"id": "m12", "op": "vadd", "in": ["m1", "m2"]},
        {"id": "m123", "op": "vadd", "in": ["m12", "m3"]},
        {"id": "mem_next", "op": "vtanh", "in": ["m123"]},                # 64
        {"id": "zm", "op": "vconcat", "in": ["z", "mem_next"]},           # 192
        {"id": "logits", "op": "dense", "in": ["zm"], "w": "Wo", "b": "bo"},
        {"id": "fh", "op": "concat", "in": ["f3", "H.h"]},                # 48
        {"id": "click", "op": "conv", "in": ["fh"], "w": "Wc", "b": "bc"},  # 1x1 -> 1
    ]}
```

Its parameter shapes (1x1 kernels are stored as (out, in, 1, 1)):
    W1   (32, 19, 5, 5)
    b1   (32,)
    W2   (32, 32, 3, 3)
    b2   (32,)
    W3   (32, 96, 1, 1)
    b3   (32,)
    Wh   (16, 16, 3, 3)
    Uh   (16, 32, 1, 1)
    bh   (16,)
    Wz   (128, 96)
    bz   (128,)
    Wm   (64, 64)
    Wmz  (64, 128)
    Wma  (64, 8)
    bm   (64,)
    Wo   (8, 192)
    bo   (8,)
    Wc   (1, 48, 1, 1)
    bc   (1,)


## Your task: design the network and write its weights

Write a Python function `build()` that returns `{"graph": [...], "memory_size": M, "params": {name:
array}}`: a network in the language above together with all of its weights. The network must make
the actor play this game well **by itself on levels and layouts it has never seen**
(the evaluation uses held-out levels and perturbed start states).
Behaviour that only replays the support levels will fail.

Rules:
- Any architecture expressible in the language within the budget is allowed (the fixed actor above,
  a modification of it, or a different design).
- Construct the weights deliberately (analytically / by design). Do not train, optimise, fit,
  search over weights, or simulate the game inside `build()`; it receives no data.
- Use only numpy and math (plus pure-Python standard modules for bookkeeping). `build()` takes no
  arguments, does not read or write anything, and must finish within 60 seconds.
- An invalid network (unknown op or field, shape mismatch, undefined or unused name, non-finite
  value, limit or budget exceeded) counts as a failed submission.

Reply with a short explanation of the architecture and mechanism you implemented (which nodes and
channels compute what), then exactly one ```python code block containing the complete program with
`build()`.
