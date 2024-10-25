import asyncio
import base64
import os
import pyautogui
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypedDict
from uuid import uuid4

from anthropic.types.beta import BetaToolComputerUse20241022Param

from .base import BaseAnthropicTool, ToolError, ToolResult
from .run import run

OUTPUT_DIR = "/tmp/outputs"

TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50

Action = Literal[
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "screenshot",
    "cursor_position",
    # minecraft
    "left_down",
    "left_up",
    "hold_arrow_up",
    "release_arrow_up",
    "hold_arrow_down",
    "release_arrow_down",
    "hold_arrow_left",
    "release_arrow_left",
    "hold_arrow_right",
    "release_arrow_right",
]


class Resolution(TypedDict):
    width: int
    height: int


# sizes above XGA/WXGA are not recommended (see README.md)
# scale down to one of these targets if ComputerTool._scaling_enabled is set
MAX_SCALING_TARGETS: dict[str, Resolution] = {
    "XGA": Resolution(width=1024, height=768),  # 4:3
    "WXGA": Resolution(width=1280, height=800),  # 16:10
    "FWXGA": Resolution(width=1366, height=768),  # ~16:9
}


class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "api"


class ComputerToolOptions(TypedDict):
    display_height_px: int
    display_width_px: int
    display_number: int | None

# <MINECRAFT_GAME_INSTRUCTIONS>
# You may be asked to play a game of minecraft.  Here are some instructions:
# * For movement, WASD is disabled.  Instead, use the hold_arrow_left, release_arrow_left, hold_arrow_right, release_arrow_right, hold_arrow_up, release_arrow_up, hold_arrow_down, release_arrow_down tool calls.
# * You can control the minecraft game with your computer tool.  Use the minecraft_button tool call to hold down the left mouse button, and the left_up tool call to release it.
# * Use the hold_arrow_up tool call to hold down the up arrow key, and the release_arrow_up tool call to release it.
# * Use the hold_arrow_down tool call to hold down the down arrow key, and the release_arrow_down tool call to release it.
# * Use the hold_arrow_left tool call to hold down the left arrow key, and the release_arrow_left tool call to release it.
# * Use the hold_arrow_right tool call to hold down the right arrow key, and the release_arrow_right tool call to release it.
# </MINECRAFT_INSTRUCTIONS>
def chunks(s: str, chunk_size: int) -> list[str]:
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]

class GameTool(BaseAnthropicTool):
    """
    A tool that allows the agent to interact with the screen, keyboard, and mouse of the current computer. Also allows the agent to control a minecraft game.
    The tool parameters are defined by Anthropic and are not editable.
    """

    name: Literal["computer"] = "computer"
    api_type: Literal["computer_20241022"] = "computer_20241022"
    width: int
    height: int
    description: str = """
    Use a mouse and keyboard to interact with a computer, and take screenshots.
    * This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
    * Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try taking another screenshot.
    * The screen's resolution is {{ display_width_px }}x{{ display_height_px }}.
    * The display number is {{ display_number }}
    * Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
    * If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
    * Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.
    In Minecraft, the standard controls are:
    space: Jump
    mouse_move: Look around
    left_down: Break blocks/attack
    right click: Place blocks/interact
    For movement in minecraft, WASD is disabled.  Instead, use the hold_arrow_left, release_arrow_left, hold_arrow_right, release_arrow_right, hold_arrow_up, release_arrow_up, hold_arrow_down, release_arrow_down tool calls.
    """
    input_schema = {
    "properties": {
        "action": {
            "description": """The action to perform. The available actions are:
                * `key`: Press a key or key-combination on the keyboard.
                  - This supports cliclick's `key` syntax.
                  - All possible keys are: arrow-up, brightness-down, brightness-up, delete, end, enter, esc, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, fwd-delete, home, keys-light-down, keys-light-toggle, keys-light-up, mute, num-0, num-1, num-2, num-3, num-4, num-5, num-6, num-7, num-8, num-9, num-clear, num-divide, num-enter, num-equals, num-minus, num-multiply, num-plus, page-down, page-up, play-next, play-pause, play-previous, return, space, tab, volume-down, volume-up
                * `type`: Type a string of text on the keyboard.
                * `cursor_position`: Get the current (x, y) pixel coordinate of the cursor on the screen.
                * `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
                * `left_click`: Click the left mouse button.
                * `right_click`: Click the right mouse button.
                * `middle_click`: Click the middle mouse button.
                * `double_click`: Double-click the left mouse button.
                * `screenshot`: Take a screenshot of the screen.
            """,
            "enum": [
                "key",
                "type",
                "mouse_move",
                "left_click",
                # "left_click_drag",
                "right_click",
                "middle_click",
                "double_click",
                "screenshot",
                "cursor_position",
                # minecraft
                "left_down",
                "left_up",
                "hold_arrow_up",
                "release_arrow_up",
                "hold_arrow_down",
                "release_arrow_down",
                "hold_arrow_left",
                "release_arrow_left",
                "hold_arrow_right",
                "release_arrow_right",
            ],
            "type": "string",
        },
        "coordinate": {
            "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.",
            "type": "array",
        },
        "text": {
            "description": "Required only by `action=type` and `action=key`.",
            "type": "string",
        },
    },
    "required": ["action"],
    "type": "object",
    }

    display_num: int | None

    _screenshot_delay = 2.0
    _scaling_enabled = True

    @property
    def options(self) -> ComputerToolOptions:
        width, height = self.scale_coordinates(
            ScalingSource.COMPUTER, self.width, self.height
        )
        return {
            "display_width_px": width,
            "display_height_px": height,
            "display_number": self.display_num,
        }

    def to_params(self) -> BetaToolComputerUse20241022Param:
        return {"name": self.name, "type": self.api_type, **self.options}

    def __init__(self):
        super().__init__()

        self.width = int(os.getenv("WIDTH") or 0)
        self.height = int(os.getenv("HEIGHT") or 0)
        assert self.width and self.height, "WIDTH, HEIGHT must be set"
        if (display_num := os.getenv("DISPLAY_NUM")) is not None:
            self.display_num = int(display_num)
            self._display_prefix = f"DISPLAY=:{self.display_num} "
        else:
            self.display_num = None
            self._display_prefix = ""

    async def __call__(
        self,
        *,
        action: Action,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        **kwargs,
    ):
        if action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                raise ToolError(f"coordinate is required for {action}")
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if not isinstance(coordinate, list) or len(coordinate) != 2:
                raise ToolError(f"{coordinate} must be a tuple of length 2")
            if not all(isinstance(i, int) and i >= 0 for i in coordinate):
                raise ToolError(f"{coordinate} must be a tuple of non-negative ints")

            x, y = self.scale_coordinates(
                ScalingSource.API, coordinate[0], coordinate[1]
            )

            if action == "mouse_move":
                pyautogui.moveTo(x, y)
                return ToolResult(output=f"Moved mouse to {x},{y}")
            elif action == "left_click_drag":
                pyautogui.dragTo(x, y)
                return ToolResult(output=f"Dragged mouse to {x},{y}")

        if action in ("key", "type"):
            if text is None:
                raise ToolError(f"text is required for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")
            if not isinstance(text, str):
                raise ToolError(output=f"{text} must be a string")

            if action == "key":
                if text.lower() in 'wasd':
                    pyautogui.keyDown(text.lower())
                    pyautogui.sleep(1)
                    pyautogui.keyUp(text.lower())
                elif text.lower() in 'abcdefghijklmnopqrstuvwxyz':
                    pyautogui.press(text.lower())
                elif text in "1234567890":
                    pyautogui.press(text)
                elif text.lower() == "return":
                    pyautogui.press('enter')
                elif text.lower() in ("right-arrow", "right", "left-arrow", "left", "up-arrow", "up", "down-arrow", "down"):
                    pyautogui.press(text.split('-')[0].lower())
                else:
                    pyautogui.press(text.lower())
                return ToolResult(output=f"Pressed key: {text}")
            elif action == "type":
                for chunk in chunks(text, TYPING_GROUP_SIZE):
                    pyautogui.write(chunk, interval=TYPING_DELAY_MS/1000)
                screenshot_base64 = (await self.screenshot()).base64_image
                return ToolResult(
                    output=f"Typed: {text}",
                    base64_image=screenshot_base64,
                )

        if action in (
            "left_click",
            "right_click",
            "double_click",
            "middle_click",
            "screenshot",
            "cursor_position",
        ):
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")

            if action == "screenshot":
                return await self.screenshot()
            elif action == "cursor_position":
                x, y = pyautogui.position()
                x, y = self.scale_coordinates(
                    ScalingSource.COMPUTER, x, y
                )
                return ToolResult(output=f"X={x},Y={y}")
            else:
                click_map = {
                    "left_click": pyautogui.click,
                    "right_click": pyautogui.rightClick,
                    "middle_click": pyautogui.middleClick,
                    "double_click": pyautogui.doubleClick,
                }
                click_map[action]()
                return ToolResult(output=f"Performed {action}")

        # minecraft
        if action == "left_down":
            pyautogui.mouseDown()
        elif action == "left_up":
            pyautogui.mouseUp()
        elif action.startswith("hold_arrow_"):
            pyautogui.keyDown(action.split('_')[-1])
        elif action.startswith("release_arrow_"):
            pyautogui.keyUp(action.split('_')[-1])
        else:
            raise ToolError(f"Invalid action: {action}")

        return ToolResult(output=f"Performed {action}")

    async def screenshot(self):
        """Take a screenshot of the current screen and return the base64 encoded image."""
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"screenshot_{uuid4().hex}.png"

        screenshot_cmd = f"{self._display_prefix}screencapture -C {path} -p"

        # resize the screenshot to default width and height
        await self.shell(f"convert {path} -resize {self.width}x{self.height}! {path}", take_screenshot=False)

        result = await self.shell(screenshot_cmd, take_screenshot=False)
        if self._scaling_enabled:
            x, y = self.scale_coordinates(
                ScalingSource.COMPUTER, self.width, self.height
            )
            await self.shell(
                f"convert {path} -resize {x}x{y}! {path}", take_screenshot=False
            )

        if path.exists():
            return result.replace(
                base64_image=base64.b64encode(path.read_bytes()).decode()
            )
        raise ToolError(f"Failed to take screenshot: {result.error}")

    async def shell(self, command: str, take_screenshot=True) -> ToolResult:
        """Run a shell command and return the output, error, and optionally a screenshot."""
        _, stdout, stderr = await run(command)
        base64_image = None

        if take_screenshot:
            # delay to let things settle before taking a screenshot
            await asyncio.sleep(self._screenshot_delay)
            base64_image = (await self.screenshot()).base64_image

        return ToolResult(output=stdout, error=stderr, base64_image=base64_image)

    def scale_coordinates(self, source: ScalingSource, x: int, y: int):
        """Scale coordinates to a target maximum resolution."""
        if not self._scaling_enabled:
            return x, y
        ratio = self.width / self.height
        target_dimension = None
        for dimension in MAX_SCALING_TARGETS.values():
            # allow some error in the aspect ratio - not ratios are exactly 16:9
            if abs(dimension["width"] / dimension["height"] - ratio) < 0.02:
                if dimension["width"] < self.width:
                    target_dimension = dimension
                break
        if target_dimension is None:
            return x, y
        # should be less than 1
        x_scaling_factor = target_dimension["width"] / self.width
        y_scaling_factor = target_dimension["height"] / self.height
        if source == ScalingSource.API:
            if x > self.width or y > self.height:
                raise ToolError(f"Coordinates {x}, {y} are out of bounds")
            # scale up
            return round(x / x_scaling_factor), round(y / y_scaling_factor)
        # scale down
        return round(x * x_scaling_factor), round(y * y_scaling_factor)

