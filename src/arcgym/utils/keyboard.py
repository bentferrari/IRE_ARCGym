"""Keyboard controller for FPV control."""

import numpy as np
import weakref
from collections.abc import Callable

import carb
import omni

from isaaclab.devices.device_base import DeviceBase


class FPVKeyboard(DeviceBase):
    def __init__(self, lin_sensitivity=0.5, rot_sensitivity=0.5):
        """Initialize the keyboard layer.

        Args:
            trans_sensitivity: Magnitude of linear velocity. 
            rot_sensitivity: Magnitude of rotational velocity. 
        """
        # store inputs
        self.lin_sensitivity = lin_sensitivity
        self.rot_sensitivity = rot_sensitivity

        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        # bindings for keyboard to command
        self._create_key_bindings()
        # command buffers
        self._base_command = np.zeros(6)
        # dictionary for additional callbacks
        self._additional_callbacks = dict()

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg="TBD"
        return msg


    """
    Operations
    """

    def reset(self):
        # default flags
        self._base_command.fill(0.0)

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind keyboard.

        A list of available keys are present in the
        `carb documentation <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

        Args:
            key: The keyboard button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func

    def advance(self) -> np.ndarray:
        """Provides the result from keyboard event state.

        Returns:
            3D array containing the linear (x,y) and angular velocity (z).
        """
        return self._base_command

    """
    Internal helpers.
    """

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html
        """
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "R":
                self.reset()
            elif event.input.name in self._INPUT_KEY_MAPPING:
                self._base_command += self._INPUT_KEY_MAPPING[event.input.name]
        # remove the command when un-pressed
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._INPUT_KEY_MAPPING:
                self.reset()
                #self._base_command -= self._INPUT_KEY_MAPPING[event.input.name]
        # additional callbacks
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._additional_callbacks:
                self._additional_callbacks[event.input.name]()

        # since no error, we are fine :)
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            # forward command
            "W": np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) * self.lin_sensitivity,
            # back command
            "S": np.asarray([0.0, 0.0, -1.0, 0.0, 0.0, 0.0]) * self.lin_sensitivity,
            # right command
            "A": np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.lin_sensitivity,
            # left command
            "D": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.lin_sensitivity,
            # up command (positive)
            "Q": np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]) * self.lin_sensitivity,
            # down command (negative)
            "E": np.asarray([0.0, -1.0, 0.0, 0.0, 0.0, 0.0]) * self.lin_sensitivity,
            # pitch command (positive)
            "I": np.asarray([0.0, 0.0, 0.0, -1.0, 0.0, 0.0]) * self.rot_sensitivity,
            # pitch command (negative)
            "K": np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]) * self.rot_sensitivity,
            # yaw command (positive)
            "J": np.asarray([0.0, 0.0, 0.0, 0.0, -1.0, 0.0]) * self.rot_sensitivity,
            # yaw command (negative)
            "L": np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]) * self.rot_sensitivity,
            # roll command (positive)
            "U": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) * self.rot_sensitivity,
            # roll command (negative)
            "O": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]) * self.rot_sensitivity,
        }

