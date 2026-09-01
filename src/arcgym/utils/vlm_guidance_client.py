from __future__ import annotations

import base64
import json
import os
import queue
import threading
import time
import urllib.request
from typing import Any

import cv2
import numpy as np
import torch


def wrap_coach_text(text: str, max_chars_per_line: int = 55) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text or "").replace("\r\n", "\n").split("\n"):
        words = paragraph.strip().split()
        if not words:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        current: list[str] = []
        for word in words:
            candidate = " ".join(current + [word])
            if len(candidate) > max_chars_per_line and current:
                lines.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            lines.append(" ".join(current))
    return lines or [""]


def format_navigation_simple_coach_for_ui(coach_output: dict[str, Any]) -> list[str]:
    def clean(value: Any, fallback: str) -> str:
        text = str(value or "").replace("\r", " ").replace("\n", " ").strip(" -:\t")
        text = " ".join(text.split()).replace("...", "").strip()
        return text or fallback

    def qwen_generated_lines() -> list[str]:
        text = str(coach_output.get("model_display_text") or coach_output.get("coach_text") or "").strip()
        if not text:
            return []
        lines: list[str] = []
        allowed_prefixes = (
            "current view status:",
            "current visible lumen direction:",
            "evidence used:",
            "predicted lumen direction:",
            "cavity:",
            "lumen:",
            "cue:",
            "action:",
            "guard:",
        )
        for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            line = clean(raw_line, "")
            if line and line.lower().startswith(allowed_prefixes):
                lines.append(line)
            if len(lines) >= 7:
                break
        return lines

    def paragraph_fallback() -> dict[str, str]:
        text = clean(coach_output.get("coach_text") or coach_output.get("raw_text") or coach_output.get("raw_model_output"), "")
        if not text:
            return {}
        sentences = [part.strip() for part in text.replace(";", ".").split(".") if part.strip()]
        cue = ""
        action = ""
        guard = ""
        for sentence in sentences:
            lower = sentence.lower()
            if not cue and any(token in lower for token in ("lumen", "view", "visibility", "center", "reward", "risk", "wall", "texture")):
                cue = sentence
            if not action and any(token in lower for token in ("advance", "withdraw", "re-center", "recenter", "slow", "hold", "adjust", "consider", "reassess", "reverse")):
                action = sentence
            if not guard and any(token in lower for token in ("avoid", "do not", "don't", "excessive", "aggressive", "pushing", "rotation", "until", "wait")):
                guard = sentence
        if not cue and sentences:
            cue = sentences[0]
        if not action:
            action = "Continue the current constrained command."
        if not guard:
            guard = "Reassess if the lumen degrades."
        return {"cue": cue, "action": action, "guard": guard}

    derived = paragraph_fallback()
    source = str(coach_output.get("source", "")).strip()
    title = "Coach: Qwen lumen coach" if "qwen_lumen" in source else "Coach: Qwen-VL"
    generated_lines = qwen_generated_lines()
    if generated_lines and "qwen_lumen" in source:
        return [title] + generated_lines
    return [
        title,
        f"Cue: {clean(coach_output.get('cue', coach_output.get('focus')), derived.get('cue', 'No harmful trend detected.'))}",
        f"Action: {clean(coach_output.get('action', coach_output.get('next')), derived.get('action', 'Continue the current constrained command.'))}",
        f"Guard: {clean(coach_output.get('guard', coach_output.get('avoid')), derived.get('guard', 'Reassess if the lumen degrades.'))}",
    ]


class VLMGuidanceClient:
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8765/predict",
        update_interval: int = 5,
        timeout: float = 120.0,
        async_mode: bool = True,
        show_window: bool = True,
        request_visualization: bool = False,
        window_name: str = "VLM Colonoscopy Guidance",
        panel_size: tuple[int, int] = (336, 448),
        text_width: int = 560,
        sync_display: bool = True,
        decouple_display: bool | None = None,
        display_interval: int | None = None,
        guidance_display_interval: int | None = None,
    ):
        self.server_url = server_url
        self.update_interval = update_interval
        self.timeout = timeout
        self.async_mode = async_mode
        self.show_window = show_window
        self.request_visualization = request_visualization
        self.window_name = window_name
        self.panel_size = panel_size
        self.text_width = text_width
        self.sync_display = sync_display
        if decouple_display is None:
            decouple_display = os.environ.get("VLM_GUIDANCE_DECOUPLE_DISPLAY", "1") != "0"
        if display_interval is None:
            display_interval = int(os.environ.get("VLM_GUIDANCE_DISPLAY_INTERVAL", "1"))
        if guidance_display_interval is None:
            guidance_display_interval = int(os.environ.get("VLM_GUIDANCE_GUIDANCE_DISPLAY_INTERVAL", str(update_interval)))
        self.decouple_display = bool(decouple_display)
        self.display_interval = max(1, int(display_interval))
        self.guidance_display_interval = max(1, int(guidance_display_interval))
        self.frame_id = 0
        self.last_response: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._display_worker: threading.Thread | None = None
        self._display_queue: queue.Queue[tuple[np.ndarray | None, dict[str, Any] | None]] = queue.Queue(maxsize=1)
        self._display_frame_id = -1
        self._last_guidance_panel_rgb: np.ndarray | None = None
        self._last_text_panel: np.ndarray | None = None
        self._last_arrow_title = "Guidance arrow"
        self._pending_frame_id: int | None = None
        self._last_response_rgb: np.ndarray | None = None
        self._new_response_available = False
        self._closed = False

    def maybe_update(
        self,
        rgb: Any,
        action_6dof: Any | None = None,
        reward: Any | None = None,
        force: bool = False,
    ) -> dict[str, Any] | None:
        should_send = force or (self.frame_id % self.update_interval == 0)
        display_this_frame = self.show_window and (self.frame_id % self.display_interval == 0)
        rgb_u8 = self._to_rgb_uint8(rgb) if rgb is not None and (display_this_frame or should_send) else None
        if should_send:
            self._submit_frame(rgb, action_6dof, reward, force=force, rgb_u8=rgb_u8)

        response = self._get_latest_response()
        if display_this_frame:
            self._show_live_visualization(rgb_u8, response)
        self.frame_id += 1
        return response

    def _submit_frame(
        self,
        rgb: Any,
        action_6dof: Any | None,
        reward: Any | None,
        force: bool = False,
        rgb_u8: np.ndarray | None = None,
    ) -> bool:
        if rgb is None and rgb_u8 is None:
            return False
        if self.async_mode and not force and self._request_in_flight():
            return False
        if rgb_u8 is None:
            rgb_u8 = self._to_rgb_uint8(rgb)

        frame_id = self.frame_id
        timestamp = time.time()
        action = self._action_list(action_6dof)
        reward_value = self._reward_float(reward)
        if self.async_mode and not force:
            worker = threading.Thread(
                target=self._post_frame_background,
                args=(frame_id, timestamp, rgb_u8.copy(), action, reward_value),
                daemon=True,
            )
            with self._lock:
                self._pending_frame_id = int(frame_id)
                self._worker = worker
            worker.start()
            return True

        payload = self._make_payload(frame_id, timestamp, rgb_u8, action, reward_value)

        try:
            response = self._post_json(payload)
        except Exception as exc:
            response = {"ok": False, "error": str(exc), "frame_id": payload["frame_id"], "source": "client_error"}
        with self._lock:
            self.last_response = response
            self._last_response_rgb = rgb_u8.copy()
            self._new_response_available = True
            self._pending_frame_id = None
        return True

    def _request_in_flight(self) -> bool:
        with self._lock:
            worker = self._worker
            return worker is not None and worker.is_alive()

    def _make_payload(
        self,
        frame_id: int,
        timestamp: float,
        rgb_u8: np.ndarray,
        action: list[float],
        reward_value: float,
    ) -> dict[str, Any]:
        return {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "image_base64": self._encode_png_base64(rgb_u8),
            "action_6dof": action,
            "reward": reward_value,
            "return_visualization": self.request_visualization,
        }

    def _post_frame_background(
        self,
        frame_id: int,
        timestamp: float,
        rgb_u8: np.ndarray,
        action: list[float],
        reward_value: float,
    ) -> None:
        try:
            payload = self._make_payload(frame_id, timestamp, rgb_u8, action, reward_value)
        except Exception as exc:
            response = {"ok": False, "error": str(exc), "frame_id": frame_id, "source": "client_encode_error"}
            with self._lock:
                self.last_response = response
                self._pending_frame_id = None
            return
        try:
            response = self._post_json(payload)
        except Exception as exc:
            response = {"ok": False, "error": str(exc), "frame_id": frame_id, "source": "client_error"}
        with self._lock:
            self.last_response = response
            self._last_response_rgb = rgb_u8.copy()
            self._new_response_available = True
            self._pending_frame_id = None

    def _get_latest_response(self) -> dict[str, Any] | None:
        with self._lock:
            if self.last_response is None:
                if self._pending_frame_id is None:
                    return None
                return {"ok": False, "pending": True, "pending_frame_id": self._pending_frame_id}
            response = dict(self.last_response)
            response["client_pending"] = self._pending_frame_id is not None
            response["client_pending_frame_id"] = self._pending_frame_id
            response["client_new_response"] = self._new_response_available
            self._new_response_available = False
            return response

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.server_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _to_rgb_uint8(rgb: Any) -> np.ndarray:
        if isinstance(rgb, torch.Tensor):
            arr = rgb.detach().cpu().numpy()
        else:
            arr = np.asarray(rgb)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in {3, 15}:
            if arr.shape[0] == 15:
                arr = arr[-3:]
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            if np.nanmax(arr) <= 1.0:
                arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr[..., :3]

    @staticmethod
    def _encode_png_base64(rgb_u8: np.ndarray) -> str:
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("Could not encode RGB frame for VLM guidance server")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    @staticmethod
    def _action_list(action: Any | None) -> list[float]:
        if action is None:
            return [0.0] * 6
        if isinstance(action, torch.Tensor):
            arr = action.detach().cpu().numpy()
        else:
            arr = np.asarray(action)
        if arr.ndim > 1:
            arr = arr[0]
        values = arr.astype(float).reshape(-1).tolist()
        if len(values) < 6:
            values = values + [0.0] * (6 - len(values))
        return values[:6]

    @staticmethod
    def _reward_float(reward: Any | None) -> float:
        if reward is None:
            return 0.0
        if isinstance(reward, torch.Tensor):
            reward = reward.detach().cpu().reshape(-1)[0].item()
        elif isinstance(reward, np.ndarray):
            reward = reward.reshape(-1)[0].item()
        elif isinstance(reward, (list, tuple)):
            reward = reward[0]
        return float(reward)

    def _show_visualization(self, jpeg_base64: str) -> None:
        data = base64.b64decode(jpeg_base64)
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        cv2.imshow(self.window_name, image)
        cv2.waitKey(1)

    def _show_live_visualization(self, rgb_u8: np.ndarray | None, response: dict[str, Any] | None) -> None:
        if rgb_u8 is None:
            if response and response.get("visualization_jpeg_base64"):
                self._show_visualization(response["visualization_jpeg_base64"])
            return
        if self.decouple_display:
            self._enqueue_display(rgb_u8, response)
            return
        with self._lock:
            guidance_rgb = None if self._last_response_rgb is None else self._last_response_rgb.copy()
        panel = self._make_live_panel(rgb_u8, response, guidance_rgb=guidance_rgb, update_guidance_panel=True)
        cv2.imshow(self.window_name, panel)
        cv2.waitKey(1)

    def _enqueue_display(self, rgb_u8: np.ndarray | None, response: dict[str, Any] | None) -> None:
        if self._display_worker is None or not self._display_worker.is_alive():
            self._display_worker = threading.Thread(target=self._display_loop, daemon=True)
            self._display_worker.start()
        item = (None if rgb_u8 is None else rgb_u8.copy(), None if response is None else dict(response))
        try:
            self._display_queue.put_nowait(item)
        except queue.Full:
            try:
                self._display_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._display_queue.put_nowait(item)
            except queue.Full:
                pass

    def _display_loop(self) -> None:
        while not self._closed:
            try:
                rgb_u8, response = self._display_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if rgb_u8 is None:
                continue
            with self._lock:
                guidance_rgb = None if self._last_response_rgb is None else self._last_response_rgb.copy()
            self._display_frame_id += 1
            update_guidance_panel = (
                self._last_guidance_panel_rgb is None
                or bool((response or {}).get("client_new_response", False))
                or self._display_frame_id % self.guidance_display_interval == 0
            )
            panel = self._make_live_panel(
                rgb_u8,
                response,
                guidance_rgb=guidance_rgb,
                update_guidance_panel=update_guidance_panel,
            )
            cv2.imshow(self.window_name, panel)
            cv2.waitKey(1)

    def _make_live_panel(
        self,
        rgb_u8: np.ndarray,
        response: dict[str, Any] | None,
        guidance_rgb: np.ndarray | None = None,
        update_guidance_panel: bool = True,
    ) -> np.ndarray:
        panel_w, panel_h = self.panel_size
        text_h = max(panel_h, 448)
        guidance_source = guidance_rgb if guidance_rgb is not None else rgb_u8
        display_source = guidance_source if self.sync_display and guidance_rgb is not None else rgb_u8
        raw = self._resize_rgb(display_source, panel_w, panel_h)
        if update_guidance_panel or self._last_guidance_panel_rgb is None or self._last_text_panel is None:
            arrow = self._resize_rgb(guidance_source, panel_w, panel_h)
            self._draw_guidance_arrow(arrow, response)
            text_panel = np.full((text_h, self.text_width, 3), 245, dtype=np.uint8)
            self._draw_text_panel(text_panel, response)
            self._last_guidance_panel_rgb = arrow
            self._last_text_panel = text_panel
            self._last_arrow_title = "Guidance arrow"
            if response and response.get("client_pending") and not response.get("client_new_response"):
                self._last_arrow_title = "Guidance arrow (last guidance frame)"
        else:
            arrow = self._last_guidance_panel_rgb
            text_panel = self._last_text_panel

        title_h = 34
        total_w = panel_w * 2 + self.text_width
        canvas_h = max(panel_h, text_h) + title_h
        canvas = np.zeros((canvas_h, total_w, 3), dtype=np.uint8)
        canvas[title_h:, :panel_w] = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        canvas[title_h:, panel_w : panel_w * 2] = cv2.cvtColor(arrow, cv2.COLOR_RGB2BGR)
        canvas[title_h:, panel_w * 2 :] = text_panel
        raw_title = "Raw endoscope"
        if self.sync_display and guidance_rgb is not None:
            raw_title = "Raw endoscope (guidance frame)"
        self._put_text(canvas, raw_title, (16, 23), scale=0.55, thickness=1, color=(245, 245, 245))
        self._put_text(canvas, self._last_arrow_title, (panel_w + 16, 23), scale=0.55, thickness=1, color=(245, 245, 245))
        return canvas

    @staticmethod
    def _resize_rgb(rgb_u8: np.ndarray, width: int, height: int) -> np.ndarray:
        return cv2.resize(rgb_u8[..., :3], (width, height), interpolation=cv2.INTER_LINEAR)

    def _draw_guidance_arrow(self, arrow_rgb: np.ndarray, response: dict[str, Any] | None) -> None:
        h, w = arrow_rgb.shape[:2]
        center = (w // 2, h // 2)
        steering = ""
        if response:
            steering = str(response.get("guidance", {}).get("steering", ""))

        dx = -1 if "left" in steering else 1 if "right" in steering else 0
        dy = -1 if "up" in steering else 1 if "down" in steering else 0

        if dx == 0 and dy == 0 and response:
            roi = response.get("roi_features", {}) or {}
            dx = float(roi.get("normalized_offset_x", roi.get("predicted_roi_normalized_offset_x", 0.0)))
            dy = float(roi.get("normalized_offset_y", roi.get("predicted_roi_normalized_offset_y", 0.0)))
            if abs(dx) < 0.08:
                dx = 0
            if abs(dy) < 0.08:
                dy = 0

        length = int(min(w, h) * 0.28)
        end = (int(center[0] + float(dx) * length), int(center[1] + float(dy) * length))
        cv2.circle(arrow_rgb, center, 9, (255, 0, 0), -1)
        cv2.line(arrow_rgb, (center[0] - 18, center[1]), (center[0] + 18, center[1]), (255, 255, 255), 3)
        cv2.line(arrow_rgb, (center[0], center[1] - 18), (center[0], center[1] + 18), (255, 255, 255), 3)
        if dx != 0 or dy != 0:
            cv2.arrowedLine(arrow_rgb, center, end, (255, 255, 0), 9, tipLength=0.35)
            cv2.circle(arrow_rgb, end, 7, (255, 255, 0), -1)

    def _draw_text_panel(self, image: np.ndarray, response: dict[str, Any] | None) -> None:
        x = 18
        y = 32
        guidance = (response or {}).get("guidance", {}) if response else {}
        quality = (response or {}).get("quality_assessment", {}) if response else {}
        reward = (response or {}).get("reward_features", {}) if response else {}
        coach_output = {}
        if response:
            candidate = response.get("coaching_action", {})
            if (
                isinstance(candidate, dict)
                and candidate
                and str(candidate.get("source", "")).startswith("guidance_arbiter")
            ):
                coach_output = candidate
            qwen_candidates = [
                response.get("qwen_lumen_coach", {}),
                response.get("frozen_qwen_vl_coach", {}),
            ]
            if not coach_output:
                for candidate in qwen_candidates:
                    if isinstance(candidate, dict) and candidate and candidate.get("status") == "ok":
                        coach_output = candidate
                        break
            if not coach_output:
                candidate = response.get("coaching_action", {})
                if isinstance(candidate, dict) and candidate:
                    coach_output = candidate
            if not coach_output:
                for candidate in qwen_candidates:
                    if isinstance(candidate, dict) and candidate:
                        coach_output = candidate
                        break
        coach_text = str(coach_output.get("coach_text", "")).strip() if isinstance(coach_output, dict) else ""
        action_coaching = str((response or {}).get("coaching", "Waiting for VLM guidance."))
        pending = bool((response or {}).get("client_pending", False)) if response else False
        frame_id = (response or {}).get("frame_id") if response else None
        pending_frame_id = (response or {}).get("client_pending_frame_id") if response else None
        ok = bool((response or {}).get("ok", False)) if response else False
        source = str((response or {}).get("source", "")) if response else ""
        error = str((response or {}).get("error", "")) if response else ""

        self._put_text(image, "Instruction", (x, y), scale=0.68, thickness=2)
        y += 25
        if error:
            self._put_text(image, f"Status: error frame {frame_id}", (x, y), scale=0.45, color=(20, 20, 180))
            y += 23
            for line in self._wrap_text(error, max_chars=48)[:2]:
                self._put_text(image, line, (x, y), scale=0.39, color=(20, 20, 180))
                y += 18
        elif pending:
            status = "Status: showing last guidance"
            if pending_frame_id is not None:
                status += f", updating frame {pending_frame_id}"
            self._put_text(image, status, (x, y), scale=0.40, color=(80, 80, 80))
            y += 20
        elif frame_id is not None:
            self._put_text(image, f"Status: guidance frame {frame_id}", (x, y), scale=0.40, color=(80, 80, 80))
            y += 20
        if ok and source:
            self._put_text(image, f"Source: {source}", (x, y), scale=0.38, color=(80, 80, 80))
            y += 18
        for line in [
            f"Advance: {guidance.get('advance', 'hold')}",
            f"Steering: {guidance.get('steering', 'hold')}",
            f"Rotation: {guidance.get('rotation', 'hold')}",
            f"Urgency: {guidance.get('urgency', 'low')}    Confidence: {float(guidance.get('confidence', 0.0)):.2f}",
        ]:
            self._put_text(image, line, (x, y), scale=0.47)
            y += 20

        y += 5
        self._put_text(image, "Quality", (x, y), scale=0.62, thickness=2)
        y += 24
        reward_value = reward.get("reward_t", reward.get("mean_reward_window", 0.0))
        for line in [
            f"Reward: {float(reward_value):.3f}    Trend: {quality.get('reward_trend', 'unknown')}",
            f"Risk: {quality.get('risk_level', 'medium')}    Jitter: {quality.get('jitter', 'unknown')}",
            f"Lumen: {quality.get('lumen_visibility', 'unknown')}",
        ]:
            self._put_text(image, line, (x, y), scale=0.47)
            y += 20

        if response:
            roi = response.get("roi_features", {}) or {}
            arbiter = response.get("guidance_arbiter", {}) or {}
            qwen = response.get("qwen_lumen_coach", response.get("frozen_qwen_vl_coach", {})) or {}
            temporal = response.get("qwen_temporal_context", {}) or {}
            current_status = str(response.get("current_view_status", arbiter.get("current_view_status", "unknown")))
            current_dir = str(response.get("current_visible_direction", arbiter.get("current_visible_direction", "unclear")))
            temporal_allowed = bool(response.get("temporal_context_allowed", arbiter.get("temporal_context_allowed", True)))
            final_source = str(response.get("final_source", arbiter.get("final_source", "unknown")))
            qwen_override = bool(response.get("qwen_temporal_override_error", qwen.get("qwen_temporal_override_error", False)))
            if current_status in {"visible_opening", "visible_but_depth_uncertain"}:
                policy_label = "CURRENT FRAME PRIMARY"
            elif final_source == "qwen_temporal_context":
                policy_label = "TEMPORAL RECOVERY"
            else:
                policy_label = "POLICY"
            if not temporal_allowed:
                policy_label += " | TEMPORAL CONTEXT BLOCKED"
            if qwen_override:
                policy_label += " | QWEN TEMPORAL OVERRIDE BLOCKED"

            y += 4
            self._put_text(image, policy_label, (x, y), scale=0.37, thickness=1, color=(40, 80, 160))
            y += 17
            opening_score = float(response.get("current_visible_opening_score", arbiter.get("current_visible_opening_score", 0.0)) or 0.0)
            trust_score = float(response.get("depth_roi_trust_score", arbiter.get("depth_roi_trust_score", 0.0)) or 0.0)
            roi_area = roi.get("predicted_roi_area_ratio", roi.get("roi_area_ratio", 0.0))
            try:
                roi_area_text = f"{float(roi_area):.3f}"
            except Exception:
                roi_area_text = str(roi_area)
            compact_lines = [
                f"View: {current_status} | dir {current_dir} | score {opening_score:.2f}",
                f"Depth/ROI: trust {trust_score:.2f} | dir {arbiter.get('roi_location', 'n/a')} | area {roi_area_text} | lost {roi.get('predicted_lost_lumen', roi.get('lost_lumen', False))}",
                f"Memory: valid {temporal.get('valid_lumen_context_count', response.get('valid_lumen_context_count', 0))} | invalid {temporal.get('invalid_recent_summary_count', response.get('invalid_recent_summary_count', 0))}",
                f"Qwen: view {qwen.get('current_view_status', 'n/a')} | evidence {qwen.get('evidence_used', 'n/a')} | pred {qwen.get('predicted_lumen_direction', qwen.get('lumen', 'n/a'))}",
                f"Final: {final_source}",
            ]
            for line in compact_lines:
                for wrapped_line in self._wrap_text(line, max_chars=70)[:2]:
                    self._put_text(image, wrapped_line, (x, y), scale=0.34)
                    y += 15
            final_reason = str(response.get("final_reason", arbiter.get("final_reason", arbiter.get("reason", ""))))
            if final_reason:
                for wrapped_line in self._wrap_text("Reason: " + final_reason, max_chars=70)[:2]:
                    self._put_text(image, wrapped_line, (x, y), scale=0.34, color=(70, 70, 70))
                    y += 15

        y += 5
        self._put_text(image, "Coaching", (x, y), scale=0.62, thickness=2)
        y += 24
        coach_status = str(coach_output.get("status", "")).strip() if isinstance(coach_output, dict) else ""
        display_mode = str(coach_output.get("display_mode", "navigation_simple")).strip() if isinstance(coach_output, dict) else "navigation_simple"
        if coach_text:
            coaching = coach_text
        elif coach_status == "running":
            coaching = "Coach is analyzing the recent maneuver..."
        elif coach_status == "failed":
            coaching = "Coach feedback is currently unavailable. Continue using the real-time constrained guidance."
        else:
            coaching = action_coaching
        if isinstance(coach_output, dict) and display_mode == "navigation_simple":
            for line in format_navigation_simple_coach_for_ui(coach_output):
                for wrapped_line in self._wrap_text(line, max_chars=58):
                    self._put_text(image, wrapped_line, (x, y), scale=0.39)
                    y += 17
            return

        if isinstance(coach_output, dict) and (coach_text or coach_status):
            title = "Coach: Frozen Qwen-VL"
            if coach_status:
                title += f" ({coach_status})"
            self._put_text(image, title, (x, y), scale=0.42, color=(80, 80, 80))
            y += 20
        for line in wrap_coach_text(coaching, max_chars_per_line=46):
            self._put_text(image, line, (x, y), scale=0.39)
            y += 17

    @staticmethod
    def _put_text(
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        scale: float = 0.55,
        thickness: int = 1,
        color: tuple[int, int, int] = (20, 20, 20),
    ) -> None:
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _wrap_text(text: str, max_chars: int) -> list[str]:
        words = text.split()
        lines: list[str] = []
        current: list[str] = []
        for word in words:
            candidate = " ".join(current + [word])
            if len(candidate) > max_chars and current:
                lines.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            lines.append(" ".join(current))
        return lines or [""]
