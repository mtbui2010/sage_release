"""
sim_client.py
ZMQ client for both ThorServer (thor_server.py) and ProcThorServer (procthor_server.py).

The client is simulator-agnostic: it speaks the same request/reply protocol to
both servers.  The only difference is which server URL you point it at and which
`simulator_type` value you pass to `reset()`.

Usage — iTHOR (thor_server.py on port 5555):
    client = SimClient(server_url="tcp://localhost:5555", simulator_type="thor")
    result = client.reset("FloorPlan1")
    result = client.step("Pick", "Apple")
    img    = client.get_frame()
    client.stop()

Usage — ProcTHOR (procthor_server.py on port 5556):
    client = SimClient(server_url="tcp://localhost:5556", simulator_type="procthor")
    result = client.reset()          # loads a random house
    result = client.reset(split="val", house_index=7)   # specific house
    result = client.next_house()     # swap to another random house
    result = client.step("MoveTo", "CoffeeMachine")
    img    = client.get_frame()
    client.stop()

Factory helpers:
    client = SimClient.for_thor(host="localhost", port=5555)
    client = SimClient.for_procthor(host="localhost", port=5556)
"""

from __future__ import annotations
import base64
import io
import logging
from typing import Optional

log = logging.getLogger(__name__)

try:
    import zmq
    _ZMQ_AVAILABLE = True
except ImportError:
    _ZMQ_AVAILABLE = False
    log.error("pyzmq not installed. Run:  pip install pyzmq")

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# SimClient
# ══════════════════════════════════════════════════════════════════════════════

class SimClient:
    """
    Thin ZMQ client that wraps the request/reply protocol shared by
    ThorServer (iTHOR) and ProcThorServer (ProcTHOR).

    Parameters
    ----------
    server_url : str
        ZMQ endpoint.  Both ZMQ format ("tcp://localhost:5555") and HTTP format
        ("http://localhost:5555") are accepted; the latter is auto-converted.
    simulator_type : str
        "thor" | "ithor" — standard iTHOR scenes (default)
        "procthor"       — procedurally generated houses
    timeout_ms : int
        Per-request timeout in milliseconds (default 60 000).
    """

    def __init__(
        self,
        server_url: str = "tcp://localhost:5555",
        simulator_type: str = "thor",
        timeout_ms: int = 60_000,
    ):
        # Normalise HTTP-style URLs
        if server_url.startswith("http://") or server_url.startswith("https://"):
            server_url = server_url.replace("http://", "tcp://").replace("https://", "tcp://")

        self.server_url     = server_url
        self.simulator_type = simulator_type.lower().strip()
        self.timeout_ms     = timeout_ms
        self._connected     = False
        self.last_image:    Optional["Image.Image"] = None

        # Extra ProcTHOR state (populated after reset)
        self.house_index:   Optional[int] = None
        self.current_split: str           = "train"
        self.current_scene: str           = ""

        # Cache connected status to avoid pinging on every Streamlit rerender
        self._connected_cache: bool       = False
        self._connected_cache_time: float = 0.0
        self._CONNECTED_TTL: float        = 2.0   # seconds

        if not _ZMQ_AVAILABLE:
            raise ImportError("pyzmq is required. Install with:  pip install pyzmq")

        self._ctx:    zmq.Context = zmq.Context()
        self._socket: zmq.Socket  = None
        self._connect()

    # ── factory helpers ───────────────────────────────────────────────────────

    @classmethod
    def for_thor(cls, host: str = "localhost", port: int = 5555, **kw) -> "SimClient":
        """Convenience constructor pointing at a ThorServer."""
        return cls(server_url=f"tcp://{host}:{port}", simulator_type="thor", **kw)

    @classmethod
    def for_procthor(cls, host: str = "localhost", port: int = 5556, **kw) -> "SimClient":
        """Convenience constructor pointing at a ProcThorServer."""
        return cls(server_url=f"tcp://{host}:{port}", simulator_type="procthor", **kw)

    # ── connection management ─────────────────────────────────────────────────

    def _connect(self):
        """Close existing socket, recreate context if needed, and open a new socket."""
        # Close old socket
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None

        # If context is dead (e.g. after a crash), recreate it
        try:
            self._ctx.underlying  # raises if context is closed
        except Exception:
            self._ctx = zmq.Context()

        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self.server_url)
        log.info(f"SimClient ({self.simulator_type}) connected to {self.server_url}")

        # Test connection (do NOT go through _send to avoid recursion)
        import json as _json
        try:
            self._socket.send_string(_json.dumps({"cmd": "ping"}))
            raw = self._socket.recv_string()
            self._connected = _json.loads(raw).get("status") == "ok"
        except Exception:
            self._connected = False
        self._connected_cache      = self._connected
        self._connected_cache_time = __import__("time").monotonic()

    def close(self):
        """Explicitly release ZMQ resources (call before replacing the client)."""
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None
        try:
            self._ctx.term()
        except Exception:
            pass

    def __del__(self):
        self.close()

    def reconnect(self):
        """Re-establish ZMQ socket (called by app.py after errors)."""
        self._connected_cache_time = 0.0   # force cache invalidation
        self._connect()

    @property
    def connected(self) -> bool:
        """
        Return True when the server is reachable.
        Caches the result for _CONNECTED_TTL seconds to avoid spamming ZMQ
        pings on every Streamlit rerender (which caused REQ socket deadlocks).
        """
        import time as _time
        now = _time.monotonic()
        if now - self._connected_cache_time < self._CONNECTED_TTL:
            return self._connected_cache
        try:
            resp = self._send({"cmd": "ping"})
            self._connected = resp.get("status") == "ok"
        except Exception:
            self._connected = False
        self._connected_cache      = self._connected
        self._connected_cache_time = now
        return self._connected

    @property
    def is_procthor(self) -> bool:
        """True when this client is configured for ProcTHOR."""
        return self.simulator_type == "procthor"

    # ── low-level send / receive ──────────────────────────────────────────────

    def _send(self, payload: dict) -> dict:
        import json
        try:
            self._socket.send_string(json.dumps(payload))
            raw  = self._socket.recv_string()
            resp = json.loads(raw)
            return resp
        except zmq.Again:
            # Timeout — reset socket so it is not stuck in a bad state
            self._connect()
            raise TimeoutError(
                f"No response from server at {self.server_url} "
                f"(timeout {self.timeout_ms} ms)"
            )
        except Exception as exc:
            self._connect()
            raise exc

    # ── public API ────────────────────────────────────────────────────────────

    def reset(
        self,
        scene_name: str = "",
        simulator_type: str = "",
        # ProcTHOR-specific kwargs (ignored by iTHOR server)
        split: str = "train",
        house_index: Optional[int] = None,
    ) -> dict:
        """
        Load a scene/house and reset episode state.

        iTHOR:
            reset("FloorPlan1")
            reset("FloorPlan201", simulator_type="thor")

        ProcTHOR:
            reset()                          # random house, train split
            reset(split="val")               # random house, val split
            reset(house_index=42)            # specific house, train split
            reset(split="test", house_index=7)

        Returns:
            dict with keys: status, obs, visible_objects, scene,
                            [house_index, split]  (ProcTHOR only)
        """
        sim_type = (simulator_type or self.simulator_type).lower().strip()

        payload: dict = {
            "cmd":            "reset",
            "simulator_type": sim_type,
        }

        if sim_type == "procthor":
            # ProcTHOR uses split + house_index; scene_name is ignored
            payload["split"] = split
            if house_index is not None:
                payload["house_index"] = house_index
            log.info(f"reset (procthor) — split={split}, house_index={house_index}")
        else:
            task = scene_name or "FloorPlan1"
            payload["task"] = task
            log.info(f"reset (thor) — scene={task}")

        try:
            resp = self._send(payload)
            self._connected = resp.get("status") == "ok"

            # Cache ProcTHOR metadata
            if sim_type == "procthor":
                self.house_index   = resp.get("house_index")
                self.current_split = resp.get("split", split)
            self.current_scene = resp.get("scene", scene_name)

            self._refresh_frame()
            return {
                "status":          "ok" if self._connected else "error",
                "obs":             resp.get("obs", ""),
                "visible_objects": resp.get("visible_objects", []),
                "msg":             resp.get("msg", ""),
                "scene":           resp.get("scene", scene_name),
                "house_index":     resp.get("house_index"),
                "split":           resp.get("split", split),
                "sim_type":        sim_type,
            }
        except Exception as exc:
            log.error(f"reset failed: {exc}")
            return {"status": "error", "obs": "", "visible_objects": [], "msg": str(exc)}

    def step(
        self,
        action: str,
        object_name: str = "",
        target_name:  str = "",
    ) -> dict:
        """
        Execute a skill action.

        Returns:
            dict with keys: obs, visible_objects, success, msg, done, reward
        """
        log.info(f"step — action={action}, object={object_name}, target={target_name}")
        try:
            resp = self._send({
                "cmd":    "step",
                "action": action,
                "object": object_name or "",
                "target": target_name or "",
            })
            self._refresh_frame()
            return {
                "obs":             resp.get("obs", ""),
                "visible_objects": resp.get("visible_objects", []),
                "success":         resp.get("success", False),
                "msg":             resp.get("msg", ""),
                "done":            resp.get("done", False),
                "reward":          resp.get("reward", 0.0),
            }
        except Exception as exc:
            log.error(f"step failed: {exc}")
            raise   # let app.py handle it

    def navigate_free(self, action: str) -> dict:
        """
        Execute a free navigation action (MoveAhead, MoveBack, RotateLeft,
        RotateRight, LookUp, LookDown) without consuming a plan step.

        Returns:
            dict with keys: status, obs, visible_objects, visible_objects_meta
        """
        log.info(f"navigate_free — action={action}")
        try:
            resp = self._send({"cmd": "nav", "action": action})
            self._refresh_frame()
            return {
                "status":               resp.get("status", "error"),
                "obs":                  resp.get("obs", ""),
                "visible_objects":      resp.get("visible_objects", []),
                "visible_objects_meta": resp.get("visible_objects_meta", []),
                "msg":                  resp.get("msg", ""),
            }
        except Exception as exc:
            log.error(f"navigate_free failed: {exc}")
            return {"status": "error", "obs": "", "visible_objects": [], "msg": str(exc)}

    def next_house(self, split: str = "") -> dict:
        """
        ProcTHOR only: load a new random house from the current (or given) split.
        No-op on iTHOR (returns an error dict without raising).

        Returns:
            dict with keys: status, obs, visible_objects, house_index, split
        """
        if not self.is_procthor:
            return {"status": "error", "msg": "next_house is only supported by ProcThorServer"}

        payload: dict = {"cmd": "next_house"}
        if split:
            payload["split"] = split

        log.info(f"next_house — split={split or self.current_split}")
        try:
            resp = self._send(payload)
            self.house_index   = resp.get("house_index")
            self.current_split = resp.get("split", self.current_split)
            self._refresh_frame()
            return {
                "status":          resp.get("status", "error"),
                "obs":             resp.get("obs", ""),
                "visible_objects": resp.get("visible_objects", []),
                "house_index":     resp.get("house_index"),
                "split":           resp.get("split"),
                "msg":             resp.get("msg", ""),
            }
        except Exception as exc:
            log.error(f"next_house failed: {exc}")
            return {"status": "error", "msg": str(exc)}

    def set_house(self, index: int, split: str = "") -> dict:
        """
        ProcTHOR only: load a specific house by index.

        Returns:
            dict with keys: status, obs, visible_objects, house_index, split
        """
        if not self.is_procthor:
            return {"status": "error", "msg": "set_house is only supported by ProcThorServer"}

        payload: dict = {
            "cmd":   "set_house",
            "index": index,
        }
        if split:
            payload["split"] = split

        log.info(f"set_house — index={index}, split={split or self.current_split}")
        try:
            resp = self._send(payload)
            self.house_index   = resp.get("house_index")
            self.current_split = resp.get("split", self.current_split)
            self._refresh_frame()
            return {
                "status":          resp.get("status", "error"),
                "obs":             resp.get("obs", ""),
                "visible_objects": resp.get("visible_objects", []),
                "house_index":     resp.get("house_index"),
                "split":           resp.get("split"),
                "msg":             resp.get("msg", ""),
            }
        except Exception as exc:
            log.error(f"set_house failed: {exc}")
            return {"status": "error", "msg": str(exc)}

    def get_frame(self) -> Optional["Image.Image"]:
        """Return the most recent RGB frame as a PIL Image (may be None)."""
        self._refresh_frame()
        return self.last_image

    def get_objects(self) -> dict:
        """Return full object metadata from the server."""
        try:
            return self._send({"cmd": "get_objects"})
        except Exception as exc:
            log.error(f"get_objects failed: {exc}")
            return {"status": "error", "msg": str(exc), "objects": []}

    def get_state(self) -> dict:
        """Return current agent state (position, rotation, held objects)."""
        try:
            return self._send({"cmd": "get_state"})
        except Exception as exc:
            log.error(f"get_state failed: {exc}")
            return {"status": "error", "msg": str(exc)}

    def nav(self, action: str) -> dict:
        """
        Send a low-level navigation action (MoveAhead, RotateLeft, …).

        Valid actions: MoveAhead, MoveBack, MoveLeft, MoveRight,
                       RotateLeft, RotateRight, LookUp, LookDown
        """
        try:
            resp = self._send({"cmd": "nav", "action": action})
            self._refresh_frame()
            return resp
        except Exception as exc:
            log.error(f"nav failed: {exc}")
            return {"status": "error", "msg": str(exc)}

    def stop(self):
        """Close the ZMQ socket and context."""
        log.info("SimClient stopping.")
        try:
            if self._socket:
                self._socket.close()
            self._ctx.term()
        except Exception:
            pass
        self._connected = False

    # ── internal helpers ──────────────────────────────────────────────────────

    def _refresh_frame(self):
        """Fire a get_frame request and cache the result in self.last_image."""
        if not _PIL_AVAILABLE:
            return
        try:
            resp = self._send({"cmd": "get_frame"})
            if resp.get("status") == "ok" and resp.get("frame"):
                img_bytes       = base64.b64decode(resp["frame"])
                self.last_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as exc:
            log.debug(f"get_frame failed (non-fatal): {exc}")

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        extra = ""
        if self.is_procthor and self.house_index is not None:
            extra = f", house={self.current_split}:{self.house_index}"
        elif self.current_scene:
            extra = f", scene={self.current_scene}"
        return f"SimClient({self.simulator_type}@{self.server_url} [{status}{extra}])"


# ── Backward-compat shim ────────────────────────────────────────────────────
# The evaluate harness (evaluate_sim.py, make_dataset_from_sim.py,
# record_reference.py) predates the SimClient rename and still constructs
# ``ThorClient(host=..., port=...)``. SimClient's __init__ now takes
# ``server_url`` + ``simulator_type``; the old host/port form maps onto the
# ``for_thor`` convention. This thin subclass keeps those scripts working
# without touching them.
class ThorClient(SimClient):
    """Compat alias for the pre-rename iTHOR client (host/port constructor)."""

    def __init__(self, host: str = "localhost", port: int = 5555, **kw):
        super().__init__(server_url=f"tcp://{host}:{port}",
                         simulator_type="thor", **kw)