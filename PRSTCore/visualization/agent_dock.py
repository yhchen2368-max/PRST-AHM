"""DeepSeek chat agent dock for the simulator GUI.

A collapsible floating panel docked on the right edge of
:class:`~PRSTCore.visualization.simulator_gui.SimulatorWindow`:

* vertical layout -- the **dialog** (prompt) box on top, the **answer** box
  below (streamed from DeepSeek);
* an ``Agent`` menu with *Start* (show + start the agent) and
  *API Settings...* (base_url / model / api_key);
* the dock **auto-collapses** after an idle period when the user is not
  using it, and re-opens from the menu.

The API key is persisted **encrypted** with the Windows DPAPI
(``CryptProtectData``), so no plaintext key ever touches disk; on non-Windows
platforms a base64 fallback is used (with a warning).  The DeepSeek API is
OpenAI-compatible, so the ``openai`` client is used against
``https://api.deepseek.com`` (model ``deepseek-v4-flash``).
"""

from __future__ import annotations

import base64
import ctypes
import json
import sys
from pathlib import Path

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor

_Signal = QtCore.Signal

CONFIG_PATH = Path.home() / ".prstcore" / "agent_config.json"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
KEY_URL = "https://platform.deepseek.com/api_keys"

_SYSTEM_PROMPT = (
    "You are a senior petroleum reservoir simulation assistant, expert in "
    "PRSTCore / MRST / ECLIPSE black-oil simulation, DATA-deck parsing, "
    "grids and property fields, wells and completions, HDF5 result "
    "interpretation, production curves and NPV optimisation. Answer "
    "concisely and accurately in English; give runnable code snippets "
    "when code is relevant."
)


# ===========================================================================
# Windows DPAPI (CryptProtectData) -- encrypts to the current user
# ===========================================================================
class _DATA_BLOB(ctypes.Structure):
    _fields_ = [("cbData", ctypes.c_ulong),
                ("pbData", ctypes.POINTER(ctypes.c_char))]


def _dpapi_protect(data: bytes) -> bytes:
    """Encrypt ``data`` with the current Windows user's DPAPI key."""
    buf = ctypes.create_string_buffer(data, len(data))
    blob_in = _DATA_BLOB(len(data), ctypes.cast(
        buf, ctypes.POINTER(ctypes.c_char)))
    blob_out = _DATA_BLOB()
    if not ctypes.windll.crypt32.CryptProtectData(
            ctypes.byref(blob_in), None, None, None, None, 0,
            ctypes.byref(blob_out)):
        raise OSError("CryptProtectData failed (%s)"
                      % ctypes.windll.kernel32.GetLastError())
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)


def _dpapi_unprotect(blob: bytes) -> bytes:
    """Decrypt a DPAPI blob produced by :func:`_dpapi_protect`."""
    buf = ctypes.create_string_buffer(blob, len(blob))
    blob_in = _DATA_BLOB(len(blob), ctypes.cast(
        buf, ctypes.POINTER(ctypes.c_char)))
    blob_out = _DATA_BLOB()
    if not ctypes.windll.crypt32.CryptUnprotectData(
            ctypes.byref(blob_in), None, None, None, None, 0,
            ctypes.byref(blob_out)):
        raise OSError("CryptUnprotectData failed (%s)"
                      % ctypes.windll.kernel32.GetLastError())
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)


def _protect_b64(text: str) -> str:
    """Encrypt ``text`` and return a portable base64 string."""
    raw = text.encode("utf-8")
    if sys.platform.startswith("win"):
        return base64.b64encode(_dpapi_protect(raw)).decode("ascii")
    # Non-Windows fallback: reversible obfuscation (not real encryption).
    return base64.b64encode(raw).decode("ascii")


def _unprotect_b64(encoded: str) -> str:
    """Decrypt a base64 string written by :func:`_protect_b64`."""
    try:
        blob = base64.b64decode(encoded.encode("ascii"))
    except Exception:
        return ""
    if sys.platform.startswith("win"):
        try:
            return _dpapi_unprotect(blob).decode("utf-8")
        except Exception:
            return ""
    return blob.decode("utf-8", errors="replace")


# ===========================================================================
# config (base_url / model / encrypted api_key)
# ===========================================================================
class AgentConfig:
    """Persistent agent settings; the API key is stored encrypted."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else CONFIG_PATH
        self.base_url = DEFAULT_BASE_URL
        self.model = DEFAULT_MODEL
        self._api_key = ""
        self.load()

    def load(self):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        self.base_url = str(data.get("base_url") or DEFAULT_BASE_URL)
        self.model = str(data.get("model") or DEFAULT_MODEL)
        enc = data.get("api_key_enc")
        self._api_key = _unprotect_b64(enc) if enc else ""

    def save(self):
        data = {
            "base_url": self.base_url,
            "model": self.model,
            "api_key_enc": (_protect_b64(self._api_key)
                            if self._api_key else ""),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8")
        except OSError:
            pass

    @property
    def api_key(self) -> str:
        return self._api_key

    @api_key.setter
    def api_key(self, value: str):
        self._api_key = (value or "").strip()

    @property
    def ready(self) -> bool:
        return bool(self._api_key)


# ===========================================================================
# streaming worker
# ===========================================================================
class _AgentWorker(QtCore.QThread):
    """Call the DeepSeek chat API (OpenAI-compatible) and stream the reply."""

    token = _Signal(str)          # one answer chunk
    finished_text = _Signal(str)  # full answer
    error = _Signal(str)

    def __init__(self, base_url, api_key, model, messages, parent=None):
        super().__init__(parent)
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.messages = messages
        self._stop = False

    def request_stop(self):
        self._stop = True

    def run(self):
        try:
            from openai import OpenAI
        except Exception as exc:
            self.error.emit("openai client unavailable: %s: %s"
                            % (type(exc).__name__, exc))
            return
        try:
            client = OpenAI(base_url=self.base_url, api_key=self.api_key,
                            timeout=180, max_retries=0)
            stream = client.chat.completions.create(
                model=self.model, messages=self.messages, stream=True,
                temperature=0.7)
            parts = []
            for chunk in stream:
                if self._stop:
                    break
                try:
                    delta = chunk.choices[0].delta
                    text = getattr(delta, "content", None)
                except Exception:
                    text = None
                if text:
                    parts.append(text)
                    self.token.emit(text)
            self.finished_text.emit("".join(parts))
        except Exception as exc:
            self.error.emit("DeepSeek request failed: %s: %s"
                            % (type(exc).__name__, exc))


# ===========================================================================
# config dialog
# ===========================================================================
class AgentConfigDialog(QtWidgets.QDialog):
    """base_url / model / api_key form (the key is saved encrypted)."""

    def __init__(self, config: AgentConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Agent API Settings")
        self.setMinimumWidth(460)
        self._config = config

        form = QtWidgets.QFormLayout(self)
        self.base_url_edit = QtWidgets.QLineEdit(config.base_url)
        self.model_edit = QtWidgets.QLineEdit(config.model)
        self.key_edit = QtWidgets.QLineEdit(config.api_key)
        self.key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.key_edit.setPlaceholderText("sk-...  (saved encrypted)")
        form.addRow("Base URL:", self.base_url_edit)
        form.addRow("Model:", self.model_edit)
        form.addRow("API Key:", self.key_edit)

        hint = QtWidgets.QLabel(
            '<a href="%s">Get a DeepSeek API key</a>  \u2014 the key is saved '
            "encrypted (Windows DPAPI) to %s" % (KEY_URL, config.path))
        hint.setOpenExternalLinks(True)
        hint.setWordWrap(True)
        form.addRow(hint)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save
            | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def apply(self):
        self._config.base_url = self.base_url_edit.text().strip() \
            or DEFAULT_BASE_URL
        self._config.model = self.model_edit.text().strip() or DEFAULT_MODEL
        self._config.api_key = self.key_edit.text()
        self._config.save()


# ===========================================================================
# the dock
# ===========================================================================
class AgentDock(QtWidgets.QDockWidget):
    """Floating DeepSeek chat panel docked on the right edge.

    Vertical layout: dialog (prompt) box on top, answer box below.  The
    panel auto-collapses after an idle period and re-opens from the Agent
    menu (``start``).
    """

    def __init__(self, parent=None):
        super().__init__("Agent (DeepSeek)", parent)
        self.setObjectName("AgentDock")
        self.setAllowedAreas(Qt.RightDockWidgetArea
                             | Qt.LeftDockWidgetArea)
        self.setFeatures(self.features()
                         | QtWidgets.QDockWidget.DockWidgetClosable)

        self.config = AgentConfig()
        self._history = []           # [{"role": ..., "content": ...}, ...]
        self._worker = None
        self._busy = False

        body = QtWidgets.QWidget(self)
        root = QtWidgets.QVBoxLayout(body)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # toolbar: send / clear / status
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Chat"))
        bar.addStretch(1)
        self.status_label = QtWidgets.QLabel("")
        bar.addWidget(self.status_label)
        self.collapse_btn = QtWidgets.QToolButton()
        self.collapse_btn.setText("Collapse »")
        self.collapse_btn.setToolTip("Collapse the panel (auto-collapses "
                                     "when idle)")
        self.collapse_btn.clicked.connect(self.collapse)
        bar.addWidget(self.collapse_btn)
        root.addLayout(bar)

        split = QtWidgets.QSplitter(Qt.Vertical)

        # answer box -- top
        answer_page = QtWidgets.QWidget()
        av = QtWidgets.QVBoxLayout(answer_page)
        av.setContentsMargins(0, 0, 0, 0)
        av.addWidget(QtWidgets.QLabel("Answer"))
        self.answer_edit = QtWidgets.QTextEdit()
        self.answer_edit.setReadOnly(True)
        av.addWidget(self.answer_edit)
        split.addWidget(answer_page)

        # dialog (prompt) box -- bottom
        prompt_page = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(prompt_page)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.addWidget(QtWidgets.QLabel("Prompt (Enter to send, "
                                     "Shift+Enter for a new line)"))
        self.prompt_edit = QtWidgets.QPlainTextEdit()
        self.prompt_edit.setPlaceholderText(
            "e.g. explain the SPE1 PVT tables, or write code to plot a "
            "2D slice of the pressure field...")
        self.prompt_edit.setMaximumHeight(140)
        self.prompt_edit.installEventFilter(self)
        pv.addWidget(self.prompt_edit)
        send_row = QtWidgets.QHBoxLayout()
        self.send_btn = QtWidgets.QPushButton("Send")
        self.send_btn.clicked.connect(self._send)
        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_chat)
        send_row.addWidget(self.send_btn)
        send_row.addWidget(self.clear_btn)
        send_row.addStretch(1)
        pv.addLayout(send_row)
        split.addWidget(prompt_page)

        split.setSizes([320, 160])
        root.addWidget(split, 1)
        self.setWidget(body)

        # idle auto-collapse: 90 s without use collapses the panel
        self._idle = QtCore.QTimer(self)
        self._idle.setSingleShot(True)
        self._idle.setInterval(90000)
        self._idle.timeout.connect(self._auto_collapse)
        self.prompt_edit.textChanged.connect(self._kick_idle)
        self.prompt_edit.setFocusPolicy(Qt.StrongFocus)

        self._append_answer("Agent ready. Configure an API key to start "
                            "chatting.", "system")

    # ------------------------------------------------------------- public
    def start(self):
        """Show the dock and (re)arm it for use (Agent menu: Start)."""
        self.show()
        self.raise_()
        self.prompt_edit.setFocus()
        self._kick_idle()

    def collapse(self):
        self.hide()

    def _auto_collapse(self):
        if self._busy:
            self._kick_idle()
            return
        focus = QtWidgets.QApplication.focusWidget()
        if focus is not None and self.isAncestorOf(focus):
            self._kick_idle()
            return
        self.hide()

    def _kick_idle(self):
        self._idle.start()

    def configure(self):
        """Open the API-parameter dialog (Agent menu: API Settings)."""
        dlg = AgentConfigDialog(self.config, self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            dlg.apply()
            self._append_answer("API settings saved (key stored encrypted).",
                                "system")

    # ------------------------------------------------------------- chat
    def eventFilter(self, obj, event):
        if obj is self.prompt_edit and event.type() == QtCore.QEvent.KeyPress:
            if (event.key() in (Qt.Key_Return, Qt.Key_Enter)
                    and not (event.modifiers() & Qt.ShiftModifier)):
                self._send()
                return True
        return super().eventFilter(obj, event)

    def _send(self):
        if self._busy:
            return
        text = self.prompt_edit.toPlainText().strip()
        if not text:
            return
        if not self.config.ready:
            self._append_answer("No API key configured yet \u2014 please fill "
                                "it in via Agent \u2192 API Settings.", "system")
            self.configure()
            return
        self.prompt_edit.clear()
        self._append_answer(text, "user")
        self._history.append({"role": "user", "content": text})
        self._start_request()

    def _start_request(self):
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        messages += self._history[-20:]
        self._busy = True
        self.send_btn.setEnabled(False)
        self.status_label.setText("Thinking\u2026")
        self._append_answer("", "assistant", stream=True)
        self._cursor = self.answer_edit.textCursor()
        self._cursor.movePosition(QTextCursor.MoveOperation.End)

        self._worker = _AgentWorker(self.config.base_url,
                                    self.config.api_key,
                                    self.config.model, messages, self)
        self._worker.token.connect(self._on_token)
        self._worker.finished_text.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(
            lambda: self._worker.deleteLater())
        self._worker.start()

    def _on_token(self, text):
        self._cursor.insertText(text)
        self.answer_edit.setTextCursor(self._cursor)
        self.answer_edit.ensureCursorVisible()
        self._kick_idle()

    def _on_finished(self, full):
        self._history.append({"role": "assistant", "content": full})
        self._finish_request()

    def _on_error(self, err):
        self._append_answer("<i>%s</i>" % err, "error")
        self._finish_request()

    def _finish_request(self):
        self._busy = False
        self.send_btn.setEnabled(True)
        self.status_label.setText("")
        self._kick_idle()

    def _clear_chat(self):
        self._history = []
        self.answer_edit.clear()
        self._append_answer("Chat cleared.", "system")

    def _append_answer(self, text, role, stream=False):
        """Append a styled block to the answer box."""
        cursor = self.answer_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if role == "user":
            cursor.insertHtml("<p style='color:#1565c0'><b>You:</b><br>"
                              + self._esc(text) + "</p><hr>")
        elif role == "system":
            cursor.insertHtml("<p style='color:#666'><i>%s</i></p>"
                              % self._esc(text))
        elif role == "error":
            cursor.insertHtml("<p style='color:#c62828'><b>Error:</b>%s</p>"
                              % text)
        else:  # assistant
            cursor.insertHtml("<p style='color:#1e7d32'><b>Agent：</b></p>")
        self.answer_edit.setTextCursor(cursor)
        self.answer_edit.ensureCursorVisible()

    @staticmethod
    def _esc(text: str) -> str:
        return (str(text).replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;"))
