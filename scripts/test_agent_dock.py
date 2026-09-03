"""Headless test of the DeepSeek agent dock (encryption, integration, chat)."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# 1. encrypted config round-trip (DPAPI)
from PRSTCore.visualization.agent_dock import (
    AgentConfig, AgentDock, _protect_b64, _unprotect_b64)

enc = _protect_b64("sk-test-secret-123")
dec = _unprotect_b64(enc)
print("dpapi round-trip:", dec == "sk-test-secret-123",
      "| stored != plaintext:", "sk-test-secret-123" not in enc)

cfg_path = os.path.join(tempfile.gettempdir(), "agent_cfg_test.json")
if os.path.exists(cfg_path):
    os.remove(cfg_path)
cfg = AgentConfig(cfg_path)
cfg.base_url = "https://api.deepseek.com"
cfg.model = "deepseek-v4-flash"
cfg.api_key = "sk-my-secret-key"
cfg.save()
raw = open(cfg_path, encoding="utf-8").read()
print("file contains plaintext key:", "sk-my-secret-key" in raw,
      "| api_key_enc present:", "api_key_enc" in raw)
cfg2 = AgentConfig(cfg_path)
print("reload key ok:", cfg2.api_key == "sk-my-secret-key",
      "| model:", cfg2.model, "| base_url:", cfg2.base_url)
os.remove(cfg_path)

# 2. window integration
from PySide6 import QtCore, QtWidgets  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

from PRSTCore.visualization.simulator_gui import SimulatorWindow  # noqa: E402

win = SimulatorWindow()
menus = [a.text().replace("&", "") for a in win.menuBar().actions()]
print("menus:", menus)
print("agent dock exists:", win.agent_dock is not None,
      "| hidden initially:", not win.agent_dock.isVisible())

# Configure the key FIRST so _agent_launch does not open the modal dialog.
win.agent_dock.config.base_url = "http://127.0.0.1:9"  # fast connection error
win.agent_dock.config.model = "deepseek-v4-flash"
win.agent_dock.config.api_key = "sk-fake"
win.show()
win._agent_launch()
print("after launch visible:", win.agent_dock.isVisible())

# 3. send -> graceful error
win.agent_dock.prompt_edit.setPlainText("你好")
win.agent_dock._send()


def wait_until(cond, t=60000):
    loop = QtCore.QEventLoop()
    timer = QtCore.QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(t)
    while not cond() and timer.isActive():
        app.processEvents()
    timer.stop()
    return cond()


ok = wait_until(lambda: not win.agent_dock._busy)
ans = win.agent_dock.answer_edit.toPlainText()
print("send done:", ok, "| error shown:", "Error" in ans,
      "| busy cleared:", not win.agent_dock._busy)
win.agent_dock.collapse()
print("collapsed visible:", not win.agent_dock.isVisible())
print("AGENT TEST", "PASSED" if (menus == ["Project", "View", "Agent",
                                           "Help"] and ok
                                 and "Error" in ans
                                 and not win.agent_dock._busy)
      else "FAILED")
