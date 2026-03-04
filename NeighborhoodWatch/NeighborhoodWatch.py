import adsk.core
import threading
import os
import traceback
import importlib.util
import time


ScriptName = "Mutate.py"
TrigFile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.trigger")
TargScript = os.path.join(os.path.dirname(os.path.abspath(__file__)), ScriptName)

Handlers = []

PollInterval = 2

BusyFlag = False

####
_app = None
_ui = None
_timer_thread = None
_stop_event = threading.Event()


def poll_loop():
    log("Polling thread started")
    log(TrigFile)
    while True:
        try:
            if _stop_event.is_set():
                log("Stop event detected — exiting poll loop")
                break

            #log("Polling iteration")

            if not BusyFlag and os.path.exists(TrigFile):
                log("Trigger file detected")

                # try:
                #     os.remove(TrigFile)
                # except Exception as e:
                #     log(f"Couldn't delete trigger: {e}")

                _app.fireCustomEvent(CustEventID, "")

        except Exception:
            log("Poll loop crash:\n" + traceback.format_exc())

        #log("Sleeping...")
        _stop_event.wait(PollInterval)
    
CustEventID = "NeighborhoodWatch_RunScript"
_event_handler = None

class ScriptRunner(adsk.core.CustomEventHandler):
    global BusyFlag

    def notify(self, args):
        try:
            BusyFlag = True
            RunTargScript()
        except Exception:
            log(f"ERROR running target script:\n{traceback.format_exc()}")
        finally:
            BusyFlag=False
def RunTargScript():
    if not os.path.exists(TargScript):
        log(f"Error, targ script missing: {TargScript}")
        return
    log(f"Ruinning {TargScript}")


    spec= importlib.util.spec_from_file_location("Target_Script", TargScript)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    log("Target script finished")

def run(context):

    global _app, _ui, _timer_thread, _event_handler

    try:
        _app = adsk.core.Application.get()
        log(f"Add-in dir: {os.path.dirname(os.path.abspath(__file__))}")
        log(f"Looking for script at: {TargScript}")
        log(f"File exists: {os.path.exists(TargScript)}")
        _ui = _app.userInterface
        
        _event_handler = ScriptRunner()
        Handlers.append(_event_handler)
        
        log("Registering custom event...")
        event = _app.registerCustomEvent(CustEventID)   
        event.add(_event_handler)
        log("We're a-watchin'")

        _stop_event.clear()
        _timer_thread = threading.Thread(target=poll_loop, daemon=True)
        _timer_thread.start()
    except Exception:
        log("RUN CRASH:\n" + traceback.format_exc())

def stop(context):
    log("STOP() CALLED")

    global _event_handler

    _stop_event.set()

    if _app:
        event = _app.customEvents.itemById(CustEventID)
        if event and _event_handler:
            event.remove(_event_handler)

    if _app:
        _app.unregisterCustomEvent(CustEventID)
    log("Watch add-in stopped")


####
def log(msg):
    try: 
        adsk.core.Application.get().log(str(msg))
    except Exception:
        pass
