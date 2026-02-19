import adsk.core
import threading
import os
import traceback
import importlib.util


ScriptName = "Runthisone.py"
TrigFile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.trigger")
TargScript = os.path.join(os.path.dirname(os.path.abspath(__file__)), ScriptName)



PollInterval = 2

####
_app = None
_ui = None
_timer_thread = None
_stop_event = threading.Event()



def poll_loop():
    log(f"{TargScript}")
    log(f"{TrigFile}")  
    while not _stop_event.is_set():
        try: 
            if os.path.exists(TrigFile):
                log('Gasp, a trigger file')

                try:
                    os.remove(TrigFile)
                except Exception as e:
                    log(f"Warning: trigger file too strong. Couldn't delete. {e}")

                adsk.core.Application.get().fireCustomEvent(CustEventID, "")

        except Exception:
            log(f"Error in poll loop: \n {traceback.format_exc()}")

        log("another iteration later")

        _stop_event.wait(PollInterval)
    
CustEventID = "NeighborhoodWatch_RunScript"
_event_handler = None

class ScriptRunner(adsk.core.CustomEventHandler):

    def notify(self, args):
        try:
            RunTargScript()
        except Exception:
            log(f"ERROR running target script:\n{traceback.format_exc()}")

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
    global _app, _ui, _timer_thread

    _app = adsk.core.Application.get()
    _ui = _app.userInterface

    _event_handler = ScriptRunner()
    _app.registerCustomEvent(CustEventID)
    _app.onCustomEvent.add(_event_handler)

    log("We're a-watchin'")

    _stop_event.clear()
    _timer_thread = threading.Thread(target=poll_loop, daemon=True)
    _timer_thread.start()

def stop(context):
    _stop_event.set()
    if _app:
        _app.unregisterCustomEvent(CustEventID)
    log("Watch add-in stopped")


####
def log(msg):
    try: 
        adsk.core.Application.get().log(str(msg))
    except Exception:
        pass
