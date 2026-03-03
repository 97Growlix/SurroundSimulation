from pathlib import Path
import json

Params = [("ConeSideThickness", 2), ("MiddleThickness", 1)]

TriggerPath = Path(__file__).parent / "run.trigger"

data = {
    "parameters": dict(Params)
}

TriggerPath.write_text(json.dumps(data, indent=4))



