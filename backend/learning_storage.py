import os
import json

class LearningStorage:
    def __init__(self, filepath="data/traffic_learning.json"):
        self.filepath = filepath

        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

        if not os.path.exists(self.filepath):
            with open(self.filepath, "w") as f:
                json.dump({}, f)

    def load(self):
        try:
            with open(self.filepath, "r") as f:
                content = f.read().strip()

                # If file is empty → return empty dict
                if not content:
                    return {}

                return json.loads(content)

        except json.JSONDecodeError:
            # If corrupted → reset safely
            return {}

    def save(self, data):
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=4)