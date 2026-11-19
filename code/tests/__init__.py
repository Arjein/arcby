import os

# Tests never read the private household (private/household.json, ~/.config/arcby/household.json): they run on the
# example names, the same on every machine. Set before any test module imports robot.config.
os.environ["ARCBY_NO_PRIVATE"] = "1"
