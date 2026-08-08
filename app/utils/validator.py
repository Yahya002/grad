REQUIRED_FIELDS = ["from", "to", "name", "phone"]

def validate(data):
    missing = [f for f in REQUIRED_FIELDS if not data.get(f)]
    return missing