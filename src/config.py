AIRLINES = [
    "Jet Airways",
    "IndiGo",
    "Air India",
    "Multiple carriers",
    "SpiceJet",
    "Vistara",
    "GoAir",
    "Air Asia",
    "Multiple carriers Premium economy",
    "Jet Airways Business",
    "Trujet",
    "Vistara Premium economy",
]

SOURCES = ["Delhi", "Kolkata", "Mumbai", "Chennai", "Banglore"]

DESTINATIONS = ["Cochin", "Delhi", "New Delhi", "Hyderabad", "Kolkata", "Banglore"]

TOTAL_STOPS_OPTIONS = ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"]

TOTAL_STOPS_MAP = {
    "non-stop": 0,
    "1 stop": 1,
    "2 stops": 2,
    "3 stops": 3,
    "4 stops": 4,
}


AIRLINE_TO_CODE = {name: i for i, name in enumerate(AIRLINES)}
DEST_TO_CODE = {name: i for i, name in enumerate(DESTINATIONS)}
