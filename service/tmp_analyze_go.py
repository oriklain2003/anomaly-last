import sqlite3, json, math
from core.models import TrackPoint, FlightTrack, RuleContext
from core.config import TRAIN_NORTH, TRAIN_SOUTH, TRAIN_EAST, TRAIN_WEST
from rules.rule_logic import evaluate_rule, AIRPORT_BY_CODE, GO_AROUND_RADIUS_NM

DB='flight_cache.db'
FID='3b5d8640'
conn=sqlite3.connect(DB)
row=conn.cursor().execute("SELECT data FROM flights WHERE flight_id=?", (FID,)).fetchone()
if not row:
    raise SystemExit('flight not found')
data=json.loads(row[0])
raw_points=data['points']
points=[TrackPoint(
    flight_id=p['flight_id'],
    timestamp=int(p['timestamp']),
    lat=float(p['lat']),
    lon=float(p['lon']),
    alt=float(p['alt']),
    gspeed=float(p['gspeed']) if p.get('gspeed') is not None else None,
    vspeed=float(p['vspeed']) if p.get('vspeed') is not None else None,
    track=float(p['track']) if p.get('track') is not None else None,
    squawk=str(p['squawk']) if p.get('squawk') is not None else None,
    callsign=p.get('callsign'),
    source=p.get('source'),
) for p in raw_points]

filtered=[p for p in points if TRAIN_SOUTH <= p.lat <= TRAIN_NORTH and TRAIN_WEST <= p.lon <= TRAIN_EAST]
print('total points', len(points), 'filtered', len(filtered))
if filtered:
    print('ts range', filtered[0].timestamp, filtered[-1].timestamp)
track=FlightTrack(flight_id=FID, points=filtered)
ctx=RuleContext(track=track, metadata=None, repository=None)
res=evaluate_rule(ctx, 6)
print('go-around matched?', res.matched)
print('summary', res.summary)
print('details', res.details)

# manual stats near Damascus (OSDI)
osdi=AIRPORT_BY_CODE['OSDI']

def haversine_nm(lat1, lon1, lat2, lon2):
    R = 3440.065
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2-lat1)
    dlambda = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

near=[p for p in filtered if haversine_nm(p.lat,p.lon, osdi.lat, osdi.lon) <= GO_AROUND_RADIUS_NM]
print('near OSDI points', len(near))
if near:
    lowest=min(near, key=lambda p:p.alt)
    print('lowest alt', lowest.alt, 'track', lowest.track, 'ts', lowest.timestamp)
    agl=lowest.alt - (osdi.elevation_ft or 0)
    print('lowest AGL', agl)
    after=[p for p in near if p.timestamp>lowest.timestamp]
    before=[p for p in near if p.timestamp<lowest.timestamp]
    if after:
        print('climb after', max(p.alt for p in after)-lowest.alt)
    if before:
        print('descent before', min(p.alt for p in before)-lowest.alt)
    print('first near', near[0].timestamp, 'last near', near[-1].timestamp)