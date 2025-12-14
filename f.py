from datetime import datetime

import pandas as pd
def get():
    from fr24sdk.client import Client
    client = Client(api_token="019a9d54-678a-73da-a927-1f7a60b27a8f|xouEuixtJFLIlwNKajdW0mFjV4sguJsf5eByvIYTde7f4b26")
    tracks = client.flight_tracks.get(flight_id=['3a9c0cbd'])
    dt = datetime.strptime("17-08 02:06:14", "%d-%m %H:%M:%S").replace(year=2025)
    dt2 = datetime.strptime("17-08 23:50:14", "%d-%m %H:%M:%S").replace(year=2025)
    tracks = client.flight_summary.get_full(callsigns=["ISR562"],flight_datetime_from=dt, flight_datetime_to=dt2)
    print()
    pd.DataFrame(tracks.model_dump()["data"][0]["tracks"]).to_csv("test.csv")

if __name__ == '__main__':
    get()
    s = pd.read_csv("test.csv")
    print()