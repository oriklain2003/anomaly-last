from fr24sdk.client import Client
import inspect

try:
    client = Client(api_token="019aca50-8288-7260-94b5-6d82fbeb351c|dC21vuw2bsf2Y43qAlrBKb7iSM9ibqSDT50x3giN763b577b")
    
    print("\nSignature of flight_events.get_light:")
    print(inspect.signature(client.historic.flight_events.get_light))

except Exception as e:
    print(e)
