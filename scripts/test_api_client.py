import urllib.request
import json

with open('data/smoke/smoke_human_like.wav', 'rb') as f:
    audio_bytes = f.read()

boundary = '----SonicShieldBoundaryTest'
parts = [
    f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="smoke_human_like.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode('utf-8'),
    audio_bytes,
    f'\r\n--{boundary}--\r\n'.encode('utf-8'),
]
body = b"".join(parts)

req = urllib.request.Request(
    'http://localhost:8000/analyze?codec=clean&mode=both&detection_mode=pretrained',
    data=body,
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
)
try:
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        print("FastAPI /analyze overall_liveness:", res.get("overall_liveness"))
        print("FastAPI /analyze aasist class:", res.get("overall_aasist_class"))
        print("FastAPI /analyze aasist score:", res.get("overall_aasist_bona_fide_score"))
        print("FastAPI /analyze duration:", res.get("duration_sec"))
        print("FastAPI /analyze status: SUCCESS")
except Exception as e:
    print("FastAPI request failed:", e)

