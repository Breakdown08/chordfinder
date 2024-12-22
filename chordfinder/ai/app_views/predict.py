# from librosa import resample
from django.http import JsonResponse
# from ..neural_network.model import predict_chord_from_audio
import json
# import numpy as np


def predict(request):
    label = ""
    chance = ""
    # if request.method != 'GET':
    #     return JsonResponse({'error': 'Only GET requests are allowed'}, status=405)
    # try:
    #     data = json.loads(request.body.decode('utf-8'))
    #     audio = np.array(data["input"])
    #     resampled = resample(y=audio, orig_sr=44100, target_sr=16000)
    # except json.JSONDecodeError:
    #     return JsonResponse({'error': 'Missing required parameter: input'}, status=400)
    # label, chance = predict_chord_from_audio(resampled)
    return JsonResponse({'label': label, 'chance': str(chance)})
