import sys
with open('mobilenet_detector.py', 'r') as f:
    text = f.read()

text = text.replace("_, _, x = signal.spectrogram(", "res = signal.spectrogram(")
text = text.replace("mode='magnitude',\n        )", "mode='magnitude',\n        )\n        x = res[-1]")
with open('mobilenet_detector.py', 'w') as f:
    f.write(text)
