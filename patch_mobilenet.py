import sys

with open('mobilenet_detector.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    new_lines.append(line)
    if "def predict_file(self, file_path: str) -> dict:" in line:
        insert_code = """
    def predict_full_audio(self, audio: np.ndarray, sr: int, chunk_sec: float = 1.0, step_sec: float = 0.5):
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)
        
        if sr != sample_rate:
            common = gcd(sample_rate, sr)
            audio = resample_poly(audio, sample_rate // common, sr // common)
        
        chunk_len = int(chunk_sec * sample_rate)
        step_len = int(step_sec * sample_rate)
        
        if len(audio) < chunk_len:
            n = math.ceil(chunk_len / len(audio))
            padded_audio = np.tile(audio, n)[:chunk_len]
            res = self.predict_audio_array(padded_audio, sample_rate)
            res['start_time'] = 0.0
            res['end_time'] = len(audio) / sample_rate
            return [res]
        
        results = []
        for start_idx in range(0, len(audio) - chunk_len + 1, step_len):
            chunk = audio[start_idx:start_idx + chunk_len]
            res = self.predict_audio_array(chunk, sample_rate)
            res['start_time'] = start_idx / sample_rate
            res['end_time'] = (start_idx + chunk_len) / sample_rate
            results.append(res)
            
        return results

    def analyze_file(self, file_path: str):
        audio, sr = soundfile.read(file_path)
        chunks_result = self.predict_full_audio(audio, sr)
        
        has_cough = any(c['is_cough'] for c in chunks_result)
        max_cough_prob = max((c['cough_probability'] for c in chunks_result), default=0.0)
        
        # Nhóm cảnh báo theo thời gian thực
        all_detected = {}
        for c in chunks_result:
            pred = c['predicted_class']
            if pred not in all_detected:
                all_detected[pred] = []
            all_detected[pred].append((c['start_time'], c['end_time'], c['cough_probability']))
            
        return {
            'has_cough': has_cough,
            'max_cough_prob': max_cough_prob,
            'detected_classes': list(all_detected.keys()),
            'timeline': all_detected,
            'chunks': chunks_result
        }
"""
        new_lines.insert(-1, insert_code)

with open('mobilenet_detector.py', 'w') as f:
    f.writelines(new_lines)
