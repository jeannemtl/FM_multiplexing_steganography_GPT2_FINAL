import pickle
import numpy as np

with open('hybrid_freq_imec_data.pkl', 'rb') as f:
    data = pickle.load(f)

key = data['metadata']['encryption_key']
print(f"Encryption key uniformity: {np.mean(key):.3f}")
print(f"Expected: ~0.500")
print(f"Key length: {len(key)}")
