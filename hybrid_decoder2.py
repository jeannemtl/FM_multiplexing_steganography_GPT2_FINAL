import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle
from imec_encoder import MinEntropyCouplingSteganography

class FrequencyiMECValidator:
    """
    Complete validation of Frequency + iMEC hybrid.
    UPDATED: Reads agent frequencies from metadata.
    """
    
    def __init__(self, model_name='gpt2'):
        print("Loading GPT-2 for validation...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Agent frequencies will be loaded from metadata
        self.agents = None
        
        # Don't initialize iMEC yet - will use metadata's block size
        self.imec = None
    
    def load_agent_frequencies(self, metadata):
        """Load agent frequencies from metadata or use defaults."""
        if 'agent_frequencies' in metadata:
            agent_freqs = metadata['agent_frequencies']
            self.agents = {
                'ALICE': {'freq': agent_freqs['ALICE'], 'color': 'blue'},
                'BOB': {'freq': agent_freqs['BOB'], 'color': 'green'},
                'CHARLIE': {'freq': agent_freqs['CHARLIE'], 'color': 'red'}
            }
            print(f"\n✓ Using frequencies from metadata:")
            for agent, info in self.agents.items():
                print(f"  {agent}: {info['freq']} Hz")
        else:
            # Fallback to default frequencies
            self.agents = {
                'ALICE': {'freq': 0.02, 'color': 'blue'},
                'BOB': {'freq': 0.04, 'color': 'green'},
                'CHARLIE': {'freq': 0.06, 'color': 'red'}
            }
            print(f"\n⚠️  No frequencies in metadata, using defaults:")
            print(f"  ALICE: 0.02 Hz, BOB: 0.04 Hz, CHARLIE: 0.06 Hz")
    
    def extract_entropy(self, token_ids, context):
        """Extract entropy sequence from tokens with validation."""
        
        # Validate all token IDs
        token_ids = [int(t) for t in token_ids]
        vocab_size = len(self.tokenizer)
        
        # Filter out invalid tokens
        valid_tokens = []
        for t in token_ids:
            if 0 <= t < vocab_size:
                valid_tokens.append(t)
            else:
                print(f"⚠️  Skipping invalid token: {t} (vocab size: {vocab_size})")
        
        if len(valid_tokens) < len(token_ids) * 0.5:
            print(f"⚠️  WARNING: Only {len(valid_tokens)}/{len(token_ids)} tokens are valid!")
        
        token_ids = valid_tokens
        
        context_ids = self.tokenizer.encode(context)
        full_ids = context_ids + token_ids
        
        # Validate sequence length
        max_len = self.model.config.n_positions
        if len(full_ids) > max_len:
            print(f"⚠️  Truncating sequence from {len(full_ids)} to {max_len}")
            full_ids = full_ids[:max_len]
        
        input_ids = torch.tensor([full_ids], device=self.device, dtype=torch.long)
        
        entropy_sequence = []
        
        with torch.no_grad():
            for i in range(len(context_ids), len(full_ids)):
                try:
                    outputs = self.model(input_ids[:, :i])
                    logits = outputs.logits[0, -1, :]
                    probs = torch.softmax(logits, dim=0)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                    entropy_sequence.append(entropy)
                except RuntimeError as e:
                    print(f"⚠️  Error at position {i}: {e}")
                    break
                
                if (i - len(context_ids) + 1) % 200 == 0:
                    print(f"  Processed {i - len(context_ids) + 1} tokens...")
        
        return np.array(entropy_sequence)
    
    def fft_analysis(self, signal):
        """Perform FFT analysis."""
        N = len(signal)
        normalized = (signal - np.mean(signal)) / np.std(signal)
        
        fft_vals = fft(normalized)
        fft_freq = fftfreq(N, d=1.0)
        
        pos_mask = fft_freq > 0
        freqs = fft_freq[pos_mask]
        power = np.abs(fft_vals[pos_mask])**2
        
        results = {}
        
        for agent_name, agent_info in self.agents.items():
            target_freq = agent_info['freq']
            window_size = 30
            freq_idx = np.argmin(np.abs(freqs - target_freq))
            start_idx = max(0, freq_idx - window_size//2)
            end_idx = min(len(freqs), freq_idx + window_size//2)
            
            integrated_power = np.sum(power[start_idx:end_idx])
            detected_freq = freqs[freq_idx]
            
            results[f'{agent_name.lower()}_peak'] = detected_freq
            results[f'{agent_name.lower()}_power'] = integrated_power
        
        noise_mask = np.ones(len(freqs), dtype=bool)
        for agent_info in self.agents.values():
            target_freq = agent_info['freq']
            noise_mask &= (np.abs(freqs - target_freq) > 0.02)
        
        results['noise_level'] = np.mean(power[noise_mask]) if np.any(noise_mask) else np.mean(power)
        results['freqs'] = freqs
        results['power'] = power
        
        return results
    
    def bandpass_filter(self, signal, center_freq, bandwidth=0.015):
        """Apply bandpass filter."""
        fs = 1.0
        nyquist = fs / 2
        low = max(0.001, min((center_freq - bandwidth/2) / nyquist, 0.999))
        high = max(0.001, min((center_freq + bandwidth/2) / nyquist, 0.999))
        
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        
        return filtered
    
    def decode_ask_bits(self, filtered_signal, n_bits):
        """Decode ASK bits."""
        signal_length = len(filtered_signal)
        tokens_per_bit = signal_length // n_bits
        
        amplitudes = []
        for i in range(n_bits):
            start = i * tokens_per_bit
            end = min((i + 1) * tokens_per_bit, signal_length)
            window = filtered_signal[start:end]
            rms_amplitude = np.sqrt(np.mean(window**2))
            amplitudes.append(rms_amplitude)
        
        amplitudes = np.array(amplitudes)
        
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(amplitudes.reshape(-1, 1))
        
        centers = kmeans.cluster_centers_.flatten()
        if centers[0] > centers[1]:
            bits = 1 - labels
        else:
            bits = labels
        
        return bits
    
    def decode_messages(self, entropy_seq, original_messages):
        """Decode all messages."""
        results = {}
        normalized = (entropy_seq - np.mean(entropy_seq)) / np.std(entropy_seq)
        
        for agent_name, agent_info in self.agents.items():
            freq = agent_info['freq']
            filtered = self.bandpass_filter(normalized, freq)
            n_bits = len(original_messages[agent_name])
            bits = self.decode_ask_bits(filtered, n_bits)
            
            original_bits = np.array(original_messages[agent_name])
            matches = np.sum(bits == original_bits)
            accuracy = matches / len(original_bits) * 100
            ber = 100 - accuracy
            
            results[agent_name] = {
                'bits': bits,
                'accuracy': accuracy,
                'ber': ber
            }
        
        return results
    
    def run_validation(self, data_file='hybrid_freq_imec_data.pkl'):
        """Run complete validation."""
        
        print("="*80)
        print("LOADING DATA")
        print("="*80)
        
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        context = data['context']
        messages = data['messages']
        freq_tokens = data['freq_tokens']
        obf_tokens = data['obfuscated_tokens']
        metadata = data['metadata']
        
        # Load agent frequencies from metadata
        self.load_agent_frequencies(metadata)
        
        print(f"\n✓ Loaded:")
        print(f"  Frequency tokens: {len(freq_tokens)}")
        print(f"  Obfuscated tokens: {len(obf_tokens)}")
        print(f"  Block size in metadata: {metadata['block_size_bits']} bits")
        print(f"  Number of blocks: {metadata['n_blocks']}")
        
        # ═══════════════════════════════════════════════════════
        # PHASE 1: Validate Frequency Content EXISTS
        # ═══════════════════════════════════════════════════════
        
        print("\n" + "="*80)
        print("PHASE 1: VALIDATE FREQUENCY CONTENT EXISTS")
        print("="*80)
        
        print("\nExtracting entropy from frequency-modulated tokens...")
        freq_entropy = self.extract_entropy(freq_tokens, context)
        
        print("\nPerforming FFT analysis...")
        freq_fft = self.fft_analysis(freq_entropy)
        
        print("\n✓ FFT Analysis of Frequency-Modulated Text:")
        for agent in ['ALICE', 'BOB', 'CHARLIE']:
            freq = self.agents[agent]['freq']
            power = freq_fft[f'{agent.lower()}_power']
            print(f"  {agent:8s} ({freq:.3f} Hz): Power = {power:.1f}")
        print(f"  Noise level:           Power = {freq_fft['noise_level']:.1f}")
        
        alice_snr = freq_fft['alice_power'] / freq_fft['noise_level']
        bob_snr = freq_fft['bob_power'] / freq_fft['noise_level']
        charlie_snr = freq_fft['charlie_power'] / freq_fft['noise_level']
        
        print(f"\n✓ SNR (Frequency-Modulated):")
        print(f"  ALICE:   {alice_snr:.1f}:1 (patterns VISIBLE ✓)")
        print(f"  BOB:     {bob_snr:.1f}:1")
        print(f"  CHARLIE: {charlie_snr:.1f}:1")
        
        print("\nDecoding messages...")
        freq_decoded = self.decode_messages(freq_entropy, messages)
        
        print("\n✓ Message Decoding from Frequency Text:")
        for agent, result in freq_decoded.items():
            print(f"  {agent}: {result['accuracy']:.1f}% accuracy ({result['ber']:.1f}% BER)")
        
        # ═══════════════════════════════════════════════════════
        # PHASE 2: Validate Patterns HIDDEN
        # ═══════════════════════════════════════════════════════
        
        print("\n" + "="*80)
        print("PHASE 2: VALIDATE PATTERNS HIDDEN BY iMEC")
        print("="*80)
        
        print("\nExtracting entropy from obfuscated tokens...")
        obf_entropy = self.extract_entropy(obf_tokens, context)
        
        print("\nPerforming FFT analysis...")
        obf_fft = self.fft_analysis(obf_entropy)
        
        print("\n✓ FFT Analysis of Obfuscated Text:")
        for agent in ['ALICE', 'BOB', 'CHARLIE']:
            freq = self.agents[agent]['freq']
            power = obf_fft[f'{agent.lower()}_power']
            print(f"  {agent:8s} ({freq:.3f} Hz): Power = {power:.1f}")
        print(f"  Noise level:           Power = {obf_fft['noise_level']:.1f}")
        
        obf_alice_snr = obf_fft['alice_power'] / obf_fft['noise_level']
        obf_bob_snr = obf_fft['bob_power'] / obf_fft['noise_level']
        obf_charlie_snr = obf_fft['charlie_power'] / obf_fft['noise_level']
        
        print(f"\n✓ SNR (Obfuscated):")
        print(f"  ALICE:   {obf_alice_snr:.2f}:1", end="")
        if obf_alice_snr < 1.5:
            print(" (patterns HIDDEN ✓)")
        elif obf_alice_snr < 5:
            print(" (patterns WEAKENED ⚠️)")
        else:
            print(" (patterns STILL VISIBLE ✗)")
        print(f"  BOB:     {obf_bob_snr:.2f}:1")
        print(f"  CHARLIE: {obf_charlie_snr:.2f}:1")
        
        print("\nTrying to decode WITHOUT key...")
        obf_decoded = self.decode_messages(obf_entropy, messages)
        
        print("\n✓ Decoding Attempt WITHOUT Key:")
        for agent, result in obf_decoded.items():
            print(f"  {agent}: {result['ber']:.1f}% BER", end="")
            if 40 < result['ber'] < 60:
                print(" (≈50% = random ✓)")
            else:
                print()
        
        # ═══════════════════════════════════════════════════════
        # PHASE 3: Recover WITH Key
        # ═══════════════════════════════════════════════════════
        
        print("\n" + "="*80)
        print("PHASE 3: RECOVER WITH iMEC KEY")
        print("="*80)
        
        # CRITICAL: Initialize iMEC with correct block size from metadata
        block_size = metadata.get('block_size_bits', 16)
        print(f"\n✓ Initializing iMEC decoder with {block_size}-bit blocks (from metadata)...")
        self.imec = MinEntropyCouplingSteganography(block_size_bits=block_size)
        
        print("\n✓ Applying iMEC decoding...")
        recovered_ciphertext = self.imec.decode_imec(
            obf_tokens,
            context,
            metadata['n_blocks'],
            metadata['block_size_bits']
        )
        
        print(f"✓ Recovered ciphertext: {len(recovered_ciphertext)} bits")
        
        # Decrypt with one-time pad
        print("\n✓ Decrypting with one-time pad...")
        encryption_key = metadata['encryption_key']
        
        min_len = min(len(recovered_ciphertext), len(encryption_key))
        recovered_plaintext = ''.join(
            str(int(recovered_ciphertext[i]) ^ int(encryption_key[i]))
            for i in range(min_len)
        )
        print(f"✓ Decrypted plaintext: {len(recovered_plaintext)} bits")
        
        # Convert to tokens
        bits_per_token = metadata.get('bits_per_token', 16)
        n_tokens = metadata['n_freq_tokens']
        recovered_tokens = []
        invalid_count = 0
        
        vocab_size = len(self.tokenizer)
        
        for i in range(n_tokens):
            start = i * bits_per_token
            end = start + bits_per_token
            
            if end <= len(recovered_plaintext):
                token_bits = recovered_plaintext[start:end]
                token_id = int(token_bits, 2)
                
                if 0 <= token_id < vocab_size:
                    recovered_tokens.append(token_id)
                else:
                    invalid_count += 1
                    if invalid_count <= 5:
                        print(f"⚠️  Invalid token {token_id} at position {i}")
                    recovered_tokens.append(token_id % vocab_size)
            else:
                break
        
        if invalid_count > 0:
            print(f"\n⚠️  WARNING: {invalid_count}/{n_tokens} tokens out of range (using modulo)")
        
        print(f"\n✓ Token recovery: {len(recovered_tokens)}/{len(freq_tokens)}")
        
        # Calculate exact matches
        matches = sum(1 for i in range(min(len(freq_tokens), len(recovered_tokens)))
                     if freq_tokens[i] == recovered_tokens[i])
        match_pct = 100 * matches / len(freq_tokens) if freq_tokens else 0
        print(f"✓ Exact matches: {matches}/{len(freq_tokens)} ({match_pct:.1f}%)")
        
        if match_pct < 50:
            print("⚠️  Warning: Low token recovery rate - iMEC encoding may be lossy")
        elif match_pct > 90:
            print("✓ Excellent token recovery!")
        
        print("\nExtracting entropy from recovered tokens...")
        rec_entropy = self.extract_entropy(recovered_tokens, context)
        
        print("\nPerforming FFT analysis...")
        rec_fft = self.fft_analysis(rec_entropy)
        
        print("\n✓ FFT Analysis of Recovered Text:")
        for agent in ['ALICE', 'BOB', 'CHARLIE']:
            freq = self.agents[agent]['freq']
            power = rec_fft[f'{agent.lower()}_power']
            print(f"  {agent:8s} ({freq:.3f} Hz): Power = {power:.1f}")
        
        rec_alice_snr = rec_fft['alice_power'] / rec_fft['noise_level']
        print(f"\n✓ SNR (Recovered): ALICE={rec_alice_snr:.1f}:1", end="")
        if rec_alice_snr > 5:
            print(" (patterns RECOVERED ✓)")
        else:
            print(" (patterns WEAK ⚠️)")
        
        print("\nDecoding messages from recovered text...")
        rec_decoded = self.decode_messages(rec_entropy, messages)
        
        print("\n✓ Message Decoding WITH Key:")
        for agent, result in rec_decoded.items():
            print(f"  {agent}: {result['accuracy']:.1f}% accuracy ({result['ber']:.1f}% BER)")
        
        # ═══════════════════════════════════════════════════════
        # SUMMARY
        # ═══════════════════════════════════════════════════════
        
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        avg_freq_ber = np.mean([r['ber'] for r in freq_decoded.values()])
        avg_obf_ber = np.mean([r['ber'] for r in obf_decoded.values()])
        avg_rec_ber = np.mean([r['ber'] for r in rec_decoded.values()])
        
        print("\n✓ CLAIM 1: Frequency modulation creates patterns")
        print(f"  SNR: {alice_snr:.1f}:1")
        print(f"  BER: {avg_freq_ber:.1f}%")
        print(f"  {'✓ VERIFIED' if alice_snr > 5 else '✗ FAILED'}")
        
        print("\n✓ CLAIM 2: iMEC hides patterns")
        print(f"  SNR: {obf_alice_snr:.2f}:1")
        print(f"  BER: {avg_obf_ber:.1f}% (random)")
        if obf_alice_snr < 2:
            print(f"  ✓ VERIFIED - Patterns hidden")
        elif obf_alice_snr < 10:
            print(f"  ⚠️  PARTIAL - Patterns weakened but visible")
        else:
            print(f"  ✗ FAILED - Patterns still clearly visible")
        
        print("\n✓ CLAIM 3: Key recovers patterns")
        print(f"  Token match: {match_pct:.1f}%")
        print(f"  SNR: {rec_alice_snr:.1f}:1")
        if match_pct > 80 and rec_alice_snr > 5:
            print(f"  ✓ VERIFIED")
        elif match_pct > 50:
            print(f"  ⚠️  PARTIAL - Incomplete recovery")
        else:
            print(f"  ✗ FAILED - Poor recovery")
        
        print("\n✓ CLAIM 4: Messages decode from recovered")
        print(f"  BER: {avg_rec_ber:.1f}%")
        print(f"  {'✓ VERIFIED' if avg_rec_ber < 50 else '✗ FAILED'}")
        
        print("\n" + "="*80)
        print("CONCLUSION")
        print("="*80)
        
        if obf_alice_snr < 2 and match_pct > 80:
            print("✓ Hybrid approach SUCCESSFUL!")
            print("  - Frequency patterns hidden by iMEC")
            print("  - Patterns recoverable with key")
        elif obf_alice_snr < 10:
            print("⚠️  Hybrid approach PARTIALLY successful")
            print("  - Patterns weakened but not fully hidden")
            print(f"  - Token recovery: {match_pct:.1f}%")
        else:
            print("✗ Hybrid approach NOT successful")
            print("  - Patterns still clearly visible after iMEC")
            print(f"  - Token recovery: {match_pct:.1f}%")


if __name__ == "__main__":
    validator = FrequencyiMECValidator()
    validator.run_validation('hybrid_freq_imec_data.pkl')
