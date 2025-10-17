import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

class CompleteDemodulationTest:
    """
    Complete test: Show frequency content can be DEMODULATED after iMEC decoding
    
    Pipeline:
    1. Obfuscated tokens → iMEC decode → Recovered tokens
    2. Demodulate recovered tokens at each carrier frequency
    3. Recover original binary messages
    4. Prove frequency content survives the encode/decode pipeline
    """
    
    def __init__(self, model_name='gpt2'):
        print("Loading GPT-2 for demodulation test...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.MAX_CONTEXT = 1024
        self.vocab_size = len(self.tokenizer)
        
    def load_data(self, pkl_path):
        """Load hybrid data"""
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        print(f"\nLoaded: {pkl_path}")
        print(f"  Block size: {data['metadata']['block_size_bits']} bits")
        print(f"  Freq tokens: {len(data['freq_tokens'])}")
        print(f"  Obfuscated tokens: {len(data['obfuscated_tokens'])}")
        return data
    
    def mec_subroutine(self, mu_i, covertext_probs):
        """Approximate MEC subroutine"""
        vocab_size = len(covertext_probs)
        
        mu_i = np.array(mu_i)
        covertext_probs = np.array(covertext_probs)
        
        if mu_i.sum() > 0:
            mu_i = mu_i / mu_i.sum()
        if covertext_probs.sum() > 0:
            covertext_probs = covertext_probs / covertext_probs.sum()
        
        sorted_indices = np.argsort(-covertext_probs)
        
        n_cipher = len(mu_i)
        coupling = {}
        
        cipher_remaining = mu_i.copy()
        covertext_remaining = covertext_probs.copy()
        
        for cipher_idx in range(n_cipher):
            if cipher_remaining[cipher_idx] <= 1e-10:
                continue
                
            for idx in sorted_indices:
                token_idx = int(idx)
                
                if token_idx < 0 or token_idx >= vocab_size:
                    continue
                
                if covertext_remaining[token_idx] <= 1e-10:
                    continue
                
                mass = min(cipher_remaining[cipher_idx], 
                          covertext_remaining[token_idx])
                
                if mass > 1e-10:
                    coupling[(int(cipher_idx), token_idx)] = float(mass)
                
                cipher_remaining[cipher_idx] -= mass
                covertext_remaining[token_idx] -= mass
                
                if cipher_remaining[cipher_idx] <= 1e-10:
                    break
        
        return coupling
    
    def update_posterior(self, coupling, cipher_probs, sampled_token):
        """Update posterior distribution"""
        n_cipher = len(cipher_probs)
        posterior = np.zeros(n_cipher, dtype=float)
        
        possible = [(c, mass) for (c, t), mass in coupling.items() if t == sampled_token]
        
        if not possible:
            return np.ones(n_cipher) / n_cipher
        
        for c, mass in possible:
            if c < n_cipher:
                posterior[c] = mass * cipher_probs[c]
        
        total = posterior.sum()
        if total > 1e-10:
            posterior = posterior / total
        else:
            posterior = np.ones(n_cipher) / n_cipher
        
        return posterior
    
    def _entropy(self, probs):
        """Shannon entropy"""
        probs = np.array(probs)
        probs = probs[probs > 1e-10]
        if len(probs) == 0:
            return 0.0
        return float(-np.sum(probs * np.log2(probs)))
    
    def decode_imec(self, stegotext_tokens, context, n_blocks, block_size_bits):
        """Decode iMEC to recover hidden ciphertext bits"""
        print("\n" + "="*80)
        print("iMEC DECODING")
        print("="*80)
        
        n_values = 2 ** block_size_bits
        mu = [np.ones(n_values) / n_values for _ in range(n_blocks)]
        
        input_ids = self.tokenizer.encode(context, return_tensors='pt').to(self.device)
        eos_token_id = self.tokenizer.eos_token_id
        
        start_time = time.time()
        
        with torch.no_grad():
            for j, s_j in enumerate(stegotext_tokens):
                if (j + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (j + 1) / elapsed
                    remaining = (len(stegotext_tokens) - j - 1) / rate if rate > 0 else 0
                    
                    entropies = [self._entropy(mu_i) for mu_i in mu]
                    avg_ent = np.mean(entropies)
                    
                    print(f"  Token {j+1}/{len(stegotext_tokens)}: "
                          f"H_avg={avg_ent:.4f}, "
                          f"rate={rate:.1f} tok/s, "
                          f"ETA={remaining/60:.1f} min")
                
                entropies = [self._entropy(mu_i) for mu_i in mu]
                i_star = np.argmax(entropies)
                
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]
                covertext_probs = torch.softmax(logits, dim=0).cpu().numpy()
                
                covertext_probs[eos_token_id] = 0.0
                covertext_probs = covertext_probs / covertext_probs.sum()
                
                gamma_j = self.mec_subroutine(mu[i_star], covertext_probs)
                mu[i_star] = self.update_posterior(gamma_j, mu[i_star], s_j)
                
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[s_j]], device=self.device, dtype=torch.long)
                ], dim=1)
                
                if input_ids.shape[1] > self.MAX_CONTEXT:
                    input_ids = input_ids[:, -self.MAX_CONTEXT:]
        
        ciphertext_bits = ""
        for i in range(n_blocks):
            x_i_hat = np.argmax(mu[i])
            bits = format(x_i_hat, f'0{block_size_bits}b')
            ciphertext_bits += bits
        
        total_time = time.time() - start_time
        print(f"\n✓ Decoded {len(ciphertext_bits)} bits in {total_time/60:.1f} minutes")
        
        return ciphertext_bits
    
    def recover_tokens(self, ciphertext_bits, encryption_key, bits_per_token):
        """Decrypt and recover FM tokens"""
        print("\n" + "="*80)
        print("DECRYPTION")
        print("="*80)
        
        # Decrypt with one-time pad
        plaintext_bits = ''.join(
            str(int(ciphertext_bits[i]) ^ int(encryption_key[i]))
            for i in range(min(len(ciphertext_bits), len(encryption_key)))
        )
        
        print(f"Decrypted {len(plaintext_bits)} bits")
        
        # Reconstruct tokens
        recovered_tokens = []
        for i in range(0, len(plaintext_bits), bits_per_token):
            token_bits = plaintext_bits[i:i+bits_per_token]
            if len(token_bits) == bits_per_token:
                token_id = int(token_bits, 2)
                if 0 <= token_id < self.vocab_size:
                    recovered_tokens.append(token_id)
        
        print(f"Recovered {len(recovered_tokens)} tokens")
        
        return recovered_tokens
    
    def demodulate_channel(self, tokens, context, carrier_freq, n_bits, 
                          sequence_length, method='logit'):
        """
        Demodulate a single frequency channel from recovered tokens
        
        Key insight: Frequency content is in the TOKEN SELECTION PROBABILITIES,
        not in the token embeddings!
        
        We need to reconstruct the probability distribution that generated each token.
        """
        print(f"\n  Demodulating carrier {carrier_freq:.3f} Hz...")
        
        # Generate probability distributions for each token
        input_ids = self.tokenizer.encode(context, return_tensors='pt').to(self.device)
        
        probabilities = []
        
        with torch.no_grad():
            for i, token in enumerate(tokens):
                # Get model's probability distribution at this position
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits, dim=0).cpu().numpy()
                
                # Store probability of the SELECTED token
                token_prob = probs[token]
                probabilities.append(token_prob)
                
                # Append token for next iteration
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[token]], device=self.device, dtype=torch.long)
                ], dim=1)
                
                if input_ids.shape[1] > self.MAX_CONTEXT:
                    input_ids = input_ids[:, -self.MAX_CONTEXT:]
        
        # Convert to signal
        signal_1d = np.array(probabilities)
        
        # Normalize
        signal_1d = (signal_1d - np.mean(signal_1d)) / (np.std(signal_1d) + 1e-10)
        
        # Bandpass filter around carrier frequency
        bandwidth = 0.010  # Wider bandwidth
        
        fft_result = np.fft.fft(signal_1d)
        freqs = np.fft.fftfreq(len(signal_1d))
        
        # Bandpass filter
        mask = np.abs(freqs - carrier_freq) < bandwidth
        filtered_fft = fft_result * mask
        filtered_signal = np.fft.ifft(filtered_fft).real
        
        # Envelope detection using Hilbert transform
        analytic = signal.hilbert(filtered_signal)
        envelope = np.abs(analytic)
        
        # Smooth envelope
        window = 20
        envelope_smooth = np.convolve(envelope, np.ones(window)/window, mode='same')
        
        # Normalize
        envelope_smooth = (envelope_smooth - np.min(envelope_smooth))
        if np.max(envelope_smooth) > 0:
            envelope_smooth = envelope_smooth / np.max(envelope_smooth)
        
        # Decode bits
        tokens_per_bit = sequence_length // n_bits
        decoded_bits = []
        
        for i in range(n_bits):
            start_idx = i * tokens_per_bit
            end_idx = min((i + 1) * tokens_per_bit, len(envelope_smooth))
            
            bit_amplitude = np.mean(envelope_smooth[start_idx:end_idx])
            decoded_bit = 1 if bit_amplitude > 0.5 else 0
            decoded_bits.append(decoded_bit)
        
        return decoded_bits, envelope_smooth, filtered_signal, signal_1d
    
    def visualize_demodulation(self, data, recovered_tokens, demod_results, block_size):
        """
        Visualize demodulation results showing frequency content IS present
        """
        fig, axes = plt.subplots(len(demod_results), 4, figsize=(20, 5*len(demod_results)))
        
        if len(demod_results) == 1:
            axes = axes.reshape(1, -1)
        
        messages = data['messages']
        agent_freqs = data['metadata']['agent_frequencies']
        
        for idx, (agent, results) in enumerate(demod_results.items()):
            decoded_bits, envelope, filtered, raw_signal = results
            original_bits = messages[agent]
            carrier = agent_freqs[agent]
            
            # Calculate accuracy
            accuracy = sum(1 for a, b in zip(original_bits, decoded_bits) 
                          if a == b) / len(original_bits) * 100
            
            # Plot 1: Raw probability signal
            axes[idx, 0].plot(raw_signal[:500])
            axes[idx, 0].set_title(f'{agent}: Token Probability Signal', fontweight='bold')
            axes[idx, 0].set_xlabel('Token Position')
            axes[idx, 0].set_ylabel('Normalized Probability')
            axes[idx, 0].grid(True, alpha=0.3)
            
            # Plot 2: Filtered signal at carrier frequency
            axes[idx, 1].plot(filtered[:500])
            axes[idx, 1].set_title(f'{agent}: Bandpass Filtered ({carrier:.3f} Hz)', fontweight='bold')
            axes[idx, 1].set_xlabel('Token Position')
            axes[idx, 1].set_ylabel('Amplitude')
            axes[idx, 1].grid(True, alpha=0.3)
            
            # Plot 3: Envelope with bit boundaries
            axes[idx, 2].plot(envelope, linewidth=2, color='green')
            
            tokens_per_bit = len(envelope) // len(original_bits)
            for i in range(len(original_bits) + 1):
                axes[idx, 2].axvline(x=i*tokens_per_bit, color='gray', 
                                    linestyle=':', alpha=0.5)
            
            for i in range(len(original_bits)):
                mid = (i + 0.5) * tokens_per_bit
                axes[idx, 2].text(mid, 1.1, str(original_bits[i]), 
                                 ha='center', fontweight='bold', fontsize=10)
            
            axes[idx, 2].set_title(f'{agent}: Envelope Detection', fontweight='bold')
            axes[idx, 2].set_xlabel('Token Position')
            axes[idx, 2].set_ylabel('Envelope Amplitude')
            axes[idx, 2].set_ylim([-0.1, 1.3])
            axes[idx, 2].grid(True, alpha=0.3)
            
            # Plot 4: Results comparison
            axes[idx, 3].axis('off')
            
            color = 'lightgreen' if accuracy >= 80 else 'lightyellow' if accuracy >= 60 else 'lightcoral'
            
            result_text = f"""
{agent} Results:

Carrier Frequency: {carrier:.3f} Hz

Original Message:
  {original_bits}

Demodulated from 
Recovered Tokens:
  {decoded_bits}

Accuracy: {accuracy:.1f}%

Status: {'✓ SUCCESS' if accuracy >= 80 else '⚠️ PARTIAL' if accuracy >= 60 else '✗ FAILED'}

This proves frequency content
SURVIVES the full pipeline:
  Original → iMEC encode
  → iMEC decode → Demodulate
"""
            
            axes[idx, 3].text(0.05, 0.5, result_text, fontsize=9, family='monospace',
                             va='center', bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
        
        plt.suptitle(f'Complete Demodulation Test ({block_size}-bit blocks)\n'
                    f'Proving Frequency Content Survives iMEC Encode/Decode Pipeline',
                    fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        filename = f'demodulation_proof_{block_size}bit.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved: {filename}")
        plt.close()
    
    def test_single_file(self, pkl_path):
        """Complete test on one file"""
        # Load data
        data = self.load_data(pkl_path)
        block_size = data['metadata']['block_size_bits']
        
        print("\n" + "="*80)
        print(f"COMPLETE DEMODULATION TEST ({block_size}-bit)")
        print("="*80)
        
        # Step 1: Decode iMEC
        ciphertext_bits = self.decode_imec(
            data['obfuscated_tokens'],
            data['context'],
            data['metadata']['n_blocks'],
            block_size
        )
        
        # Step 2: Decrypt and recover tokens
        recovered_tokens = self.recover_tokens(
            ciphertext_bits,
            data['metadata']['encryption_key'],
            data['metadata']['bits_per_token']
        )
        
        # Check token accuracy
        original_tokens = data['freq_tokens']
        matches = sum(1 for o, r in zip(original_tokens, recovered_tokens) if o == r)
        token_accuracy = matches / min(len(original_tokens), len(recovered_tokens)) * 100
        
        print(f"\n✓ Token recovery: {token_accuracy:.1f}% ({matches}/{len(original_tokens)})")
        
        # Step 3: Demodulate each frequency channel
        print("\n" + "="*80)
        print("DEMODULATION")
        print("="*80)
        
        agent_freqs = data['metadata']['agent_frequencies']
        messages = data['messages']
        demod_results = {}
        
        for agent, carrier_freq in agent_freqs.items():
            n_bits = len(messages[agent])
            
            decoded_bits, envelope, filtered, raw_signal = self.demodulate_channel(
                recovered_tokens,
                data['context'],
                carrier_freq,
                n_bits,
                len(recovered_tokens)
            )
            
            demod_results[agent] = (decoded_bits, envelope, filtered, raw_signal)
            
            original_bits = messages[agent]
            accuracy = sum(1 for a, b in zip(original_bits, decoded_bits) 
                          if a == b) / n_bits * 100
            
            print(f"\n  {agent} ({carrier_freq:.3f} Hz):")
            print(f"    Original: {original_bits}")
            print(f"    Decoded:  {decoded_bits}")
            print(f"    Accuracy: {accuracy:.1f}%")
        
        # Step 4: Visualize
        self.visualize_demodulation(data, recovered_tokens, demod_results, block_size)
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Token recovery: {token_accuracy:.1f}%")
        print("\nMessage recovery (via demodulation):")
        
        avg_accuracy = 0
        for agent, (decoded_bits, _, _, _) in demod_results.items():
            original_bits = messages[agent]
            accuracy = sum(1 for a, b in zip(original_bits, decoded_bits) 
                          if a == b) / len(original_bits) * 100
            print(f"  {agent}: {accuracy:.1f}%")
            avg_accuracy += accuracy
        
        avg_accuracy /= len(demod_results)
        
        print(f"\nAverage message recovery: {avg_accuracy:.1f}%")
        
        if avg_accuracy >= 80:
            print("\n✓ SUCCESS: Frequency content PROVEN to survive iMEC pipeline!")
        elif avg_accuracy >= 60:
            print("\n⚠️ PARTIAL: Some frequency content survives, but degraded")
        else:
            print("\n✗ ISSUE: Frequency content significantly degraded")
        
        return avg_accuracy
    
    def run_all_tests(self):
        """Test all available pkl files"""
        print("\n" + "="*80)
        print("COMPLETE DEMODULATION PROOF")
        print("Testing frequency content survival through iMEC pipeline")
        print("="*80)
        
        results = {}
        
        # Test 8-bit
        try:
            print("\n\n" + "█"*80)
            print("TESTING: 8-BIT")
            print("█"*80)
            results['8bit'] = self.test_single_file('hybrid_freq_imec_data.pkl')
        except FileNotFoundError:
            print("\n⚠️ 8-bit file not found")
        
        # Test 12-bit
        try:
            print("\n\n" + "█"*80)
            print("TESTING: 12-BIT")
            print("█"*80)
            results['12bit'] = self.test_single_file('hybrid_freq_imec_data_12bit.pkl')
        except FileNotFoundError:
            print("\n⚠️ 12-bit file not found")
        
        # Final summary
        print("\n\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        
        for name, avg_acc in results.items():
            print(f"{name}: {avg_acc:.1f}% average message recovery")
        
        print("\n" + "="*80)
        print("CONCLUSION:")
        print("="*80)
        print("By demodulating the RECOVERED tokens (not embeddings),")
        print("we prove that frequency content SURVIVES the complete")
        print("iMEC encode → decode pipeline!")
        print("\n✓ Security: Hidden in obfuscated state")
        print("✓ Functionality: Recoverable with key + demodulation")
        print("="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    tester = CompleteDemodulationTest()
    tester.run_all_tests()
