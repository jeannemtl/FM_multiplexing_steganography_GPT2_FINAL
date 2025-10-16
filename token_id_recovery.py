import numpy as np
import matplotlib.pyplot as plt
from scipy import fft, signal
import pickle
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class TokenIDMessageRecovery:
    """
    CORRECT APPROACH: FFT on token IDs, not embeddings!
    
    The FM bias affects token selection directly, so we should analyze
    the token ID sequence, not the embedding space.
    """
    
    def __init__(self, model_name='gpt2'):
        print("Loading GPT-2 for token ID-based recovery...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def load_data(self):
        """Load data"""
        with open('hybrid_freq_imec_data_ULTRA.pkl', 'rb') as f:
            data = pickle.load(f)
        with open('recovered_ciphertext.txt', 'r') as f:
            recovered_ciphertext = f.read().strip()
        return data, recovered_ciphertext
    
    def recover_fm_tokens(self, ciphertext_bits, encryption_key, bits_per_token=16):
        """Decrypt and recover FM tokens"""
        # Decrypt
        plaintext_bits = ''.join(
            str(int(ciphertext_bits[i]) ^ int(encryption_key[i]))
            for i in range(len(ciphertext_bits))
        )
        
        # Convert to tokens
        recovered_tokens = []
        for i in range(0, len(plaintext_bits), bits_per_token):
            token_bits = plaintext_bits[i:i+bits_per_token]
            if len(token_bits) == bits_per_token:
                token_id = int(token_bits, 2)
                recovered_tokens.append(token_id)
        
        return recovered_tokens
    
    def improved_demodulation_on_token_ids(self, token_ids, carrier_freq, n_bits, 
                                          sequence_length):
        """
        Perform FFT analysis directly on TOKEN IDs (not embeddings!)
        """
        # Normalize token IDs to [0, 1] range
        token_array = np.array(token_ids, dtype=float)
        token_normalized = (token_array - np.mean(token_array)) / (np.std(token_array) + 1e-10)
        
        # Bandpass filter around carrier frequency
        bandwidth = 0.010  # Wider bandwidth
        
        # FFT on token IDs
        fft_result = np.fft.fft(token_normalized)
        freqs = np.fft.fftfreq(len(token_normalized))
        
        # Bandpass filter
        mask = np.abs(freqs - carrier_freq) < bandwidth
        filtered_fft = fft_result * mask
        filtered_signal = np.fft.ifft(filtered_fft).real
        
        # Get envelope using Hilbert transform
        analytic = signal.hilbert(filtered_signal)
        envelope = np.abs(analytic)
        
        # Smooth envelope
        window = 20
        envelope_smooth = np.convolve(envelope, np.ones(window)/window, mode='same')
        
        # Normalize envelope
        envelope_smooth = (envelope_smooth - np.min(envelope_smooth))
        if np.max(envelope_smooth) > 0:
            envelope_smooth = envelope_smooth / np.max(envelope_smooth)
        
        # Decode bits
        tokens_per_bit = sequence_length // n_bits
        decoded_bits = []
        
        for i in range(n_bits):
            start_idx = i * tokens_per_bit
            end_idx = min((i + 1) * tokens_per_bit, len(envelope_smooth))
            
            # Average amplitude in this bit period
            bit_amplitude = np.mean(envelope_smooth[start_idx:end_idx])
            
            # Decode: high amplitude = 1, low = 0
            decoded_bit = 1 if bit_amplitude > 0.5 else 0
            decoded_bits.append(decoded_bit)
        
        return decoded_bits, envelope_smooth, filtered_signal
    
    def analyze_with_token_ids(self, original_tokens, recovered_tokens, data):
        """
        Comprehensive analysis using TOKEN IDs instead of embeddings
        """
        print("\n" + "="*80)
        print("TOKEN ID-BASED MESSAGE RECOVERY")
        print("="*80)
        print("KEY INSIGHT: Analyzing token IDs directly, not embeddings!")
        print("="*80)
        
        # Token comparison
        matches = sum(1 for o, r in zip(original_tokens, recovered_tokens) if o == r)
        token_accuracy = matches / len(original_tokens) * 100
        print(f"\n✓ Token Recovery: {token_accuracy:.1f}% ({matches}/{len(original_tokens)})")
        
        # Message recovery for each agent
        agents = list(data['metadata']['agent_frequencies'].keys())
        messages = data['messages']
        agent_freqs = data['metadata']['agent_frequencies']
        
        results = {}
        
        fig, axes = plt.subplots(len(agents), 4, figsize=(20, 4*len(agents)))
        
        for idx, agent in enumerate(agents):
            carrier = agent_freqs[agent]
            original_bits = messages[agent]
            n_bits = len(original_bits)
            
            print(f"\n{agent} (carrier: {carrier:.3f} Hz):")
            
            # Demodulation on ORIGINAL token IDs
            orig_decoded, orig_envelope, orig_filtered = self.improved_demodulation_on_token_ids(
                original_tokens, carrier, n_bits, len(original_tokens)
            )
            
            # Demodulation on RECOVERED token IDs
            rec_decoded, rec_envelope, rec_filtered = self.improved_demodulation_on_token_ids(
                recovered_tokens, carrier, n_bits, len(recovered_tokens)
            )
            
            # Calculate accuracies
            orig_accuracy = sum(1 for a, b in zip(original_bits, orig_decoded) 
                              if a == b) / n_bits * 100
            rec_accuracy = sum(1 for a, b in zip(original_bits, rec_decoded) 
                             if a == b) / n_bits * 100
            
            print(f"  Original message:      {original_bits}")
            print(f"  From original tokens:  {orig_decoded} ({orig_accuracy:.1f}%)")
            print(f"  From recovered tokens: {rec_decoded} ({rec_accuracy:.1f}%)")
            
            results[agent] = {
                'original_bits': original_bits,
                'from_original_tokens': orig_decoded,
                'from_recovered_tokens': rec_decoded,
                'orig_accuracy': orig_accuracy,
                'rec_accuracy': rec_accuracy
            }
            
            # Visualization
            ax = axes[idx] if len(agents) > 1 else axes
            
            # Plot 1: Raw token IDs
            ax[0].plot(original_tokens, label='Original', alpha=0.7, linewidth=0.5)
            ax[0].plot(recovered_tokens, label='Recovered', alpha=0.7, linewidth=0.5, linestyle='--')
            ax[0].set_title(f'{agent}: Token ID Sequences')
            ax[0].set_xlabel('Position')
            ax[0].set_ylabel('Token ID')
            ax[0].legend()
            ax[0].grid(True, alpha=0.3)
            
            # Plot 2: Filtered signals
            ax[1].plot(orig_filtered, label='Original tokens', alpha=0.7)
            ax[1].plot(rec_filtered, label='Recovered tokens', alpha=0.7, linestyle='--')
            ax[1].set_title(f'{agent}: Filtered Signal ({carrier:.3f} Hz)')
            ax[1].set_xlabel('Token Position')
            ax[1].set_ylabel('Amplitude')
            ax[1].legend()
            ax[1].grid(True, alpha=0.3)
            
            # Plot 3: Envelopes with bit boundaries
            ax[2].plot(orig_envelope, label='Original', alpha=0.7)
            ax[2].plot(rec_envelope, label='Recovered', alpha=0.7, linestyle='--')
            
            # Mark bit boundaries
            tokens_per_bit = len(orig_envelope) // n_bits
            for i in range(n_bits + 1):
                ax[2].axvline(x=i*tokens_per_bit, color='gray', 
                            linestyle=':', alpha=0.5)
            
            # Mark bit values
            for i in range(n_bits):
                mid = (i + 0.5) * tokens_per_bit
                ax[2].text(mid, 1.1, str(original_bits[i]), 
                         ha='center', fontweight='bold', fontsize=9)
            
            ax[2].set_title(f'{agent}: Envelope Detection (Token IDs)')
            ax[2].set_xlabel('Token Position')
            ax[2].set_ylabel('Envelope Amplitude')
            ax[2].legend()
            ax[2].grid(True, alpha=0.3)
            ax[2].set_ylim([-0.1, 1.3])
            
            # Plot 4: Results
            ax[3].axis('off')
            result_text = f"""
{agent} Results:

Original Message:
  {original_bits}

From Original Tokens:
  {orig_decoded}
  Accuracy: {orig_accuracy:.1f}%

From Recovered Tokens:
  {rec_decoded}
  Accuracy: {rec_accuracy:.1f}%

Token Recovery: {token_accuracy:.1f}%

METHOD: FFT on Token IDs
(NOT embeddings!)
"""
            color = 'lightgreen' if rec_accuracy > 80 else 'lightyellow' if rec_accuracy > 60 else 'lightcoral'
            ax[3].text(0.1, 0.5, result_text, fontsize=9, family='monospace',
                      va='center', bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('token_id_message_recovery.png', dpi=150, bbox_inches='tight')
        print("\n✓ Saved: token_id_message_recovery.png")
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Token Recovery: {token_accuracy:.1f}%")
        print("\nMessage Recovery (using TOKEN IDs):")
        for agent, res in results.items():
            print(f"  {agent}:")
            print(f"    From original tokens:  {res['orig_accuracy']:.1f}%")
            print(f"    From recovered tokens: {res['rec_accuracy']:.1f}%")
        
        # Diagnosis
        print("\n" + "="*80)
        print("DIAGNOSIS")
        print("="*80)
        
        avg_orig_acc = np.mean([r['orig_accuracy'] for r in results.values()])
        avg_rec_acc = np.mean([r['rec_accuracy'] for r in results.values()])
        
        if avg_orig_acc > 80:
            print(f"✓ EXCELLENT! Token ID approach works! ({avg_orig_acc:.1f}% from original)")
            if avg_rec_acc > 80:
                print(f"✓ Complete pipeline SUCCESS! ({avg_rec_acc:.1f}% from recovered)")
            else:
                print(f"⚠️  Token loss during iMEC affected recovery ({avg_rec_acc:.1f}%)")
        elif avg_orig_acc > 60:
            print(f"✓ Good progress with token IDs ({avg_orig_acc:.1f}%)")
            print("   → May need: longer sequences, cleaner carrier separation")
        else:
            print(f"⚠️  Still low ({avg_orig_acc:.1f}%) even with token IDs")
            print("   → Issue: Token ID sequence may not capture FM bias well")
            print("   → Consider: Alternative encoding (direct bit patterns)")
        
        return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    analyzer = TokenIDMessageRecovery()
    
    print("Loading data...")
    data, ciphertext = analyzer.load_data()
    
    print("Recovering FM tokens...")
    recovered_tokens = analyzer.recover_fm_tokens(
        ciphertext,
        data['metadata']['encryption_key'],
        data['metadata']['bits_per_token']
    )
    
    print("Analyzing with TOKEN IDs (not embeddings)...")
    results = analyzer.analyze_with_token_ids(
        data['freq_tokens'],
        recovered_tokens,
        data
    )
