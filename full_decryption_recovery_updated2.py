import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class ProperFrequencyDemodulation:
    """
    CORRECT demodulation approach:
    Extract frequency signal from TOKEN SELECTION PROBABILITIES,
    not from token embeddings!
    
    The key insight: Frequency modulation affects WHICH tokens are selected
    by biasing their probabilities. We need to measure this bias deviation.
    """
    
    def __init__(self, model_name='gpt2'):
        print("Loading GPT-2 for proper demodulation...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.MAX_CONTEXT = 1024
    
    def load_data(self):
        """Load all data"""
        with open('hybrid_freq_imec_data_10bit.pkl', 'rb') as f:
            data = pickle.load(f)
        with open('recovered_ciphertext.txt', 'r') as f:
            recovered_ciphertext = f.read().strip()
        return data, recovered_ciphertext
    
    def recover_fm_tokens(self, ciphertext_bits, encryption_key, bits_per_token=16):
        """Decrypt and recover FM tokens"""
        plaintext_bits = ''.join(
            str(int(ciphertext_bits[i]) ^ int(encryption_key[i]))
            for i in range(min(len(ciphertext_bits), len(encryption_key)))
        )
        
        recovered_tokens = []
        for i in range(0, len(plaintext_bits), bits_per_token):
            token_bits = plaintext_bits[i:i+bits_per_token]
            if len(token_bits) == bits_per_token:
                token_id = int(token_bits, 2)
                if 0 <= token_id < len(self.tokenizer):
                    recovered_tokens.append(token_id)
        
        return recovered_tokens
    
    def extract_probability_deviation_signal(self, tokens, context):
        """
        THE KEY METHOD!
        
        For each token, compute how "surprising" it was compared to
        the base GPT-2 distribution. This deviation captures the
        frequency modulation bias!
        """
        print("\n  Extracting probability deviation signal...")
        
        input_ids = self.tokenizer.encode(context, return_tensors='pt').to(self.device)
        eos_token_id = self.tokenizer.eos_token_id
        
        # For each position, get the log-probability of the selected token
        log_probs = []
        
        with torch.no_grad():
            for i, token in enumerate(tokens):
                # Get base GPT-2 distribution at this position
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]
                
                # Remove EOS token (same as encoding)
                logits[eos_token_id] = -float('inf')
                
                # Get log probabilities
                log_probs_dist = torch.log_softmax(logits, dim=0).cpu().numpy()
                
                # Record log probability of the SELECTED token
                token_log_prob = log_probs_dist[token]
                log_probs.append(token_log_prob)
                
                # Append token for next iteration
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[token]], device=self.device, dtype=torch.long)
                ], dim=1)
                
                if input_ids.shape[1] > self.MAX_CONTEXT:
                    input_ids = input_ids[:, -self.MAX_CONTEXT:]
                
                if (i + 1) % 50 == 0:
                    print(f"    Processed {i+1}/{len(tokens)} tokens...")
        
        # Convert to deviation signal
        # Negative log prob = "surprise" 
        # High surprise = token was unlikely = strong bias was applied
        deviation_signal = -np.array(log_probs)
        
        # Normalize
        deviation_signal = (deviation_signal - np.mean(deviation_signal)) / (np.std(deviation_signal) + 1e-10)
        
        print(f"  ✓ Extracted deviation signal (mean={np.mean(deviation_signal):.3f}, std={np.std(deviation_signal):.3f})")
        
        return deviation_signal
    
    def demodulate_carrier(self, deviation_signal, carrier_freq, n_bits, bandwidth=0.012):
        """
        Demodulate a single carrier frequency from the deviation signal
        
        Steps:
        1. Bandpass filter around carrier frequency
        2. Hilbert transform to get envelope (ASK demodulation)
        3. Smooth and threshold to extract bits
        """
        # Bandpass filter
        fft_result = np.fft.fft(deviation_signal)
        freqs = np.fft.fftfreq(len(deviation_signal))
        
        # Filter around carrier
        mask = np.abs(freqs - carrier_freq) < bandwidth
        filtered_fft = fft_result * mask
        filtered_signal = np.fft.ifft(filtered_fft).real
        
        # Envelope detection using Hilbert transform
        analytic = signal.hilbert(filtered_signal)
        envelope = np.abs(analytic)
        
        # Smooth envelope
        window_size = max(5, len(envelope) // (n_bits * 4))
        envelope_smooth = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
        
        # Normalize
        envelope_smooth = (envelope_smooth - np.min(envelope_smooth))
        if np.max(envelope_smooth) > 0:
            envelope_smooth = envelope_smooth / np.max(envelope_smooth)
        
        # Decode bits
        tokens_per_bit = len(envelope_smooth) // n_bits
        bit_amplitudes = []
        
        for i in range(n_bits):
            start_idx = i * tokens_per_bit
            end_idx = min((i + 1) * tokens_per_bit, len(envelope_smooth))
            bit_amplitude = np.mean(envelope_smooth[start_idx:end_idx])
            bit_amplitudes.append(bit_amplitude)
        
        # Adaptive threshold
        threshold = np.median(bit_amplitudes)
        decoded_bits = [1 if amp > threshold else 0 for amp in bit_amplitudes]
        
        return decoded_bits, envelope_smooth, filtered_signal, bit_amplitudes, threshold
    
    def test_demodulation(self, tokens, context, messages, agent_freqs, label=""):
        """
        Test demodulation on a token sequence
        """
        print(f"\n{'='*80}")
        print(f"DEMODULATION TEST: {label}")
        print(f"{'='*80}")
        
        # Extract the probability deviation signal
        deviation_signal = self.extract_probability_deviation_signal(tokens, context)
        
        # Demodulate each agent's carrier
        results = {}
        
        for agent, carrier_freq in agent_freqs.items():
            print(f"\n{agent} (carrier: {carrier_freq:.3f} Hz):")
            
            original_bits = messages[agent]
            n_bits = len(original_bits)
            
            decoded_bits, envelope, filtered, amplitudes, threshold = self.demodulate_carrier(
                deviation_signal, carrier_freq, n_bits
            )
            
            accuracy = sum(1 for a, b in zip(original_bits, decoded_bits) if a == b) / n_bits * 100
            
            print(f"  Original:  {original_bits}")
            print(f"  Decoded:   {decoded_bits}")
            print(f"  Accuracy:  {accuracy:.1f}%")
            print(f"  Threshold: {threshold:.3f}")
            print(f"  Amp range: [{np.min(amplitudes):.3f}, {np.max(amplitudes):.3f}]")
            
            results[agent] = {
                'original': original_bits,
                'decoded': decoded_bits,
                'accuracy': accuracy,
                'envelope': envelope,
                'filtered': filtered,
                'amplitudes': amplitudes,
                'threshold': threshold
            }
        
        return results, deviation_signal
    
    def visualize_results(self, orig_results, rec_results, orig_deviation, 
                         rec_deviation, messages, agent_freqs, token_accuracy):
        """
        Create comprehensive visualization
        """
        agents = list(agent_freqs.keys())
        
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(len(agents) + 1, 4, hspace=0.3, wspace=0.3)
        
        # Row 0: Deviation signals comparison
        ax_dev1 = fig.add_subplot(gs[0, 0:2])
        ax_dev1.plot(orig_deviation[:500], label='Original tokens', alpha=0.7, linewidth=2)
        ax_dev1.set_title('Probability Deviation Signal (Original Tokens)', 
                         fontsize=12, fontweight='bold')
        ax_dev1.set_xlabel('Token Position')
        ax_dev1.set_ylabel('Deviation (normalized)')
        ax_dev1.grid(True, alpha=0.3)
        ax_dev1.legend()
        
        ax_dev2 = fig.add_subplot(gs[0, 2:4])
        ax_dev2.plot(rec_deviation[:500], label='Recovered tokens', alpha=0.7, 
                    linewidth=2, color='orange')
        ax_dev2.set_title('Probability Deviation Signal (Recovered Tokens)', 
                         fontsize=12, fontweight='bold')
        ax_dev2.set_xlabel('Token Position')
        ax_dev2.set_ylabel('Deviation (normalized)')
        ax_dev2.grid(True, alpha=0.3)
        ax_dev2.legend()
        
        # Rows 1+: Each agent's results
        for idx, agent in enumerate(agents):
            row = idx + 1
            carrier = agent_freqs[agent]
            original_bits = messages[agent]
            n_bits = len(original_bits)
            
            orig_res = orig_results[agent]
            rec_res = rec_results[agent]
            
            # Plot 1: Filtered signals
            ax1 = fig.add_subplot(gs[row, 0])
            ax1.plot(orig_res['filtered'][:500], label='Original', alpha=0.7)
            ax1.plot(rec_res['filtered'][:500], label='Recovered', alpha=0.7, linestyle='--')
            ax1.set_title(f'{agent}: Bandpass Filtered @ {carrier:.3f} Hz', fontsize=10)
            ax1.set_xlabel('Position')
            ax1.set_ylabel('Amplitude')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Envelopes
            ax2 = fig.add_subplot(gs[row, 1])
            ax2.plot(orig_res['envelope'], label='Original', alpha=0.7)
            ax2.plot(rec_res['envelope'], label='Recovered', alpha=0.7, linestyle='--')
            
            # Mark bit boundaries
            tokens_per_bit = len(orig_res['envelope']) // n_bits
            for i in range(n_bits + 1):
                ax2.axvline(x=i*tokens_per_bit, color='gray', linestyle=':', alpha=0.3)
            
            # Mark original bits
            for i in range(n_bits):
                mid = (i + 0.5) * tokens_per_bit
                ax2.text(mid, 1.15, str(original_bits[i]), ha='center', 
                        fontsize=8, fontweight='bold')
            
            ax2.axhline(y=orig_res['threshold'], color='blue', linestyle='--', 
                       alpha=0.5, linewidth=1, label=f'Thresh={orig_res["threshold"]:.2f}')
            ax2.set_title(f'{agent}: Envelope Detection', fontsize=10)
            ax2.set_xlabel('Position')
            ax2.set_ylabel('Envelope')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([-0.1, 1.3])
            
            # Plot 3: Bit amplitudes
            ax3 = fig.add_subplot(gs[row, 2])
            x_pos = np.arange(n_bits)
            width = 0.35
            
            ax3.bar(x_pos - width/2, orig_res['amplitudes'], width, 
                   label='Original', alpha=0.7)
            ax3.bar(x_pos + width/2, rec_res['amplitudes'], width, 
                   label='Recovered', alpha=0.7)
            ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, 
                       label='Threshold=0.5')
            
            # Mark original bits
            for i in range(n_bits):
                ax3.text(i, 1.1, str(original_bits[i]), ha='center', 
                        fontsize=8, fontweight='bold')
            
            ax3.set_title(f'{agent}: Bit Amplitudes', fontsize=10)
            ax3.set_xlabel('Bit Position')
            ax3.set_ylabel('Amplitude')
            ax3.set_xticks(x_pos)
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.set_ylim([0, 1.3])
            
            # Plot 4: Results summary
            ax4 = fig.add_subplot(gs[row, 3])
            ax4.axis('off')
            
            orig_color = 'lightgreen' if orig_res['accuracy'] >= 80 else 'lightyellow' if orig_res['accuracy'] >= 60 else 'lightcoral'
            rec_color = 'lightgreen' if rec_res['accuracy'] >= 80 else 'lightyellow' if rec_res['accuracy'] >= 60 else 'lightcoral'
            
            summary_text = f"""
{agent} @ {carrier:.3f} Hz

Original Message:
  {original_bits}

From ORIGINAL tokens:
  {orig_res['decoded']}
  Accuracy: {orig_res['accuracy']:.1f}%

From RECOVERED tokens:
  {rec_res['decoded']}
  Accuracy: {rec_res['accuracy']:.1f}%

Token Recovery: {token_accuracy:.1f}%
"""
            
            ax4.text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
                    va='center', bbox=dict(boxstyle='round', 
                    facecolor=rec_color, alpha=0.5))
        
        plt.suptitle('Proper Probability-Based Demodulation Results', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        plt.savefig('proper_probability_demodulation.png', dpi=150, bbox_inches='tight')
        print("\n✓ Saved: proper_probability_demodulation.png")
        plt.close()
    
    def run_complete_test(self):
        """
        Run complete end-to-end test with proper demodulation
        """
        print("\n" + "="*80)
        print("PROPER PROBABILITY-BASED DEMODULATION TEST")
        print("="*80)
        print("\nThis test uses the CORRECT approach:")
        print("  - Extract probability deviation signal")
        print("  - Demodulate from token selection probabilities")
        print("  - NOT from token embeddings!")
        print("="*80)
        
        # Load data
        print("\nLoading data...")
        data, ciphertext = self.load_data()
        
        # Recover tokens
        print("Recovering FM tokens...")
        recovered_tokens = self.recover_fm_tokens(
            ciphertext,
            data['metadata']['encryption_key'],
            data['metadata']['bits_per_token']
        )
        
        original_tokens = data['freq_tokens']
        context = data['context']
        messages = data['messages']
        agent_freqs = data['metadata']['agent_frequencies']
        
        # Calculate token accuracy
        matches = sum(1 for o, r in zip(original_tokens, recovered_tokens) if o == r)
        token_accuracy = matches / len(original_tokens) * 100
        print(f"✓ Token recovery: {token_accuracy:.1f}% ({matches}/{len(original_tokens)})")
        
        # Test on ORIGINAL tokens (baseline)
        orig_results, orig_deviation = self.test_demodulation(
            original_tokens, context, messages, agent_freqs, 
            "ORIGINAL TOKENS (Baseline)"
        )
        
        # Test on RECOVERED tokens (full pipeline)
        rec_results, rec_deviation = self.test_demodulation(
            recovered_tokens, context, messages, agent_freqs,
            "RECOVERED TOKENS (After iMEC)"
        )
        
        # Visualize
        print("\nCreating visualization...")
        self.visualize_results(orig_results, rec_results, orig_deviation,
                              rec_deviation, messages, agent_freqs, token_accuracy)
        
        # Final summary
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        print(f"\nToken Recovery: {token_accuracy:.1f}%\n")
        print("Message Recovery:")
        print(f"{'Agent':<10} {'From Original':<15} {'From Recovered':<15} {'Degradation':<15}")
        print("-" * 60)
        
        for agent in agent_freqs.keys():
            orig_acc = orig_results[agent]['accuracy']
            rec_acc = rec_results[agent]['accuracy']
            degradation = orig_acc - rec_acc
            
            print(f"{agent:<10} {orig_acc:>6.1f}%         {rec_acc:>6.1f}%         {degradation:>+6.1f}%")
        
        avg_orig = np.mean([r['accuracy'] for r in orig_results.values()])
        avg_rec = np.mean([r['accuracy'] for r in rec_results.values()])
        avg_deg = avg_orig - avg_rec
        
        print("-" * 60)
        print(f"{'AVERAGE':<10} {avg_orig:>6.1f}%         {avg_rec:>6.1f}%         {avg_deg:>+6.1f}%")
        
        print("\n" + "="*80)
        print("CONCLUSION")
        print("="*80)
        
        if avg_rec >= 80:
            print("✓✓ EXCELLENT: Message recovery >80%!")
            print("   The probability-based demodulation works great!")
        elif avg_rec >= 70:
            print("✓ GOOD: Message recovery 70-80%")
            print("  The approach works, might benefit from tuning")
        elif avg_rec >= 60:
            print("⚠️ MODERATE: Message recovery 60-70%")
            print("   May need stronger bias or better demodulation parameters")
        else:
            print("✗ POOR: Message recovery <60%")
            print("  Need to investigate bias strength or carrier frequencies")
        
        if abs(avg_deg) < 10:
            print("\n✓ Information survives iMEC encode/decode well!")
        else:
            print(f"\n⚠️ {abs(avg_deg):.1f}% degradation through iMEC pipeline")
        
        return orig_results, rec_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    demod = ProperFrequencyDemodulation()
    orig_results, rec_results = demod.run_complete_test()
