import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class ActualMessageRecovery:
    """
    Actually recover the messages and visualize the demodulation process!
    """
    
    def __init__(self):
        print("Loading GPT-2 for probability analysis...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.MAX_CONTEXT = 1024
        
        # Load data
        with open('hybrid_freq_imec_data.pkl', 'rb') as f:
            self.data = pickle.load(f)
        
        with open('recovered_ciphertext.txt', 'r') as f:
            ciphertext = f.read().strip()
        
        # Recover FM tokens
        self.recovered_tokens = self.recover_fm_tokens(
            ciphertext,
            self.data['metadata']['encryption_key'],
            self.data['metadata']['bits_per_token']
        )
        
        self.original_tokens = self.data['freq_tokens']
        self.context = self.data['context']
        self.agent_freqs = self.data['metadata']['agent_frequencies']
        self.messages = self.data['messages']
        
        print(f"Token recovery: {sum(1 for o, r in zip(self.original_tokens, self.recovered_tokens) if o == r) / len(self.original_tokens) * 100:.1f}%")
    
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
                if 0 <= token_id < 50257:
                    recovered_tokens.append(token_id)
        
        return recovered_tokens
    
    def extract_probability_deviation(self, tokens):
        """Extract probability deviation signal"""
        print("  Extracting probability deviation...")
        input_ids = self.tokenizer.encode(self.context, return_tensors='pt').to(self.device)
        eos_token_id = self.tokenizer.eos_token_id
        
        log_probs = []
        
        with torch.no_grad():
            for i, token in enumerate(tokens):
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]
                logits[eos_token_id] = -float('inf')
                log_probs_dist = torch.log_softmax(logits, dim=0).cpu().numpy()
                log_probs.append(log_probs_dist[token])
                
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[token]], device=self.device, dtype=torch.long)
                ], dim=1)
                
                if input_ids.shape[1] > self.MAX_CONTEXT:
                    input_ids = input_ids[:, -self.MAX_CONTEXT:]
        
        deviation_signal = -np.array(log_probs)
        deviation_signal = (deviation_signal - np.mean(deviation_signal)) / (np.std(deviation_signal) + 1e-10)
        
        return deviation_signal
    
    def demodulate_carrier(self, deviation_signal, carrier_freq, n_bits, bandwidth=0.012):
        """
        Demodulate a single carrier to recover bits
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
    
    def visualize_demodulation(self):
        """
        Create comprehensive demodulation visualization
        """
        print("\n" + "="*80)
        print("COMPLETE DEMODULATION PIPELINE")
        print("="*80)
        
        # Extract deviation signals
        print("\nOriginal tokens:")
        orig_deviation = self.extract_probability_deviation(self.original_tokens)
        print("Recovered tokens:")
        rec_deviation = self.extract_probability_deviation(self.recovered_tokens)
        
        # Create figure
        fig, axes = plt.subplots(3, 5, figsize=(22, 12))
        
        colors = {'ALICE': '#1f77b4', 'BOB': '#ff7f0e', 'CHARLIE': '#2ca02c'}
        
        print("\n" + "="*80)
        print("MESSAGE RECOVERY RESULTS")
        print("="*80)
        
        for idx, (agent, carrier_freq) in enumerate(sorted(self.agent_freqs.items(), key=lambda x: x[1])):
            color = colors[agent]
            original_bits = self.messages[agent]
            n_bits = len(original_bits)
            
            # Demodulate original
            orig_decoded, orig_envelope, orig_filtered, orig_amps, orig_thresh = \
                self.demodulate_carrier(orig_deviation, carrier_freq, n_bits)
            
            # Demodulate recovered
            rec_decoded, rec_envelope, rec_filtered, rec_amps, rec_thresh = \
                self.demodulate_carrier(rec_deviation, carrier_freq, n_bits)
            
            # Calculate accuracy
            orig_accuracy = sum(1 for a, b in zip(original_bits, orig_decoded) if a == b) / n_bits * 100
            rec_accuracy = sum(1 for a, b in zip(original_bits, rec_decoded) if a == b) / n_bits * 100
            
            print(f"\n{agent} ({carrier_freq:.3f} Hz):")
            print(f"  Original:  {original_bits}")
            print(f"  From orig: {orig_decoded} → {orig_accuracy:.1f}% accuracy")
            print(f"  From rec:  {rec_decoded} → {rec_accuracy:.1f}% accuracy")
            
            # Plot 1: Filtered signals in time domain
            axes[idx, 0].plot(orig_filtered, linewidth=1.5, alpha=0.7, color='blue', label='Original')
            axes[idx, 0].plot(rec_filtered, linewidth=1.5, alpha=0.7, color='orange', linestyle='--', label='Recovered')
            axes[idx, 0].set_title(f'{agent}: Bandpass Filtered Signal', fontsize=11, fontweight='bold')
            axes[idx, 0].set_xlabel('Token Position', fontsize=9)
            axes[idx, 0].set_ylabel('Amplitude', fontsize=9)
            axes[idx, 0].legend(fontsize=8)
            axes[idx, 0].grid(True, alpha=0.3)
            
            # Plot 2: Envelope detection
            axes[idx, 1].plot(orig_envelope, linewidth=2, alpha=0.7, color='blue', label='Original')
            axes[idx, 1].plot(rec_envelope, linewidth=2, alpha=0.7, color='orange', linestyle='--', label='Recovered')
            axes[idx, 1].axhline(y=orig_thresh, color='blue', linestyle=':', alpha=0.5, linewidth=1)
            axes[idx, 1].axhline(y=rec_thresh, color='orange', linestyle=':', alpha=0.5, linewidth=1)
            
            # Mark bit boundaries and original bits
            tokens_per_bit = len(orig_envelope) // n_bits
            for i in range(n_bits + 1):
                axes[idx, 1].axvline(x=i*tokens_per_bit, color='gray', linestyle=':', alpha=0.3)
            for i in range(n_bits):
                mid = (i + 0.5) * tokens_per_bit
                axes[idx, 1].text(mid, 1.1, str(original_bits[i]), ha='center', 
                                fontsize=9, fontweight='bold', color='red')
            
            axes[idx, 1].set_title(f'{agent}: Hilbert Envelope', fontsize=11, fontweight='bold')
            axes[idx, 1].set_xlabel('Token Position', fontsize=9)
            axes[idx, 1].set_ylabel('Envelope', fontsize=9)
            axes[idx, 1].legend(fontsize=8)
            axes[idx, 1].grid(True, alpha=0.3)
            axes[idx, 1].set_ylim([-0.1, 1.3])
            
            # Plot 3: Bit amplitudes comparison
            x_pos = np.arange(n_bits)
            width = 0.35
            
            axes[idx, 2].bar(x_pos - width/2, orig_amps, width, label='Original', alpha=0.7, color='blue')
            axes[idx, 2].bar(x_pos + width/2, rec_amps, width, label='Recovered', alpha=0.7, color='orange')
            axes[idx, 2].axhline(y=orig_thresh, color='blue', linestyle='--', alpha=0.5, linewidth=1)
            axes[idx, 2].axhline(y=rec_thresh, color='orange', linestyle='--', alpha=0.5, linewidth=1)
            
            # Mark original bits
            for i in range(n_bits):
                axes[idx, 2].text(i, 1.15, str(original_bits[i]), ha='center', 
                                fontsize=9, fontweight='bold', color='red')
            
            axes[idx, 2].set_title(f'{agent}: Bit Amplitudes', fontsize=11, fontweight='bold')
            axes[idx, 2].set_xlabel('Bit Position', fontsize=9)
            axes[idx, 2].set_ylabel('Amplitude', fontsize=9)
            axes[idx, 2].set_xticks(x_pos)
            axes[idx, 2].legend(fontsize=8)
            axes[idx, 2].grid(True, alpha=0.3, axis='y')
            axes[idx, 2].set_ylim([0, 1.3])
            
            # Plot 4: Bit comparison
            axes[idx, 3].axis('off')
            
            # Color code based on accuracy
            orig_color = 'lightgreen' if orig_accuracy >= 70 else 'lightyellow' if orig_accuracy >= 50 else 'lightcoral'
            rec_color = 'lightgreen' if rec_accuracy >= 70 else 'lightyellow' if rec_accuracy >= 50 else 'lightcoral'
            
            comparison_text = f"""
{agent} @ {carrier_freq:.3f} Hz

ORIGINAL MESSAGE:
{original_bits}

FROM ORIGINAL TOKENS:
{orig_decoded}
Accuracy: {orig_accuracy:.1f}%
Threshold: {orig_thresh:.3f}

FROM RECOVERED TOKENS:
{rec_decoded}
Accuracy: {rec_accuracy:.1f}%
Threshold: {rec_thresh:.3f}

Degradation: {orig_accuracy - rec_accuracy:+.1f}%
"""
            
            axes[idx, 3].text(0.05, 0.5, comparison_text, fontsize=9, family='monospace',
                            va='center', bbox=dict(boxstyle='round', 
                            facecolor=rec_color, alpha=0.5))
            
            # Plot 5: Bit-by-bit comparison
            bit_comparison = []
            for i in range(n_bits):
                if original_bits[i] == orig_decoded[i] == rec_decoded[i]:
                    bit_comparison.append(2)  # Both correct
                elif original_bits[i] == orig_decoded[i]:
                    bit_comparison.append(1)  # Only original correct
                elif original_bits[i] == rec_decoded[i]:
                    bit_comparison.append(0.5)  # Only recovered correct
                else:
                    bit_comparison.append(0)  # Both wrong
            
            axes[idx, 4].bar(range(n_bits), bit_comparison, color=['red' if bc == 0 else 'orange' if bc == 0.5 else 'yellow' if bc == 1 else 'green' for bc in bit_comparison])
            axes[idx, 4].set_title(f'{agent}: Bit Recovery Status', fontsize=11, fontweight='bold')
            axes[idx, 4].set_xlabel('Bit Position', fontsize=9)
            axes[idx, 4].set_ylabel('Status', fontsize=9)
            axes[idx, 4].set_yticks([0, 1, 2])
            axes[idx, 4].set_yticklabels(['Both Wrong', 'Orig Only', 'Both Correct'], fontsize=8)
            axes[idx, 4].set_ylim([0, 2.2])
            axes[idx, 4].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Complete Demodulation: From Probability Deviation to Message Recovery', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig('complete_message_recovery.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: complete_message_recovery.png")
        
        plt.show()


if __name__ == "__main__":
    analyzer = ActualMessageRecovery()
    analyzer.visualize_demodulation()
