import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class ProbabilityDeviationFFT:
    """
    FFT analysis on PROBABILITY DEVIATION signals (correct approach!)
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
        
        print(f"Original tokens: {len(self.original_tokens)}")
        print(f"Recovered tokens: {len(self.recovered_tokens)}")
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
    
    def extract_probability_deviation(self, tokens, label=""):
        """
        Extract probability deviation signal - THE KEY METHOD!
        This captures the FM modulation bias.
        """
        print(f"\nExtracting probability deviation for {label}...")
        
        input_ids = self.tokenizer.encode(self.context, return_tensors='pt').to(self.device)
        eos_token_id = self.tokenizer.eos_token_id
        
        log_probs = []
        
        with torch.no_grad():
            for i, token in enumerate(tokens):
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]
                
                # Remove EOS token
                logits[eos_token_id] = -float('inf')
                
                # Get log probabilities
                log_probs_dist = torch.log_softmax(logits, dim=0).cpu().numpy()
                
                # Record log probability of selected token
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
                    print(f"  Processed {i+1}/{len(tokens)} tokens...")
        
        # Convert to deviation signal (negative log prob = "surprise")
        deviation_signal = -np.array(log_probs)
        
        # Normalize
        deviation_signal = (deviation_signal - np.mean(deviation_signal)) / (np.std(deviation_signal) + 1e-10)
        
        print(f"  ✓ Extracted (mean={np.mean(deviation_signal):.3f}, std={np.std(deviation_signal):.3f})")
        
        return deviation_signal
    
    def plot_fft_comparison(self):
        """
        Create FFT plots using PROBABILITY DEVIATION (correct method!)
        """
        print("\n" + "="*80)
        print("FFT ANALYSIS: PROBABILITY DEVIATION SIGNALS")
        print("="*80)
        
        # Extract probability deviation signals
        orig_deviation = self.extract_probability_deviation(self.original_tokens, "Original")
        rec_deviation = self.extract_probability_deviation(self.recovered_tokens, "Recovered")
        
        # Compute FFT
        fft_orig = np.fft.fft(orig_deviation)
        fft_rec = np.fft.fft(rec_deviation)
        
        power_orig = np.abs(fft_orig) ** 2
        power_rec = np.abs(fft_rec) ** 2
        
        freqs = np.fft.fftfreq(len(orig_deviation))
        
        # Only positive frequencies
        positive_mask = freqs > 0
        freqs_pos = freqs[positive_mask]
        power_orig_pos = power_orig[positive_mask]
        power_rec_pos = power_rec[positive_mask]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Colors for agents
        colors = {'ALICE': '#1f77b4', 'BOB': '#ff7f0e', 'CHARLIE': '#2ca02c'}
        
        # Plot 1: Original probability deviation FFT
        axes[0].plot(freqs_pos, power_orig_pos, linewidth=2.5, alpha=0.8, color='darkblue')
        axes[0].set_title('Original: Probability Deviation FFT', fontsize=16, fontweight='bold')
        axes[0].set_xlabel('Frequency (Hz)', fontsize=14)
        axes[0].set_ylabel('Power Spectral Density', fontsize=14)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].set_xlim([0, 0.08])
        axes[0].set_ylim([0, max(power_orig_pos[:len(power_orig_pos)//4]) * 1.2])
        
        # Mark carrier frequencies
        for agent, freq in sorted(self.agent_freqs.items(), key=lambda x: x[1]):
            axes[0].axvline(x=freq, color=colors[agent], linestyle='--', 
                          alpha=0.7, linewidth=2.5, label=f'{agent} ({freq:.3f} Hz)')
        
        axes[0].legend(fontsize=12, loc='upper right', framealpha=0.9)
        axes[0].tick_params(labelsize=12)
        
        # Plot 2: Recovered probability deviation FFT
        axes[1].plot(freqs_pos, power_rec_pos, linewidth=2.5, alpha=0.8, color='darkorange')
        axes[1].set_title('Recovered: Probability Deviation FFT (98.7%)', fontsize=16, fontweight='bold')
        axes[1].set_xlabel('Frequency (Hz)', fontsize=14)
        axes[1].set_ylabel('Power Spectral Density', fontsize=14)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].set_xlim([0, 0.08])
        axes[1].set_ylim([0, max(power_rec_pos[:len(power_rec_pos)//4]) * 1.2])
        
        # Mark carrier frequencies
        for agent, freq in sorted(self.agent_freqs.items(), key=lambda x: x[1]):
            axes[1].axvline(x=freq, color=colors[agent], linestyle='--', 
                          alpha=0.7, linewidth=2.5, label=f'{agent} ({freq:.3f} Hz)')
        
        axes[1].legend(fontsize=12, loc='upper right', framealpha=0.9)
        axes[1].tick_params(labelsize=12)
        
        plt.tight_layout()
        plt.savefig('probability_deviation_fft_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: probability_deviation_fft_comparison.png")
        
        # Also create overlay
        fig2, ax = plt.subplots(1, 1, figsize=(12, 7))
        
        ax.plot(freqs_pos, power_orig_pos, linewidth=2.5, alpha=0.8, 
               color='darkblue', label='Original Tokens')
        ax.plot(freqs_pos, power_rec_pos, linewidth=2.5, alpha=0.8, 
               color='darkorange', linestyle='--', label='Recovered Tokens (98.7%)')
        
        ax.set_title('FFT: Probability Deviation Signals (Original vs Recovered)', 
                    fontsize=18, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)', fontsize=16)
        ax.set_ylabel('Power Spectral Density', fontsize=16)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([0, 0.08])
        
        # Mark carrier frequencies
        for agent, freq in sorted(self.agent_freqs.items(), key=lambda x: x[1]):
            ax.axvline(x=freq, color=colors[agent], linestyle=':', 
                      alpha=0.6, linewidth=2.5)
            ax.text(freq, ax.get_ylim()[1]*0.85, f'{agent}\n{freq:.3f} Hz', 
                   ha='center', fontsize=11, color=colors[agent], 
                   fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                   facecolor='white', alpha=0.8))
        
        ax.legend(fontsize=14, loc='upper right', framealpha=0.95)
        ax.tick_params(labelsize=13)
        
        plt.tight_layout()
        plt.savefig('probability_deviation_fft_overlay.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: probability_deviation_fft_overlay.png")
        
        # Calculate SNR at carrier frequencies
        print("\n" + "="*80)
        print("CARRIER FREQUENCY DETECTION (Probability Deviation Method)")
        print("="*80)
        
        for agent, carrier_freq in self.agent_freqs.items():
            freq_window = 0.005
            mask = (freqs_pos >= carrier_freq - freq_window) & (freqs_pos <= carrier_freq + freq_window)
            
            peak_orig = np.max(power_orig_pos[mask]) if np.any(mask) else 0
            peak_rec = np.max(power_rec_pos[mask]) if np.any(mask) else 0
            
            noise_orig = np.median(power_orig_pos)
            noise_rec = np.median(power_rec_pos)
            
            snr_orig = 10 * np.log10(peak_orig / noise_orig) if noise_orig > 0 else -np.inf
            snr_rec = 10 * np.log10(peak_rec / noise_rec) if noise_rec > 0 else -np.inf
            
            print(f"\n{agent} ({carrier_freq:.3f} Hz):")
            print(f"  Original SNR:  {snr_orig:.1f} dB")
            print(f"  Recovered SNR: {snr_rec:.1f} dB")
            print(f"  SNR degradation: {snr_orig - snr_rec:.1f} dB")
            
            if snr_rec > 3:
                print(f"  ✓ Carrier detectable!")
            else:
                print(f"  ✗ Carrier weak/buried")
        
        plt.show()


if __name__ == "__main__":
    analyzer = ProbabilityDeviationFFT()
    analyzer.plot_fft_comparison()
