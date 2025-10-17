import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fft
import pickle

class RecoveredTokenFFTAnalysis:
    """
    Analyze FFT spectrum of recovered tokens to visualize carrier frequencies
    """
    
    def __init__(self):
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
                if 0 <= token_id < 50257:  # GPT-2 vocab size
                    recovered_tokens.append(token_id)
        
        return recovered_tokens
    
    def analyze_fft_spectrum(self):
        """
        Compare FFT spectra of original vs recovered tokens
        """
        print("\n" + "="*80)
        print("FFT SPECTRUM ANALYSIS: ORIGINAL vs RECOVERED TOKENS")
        print("="*80)
        
        # Normalize token sequences
        orig_norm = (np.array(self.original_tokens) - np.mean(self.original_tokens)) / np.std(self.original_tokens)
        rec_norm = (np.array(self.recovered_tokens) - np.mean(self.recovered_tokens)) / np.std(self.recovered_tokens)
        
        # Compute FFT
        fft_orig = np.fft.fft(orig_norm)
        fft_rec = np.fft.fft(rec_norm)
        
        power_orig = np.abs(fft_orig) ** 2
        power_rec = np.abs(fft_rec) ** 2
        
        freqs = np.fft.fftfreq(len(orig_norm))
        
        # Only positive frequencies
        positive_mask = freqs > 0
        freqs_pos = freqs[positive_mask]
        power_orig_pos = power_orig[positive_mask]
        power_rec_pos = power_rec[positive_mask]
        
        # Create visualization
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Plot 1: Original tokens FFT
        axes[0].semilogy(freqs_pos, power_orig_pos, linewidth=2, alpha=0.7, color='blue')
        axes[0].set_title('FFT Spectrum: Original FM-Modulated Tokens', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Frequency (Hz)', fontsize=12)
        axes[0].set_ylabel('Power (log scale)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([0, 0.1])
        
        # Mark carrier frequencies
        for agent, freq in self.agent_freqs.items():
            axes[0].axvline(x=freq, color='red', linestyle='--', alpha=0.6, linewidth=2)
            axes[0].text(freq, axes[0].get_ylim()[1]*0.5, f'{agent}\n{freq:.3f} Hz', 
                        ha='center', fontsize=10, color='red', fontweight='bold')
        
        # Plot 2: Recovered tokens FFT
        axes[1].semilogy(freqs_pos, power_rec_pos, linewidth=2, alpha=0.7, color='orange')
        axes[1].set_title('FFT Spectrum: Recovered Tokens (After iMEC Decode)', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Frequency (Hz)', fontsize=12)
        axes[1].set_ylabel('Power (log scale)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([0, 0.1])
        
        # Mark carrier frequencies
        for agent, freq in self.agent_freqs.items():
            axes[1].axvline(x=freq, color='red', linestyle='--', alpha=0.6, linewidth=2)
            axes[1].text(freq, axes[1].get_ylim()[1]*0.5, f'{agent}\n{freq:.3f} Hz', 
                        ha='center', fontsize=10, color='red', fontweight='bold')
        
        # Plot 3: Overlay comparison
        axes[2].semilogy(freqs_pos, power_orig_pos, linewidth=2, alpha=0.7, 
                        color='blue', label='Original')
        axes[2].semilogy(freqs_pos, power_rec_pos, linewidth=2, alpha=0.7, 
                        color='orange', label='Recovered', linestyle='--')
        axes[2].set_title('FFT Spectrum: Original vs Recovered (Overlay)', 
                         fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Frequency (Hz)', fontsize=12)
        axes[2].set_ylabel('Power (log scale)', fontsize=12)
        axes[2].legend(fontsize=12, loc='upper right')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim([0, 0.1])
        
        # Mark carrier frequencies
        for agent, freq in self.agent_freqs.items():
            axes[2].axvline(x=freq, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
        
        plt.tight_layout()
        plt.savefig('recovered_tokens_fft_spectrum.png', dpi=150, bbox_inches='tight')
        print("\n✓ Saved: recovered_tokens_fft_spectrum.png")
        
        # Calculate peak detection
        print("\n" + "="*80)
        print("CARRIER FREQUENCY DETECTION")
        print("="*80)
        
        for agent, carrier_freq in self.agent_freqs.items():
            # Find power near carrier
            freq_window = 0.005
            mask = (freqs_pos >= carrier_freq - freq_window) & (freqs_pos <= carrier_freq + freq_window)
            
            peak_orig = np.max(power_orig_pos[mask])
            peak_rec = np.max(power_rec_pos[mask])
            
            # Background noise
            noise_orig = np.median(power_orig_pos)
            noise_rec = np.median(power_rec_pos)
            
            snr_orig = 10 * np.log10(peak_orig / noise_orig)
            snr_rec = 10 * np.log10(peak_rec / noise_rec)
            
            print(f"\n{agent} ({carrier_freq:.3f} Hz):")
            print(f"  Original SNR:  {snr_orig:.1f} dB")
            print(f"  Recovered SNR: {snr_rec:.1f} dB")
            print(f"  SNR degradation: {snr_orig - snr_rec:.1f} dB")
            
            if snr_rec > 3:
                print(f"  ✓ Carrier still detectable!")
            else:
                print(f"  ✗ Carrier buried in noise")
        
        plt.show()


if __name__ == "__main__":
    analyzer = RecoveredTokenFFTAnalysis()
    analyzer.analyze_fft_spectrum()
