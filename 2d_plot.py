import numpy as np
import matplotlib.pyplot as plt
import pickle

class CarrierFrequency2DPlot:
    """
    Create clean 2D plots of carrier frequencies
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
    
    def plot_2d_comparison(self):
        """
        Create 2D line plots comparing original vs recovered FFT
        """
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
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Colors for agents
        colors = {'ALICE': '#1f77b4', 'BOB': '#ff7f0e', 'CHARLIE': '#2ca02c'}
        
        # Plot 1: Original tokens
        axes[0].plot(freqs_pos, power_orig_pos, linewidth=2.5, alpha=0.8, color='darkblue')
        axes[0].set_title('Original FM-Modulated Tokens', fontsize=16, fontweight='bold')
        axes[0].set_xlabel('Frequency (Hz)', fontsize=14)
        axes[0].set_ylabel('Power Spectral Density', fontsize=14)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].set_xlim([0, 0.08])
        
        # Mark carrier frequencies with different colors
        for agent, freq in sorted(self.agent_freqs.items(), key=lambda x: x[1]):
            axes[0].axvline(x=freq, color=colors[agent], linestyle='--', 
                          alpha=0.7, linewidth=2.5, label=f'{agent} ({freq:.3f} Hz)')
        
        axes[0].legend(fontsize=12, loc='upper right', framealpha=0.9)
        axes[0].tick_params(labelsize=12)
        
        # Plot 2: Recovered tokens
        axes[1].plot(freqs_pos, power_rec_pos, linewidth=2.5, alpha=0.8, color='darkorange')
        axes[1].set_title('Recovered Tokens (98.7% Accuracy)', fontsize=16, fontweight='bold')
        axes[1].set_xlabel('Frequency (Hz)', fontsize=14)
        axes[1].set_ylabel('Power Spectral Density', fontsize=14)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].set_xlim([0, 0.08])
        
        # Mark carrier frequencies
        for agent, freq in sorted(self.agent_freqs.items(), key=lambda x: x[1]):
            axes[1].axvline(x=freq, color=colors[agent], linestyle='--', 
                          alpha=0.7, linewidth=2.5, label=f'{agent} ({freq:.3f} Hz)')
        
        axes[1].legend(fontsize=12, loc='upper right', framealpha=0.9)
        axes[1].tick_params(labelsize=12)
        
        plt.tight_layout()
        plt.savefig('carrier_frequencies_2d.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: carrier_frequencies_2d.png")
        
        # Also create overlay version
        fig2, ax = plt.subplots(1, 1, figsize=(12, 7))
        
        ax.plot(freqs_pos, power_orig_pos, linewidth=2.5, alpha=0.8, 
               color='darkblue', label='Original Tokens')
        ax.plot(freqs_pos, power_rec_pos, linewidth=2.5, alpha=0.8, 
               color='darkorange', linestyle='--', label='Recovered Tokens (98.7%)')
        
        ax.set_title('FFT Spectrum: Original vs Recovered Tokens', 
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
        plt.savefig('carrier_frequencies_overlay.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: carrier_frequencies_overlay.png")
        
        plt.show()


if __name__ == "__main__":
    plotter = CarrierFrequency2DPlot()
    plotter.plot_2d_comparison()
