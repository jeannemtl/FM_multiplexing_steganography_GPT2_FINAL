import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class BandpassFilteredFFT:
    """
    Show FFT AFTER bandpass filtering - carriers should be visible!
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
    
    def bandpass_filter(self, signal_data, carrier_freq, bandwidth=0.012):
        """Apply bandpass filter in frequency domain"""
        fft_result = np.fft.fft(signal_data)
        freqs = np.fft.fftfreq(len(signal_data))
        
        # Bandpass mask
        mask = np.abs(freqs - carrier_freq) < bandwidth
        filtered_fft = fft_result * mask
        filtered_signal = np.fft.ifft(filtered_fft).real
        
        return filtered_signal, filtered_fft, freqs
    
    def plot_bandpass_filtered_fft(self):
        """
        Show FFT BEFORE and AFTER bandpass filtering for each carrier
        """
        print("\nExtracting probability deviation signals...")
        orig_deviation = self.extract_probability_deviation(self.original_tokens)
        rec_deviation = self.extract_probability_deviation(self.recovered_tokens)
        
        # Create figure with 3 rows (one per agent) and 4 columns
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        
        colors = {'ALICE': '#1f77b4', 'BOB': '#ff7f0e', 'CHARLIE': '#2ca02c'}
        
        for idx, (agent, carrier_freq) in enumerate(sorted(self.agent_freqs.items(), key=lambda x: x[1])):
            color = colors[agent]
            
            # Original signal - before filtering
            fft_orig_full = np.fft.fft(orig_deviation)
            freqs_full = np.fft.fftfreq(len(orig_deviation))
            power_orig_full = np.abs(fft_orig_full) ** 2
            
            pos_mask = freqs_full > 0
            freqs_pos = freqs_full[pos_mask]
            power_orig_pos = power_orig_full[pos_mask]
            
            # Plot 1: Full spectrum (original)
            axes[idx, 0].plot(freqs_pos, power_orig_pos, linewidth=2, alpha=0.8, color='darkblue')
            axes[idx, 0].axvline(x=carrier_freq, color=color, linestyle='--', alpha=0.7, linewidth=2)
            axes[idx, 0].set_title(f'{agent}: Full Spectrum (Original)', fontsize=12, fontweight='bold')
            axes[idx, 0].set_xlabel('Frequency (Hz)', fontsize=10)
            axes[idx, 0].set_ylabel('Power', fontsize=10)
            axes[idx, 0].set_xlim([0, 0.08])
            axes[idx, 0].grid(True, alpha=0.3)
            
            # Bandpass filter original
            filtered_orig, filtered_fft_orig, freqs_filt = self.bandpass_filter(orig_deviation, carrier_freq)
            power_filtered_orig = np.abs(filtered_fft_orig) ** 2
            power_filtered_orig_pos = power_filtered_orig[pos_mask]
            
            # Plot 2: After bandpass (original)
            axes[idx, 1].plot(freqs_pos, power_filtered_orig_pos, linewidth=2, alpha=0.8, color=color)
            axes[idx, 1].axvline(x=carrier_freq, color='red', linestyle='--', alpha=0.7, linewidth=2)
            axes[idx, 1].set_title(f'{agent}: After Bandpass ({carrier_freq:.3f} Hz)', fontsize=12, fontweight='bold')
            axes[idx, 1].set_xlabel('Frequency (Hz)', fontsize=10)
            axes[idx, 1].set_ylabel('Power', fontsize=10)
            axes[idx, 1].set_xlim([carrier_freq - 0.02, carrier_freq + 0.02])
            axes[idx, 1].grid(True, alpha=0.3)
            
            # Recovered signal - before filtering
            fft_rec_full = np.fft.fft(rec_deviation)
            power_rec_full = np.abs(fft_rec_full) ** 2
            power_rec_pos = power_rec_full[pos_mask]
            
            # Plot 3: Full spectrum (recovered)
            axes[idx, 2].plot(freqs_pos, power_rec_pos, linewidth=2, alpha=0.8, color='darkorange')
            axes[idx, 2].axvline(x=carrier_freq, color=color, linestyle='--', alpha=0.7, linewidth=2)
            axes[idx, 2].set_title(f'{agent}: Full Spectrum (Recovered)', fontsize=12, fontweight='bold')
            axes[idx, 2].set_xlabel('Frequency (Hz)', fontsize=10)
            axes[idx, 2].set_ylabel('Power', fontsize=10)
            axes[idx, 2].set_xlim([0, 0.08])
            axes[idx, 2].grid(True, alpha=0.3)
            
            # Bandpass filter recovered
            filtered_rec, filtered_fft_rec, _ = self.bandpass_filter(rec_deviation, carrier_freq)
            power_filtered_rec = np.abs(filtered_fft_rec) ** 2
            power_filtered_rec_pos = power_filtered_rec[pos_mask]
            
            # Plot 4: After bandpass (recovered)
            axes[idx, 3].plot(freqs_pos, power_filtered_rec_pos, linewidth=2, alpha=0.8, color=color, linestyle='--')
            axes[idx, 3].axvline(x=carrier_freq, color='red', linestyle='--', alpha=0.7, linewidth=2)
            axes[idx, 3].set_title(f'{agent}: After Bandpass (Recovered)', fontsize=12, fontweight='bold')
            axes[idx, 3].set_xlabel('Frequency (Hz)', fontsize=10)
            axes[idx, 3].set_ylabel('Power', fontsize=10)
            axes[idx, 3].set_xlim([carrier_freq - 0.02, carrier_freq + 0.02])
            axes[idx, 3].grid(True, alpha=0.3)
            
            # Calculate SNR improvements
            noise_full_orig = np.median(power_orig_pos)
            noise_full_rec = np.median(power_rec_pos)
            
            # Find power near carrier in filtered signal
            carrier_mask = (freqs_pos >= carrier_freq - 0.005) & (freqs_pos <= carrier_freq + 0.005)
            peak_filtered_orig = np.max(power_filtered_orig_pos[carrier_mask]) if np.any(carrier_mask) else 0
            peak_filtered_rec = np.max(power_filtered_rec_pos[carrier_mask]) if np.any(carrier_mask) else 0
            
            noise_filtered_orig = np.median(power_filtered_orig_pos[power_filtered_orig_pos > 0])
            noise_filtered_rec = np.median(power_filtered_rec_pos[power_filtered_rec_pos > 0])
            
            snr_before_orig = 10 * np.log10(peak_filtered_orig / noise_full_orig) if noise_full_orig > 0 else -np.inf
            snr_after_orig = 10 * np.log10(peak_filtered_orig / noise_filtered_orig) if noise_filtered_orig > 0 else -np.inf
            
            snr_before_rec = 10 * np.log10(peak_filtered_rec / noise_full_rec) if noise_full_rec > 0 else -np.inf
            snr_after_rec = 10 * np.log10(peak_filtered_rec / noise_filtered_rec) if noise_filtered_rec > 0 else -np.inf
            
            print(f"\n{agent} ({carrier_freq:.3f} Hz):")
            print(f"  Original - SNR before filter: {snr_before_orig:.1f} dB")
            print(f"  Original - SNR after filter:  {snr_after_orig:.1f} dB (improvement: {snr_after_orig - snr_before_orig:.1f} dB)")
            print(f"  Recovered - SNR before filter: {snr_before_rec:.1f} dB")
            print(f"  Recovered - SNR after filter:  {snr_after_rec:.1f} dB (improvement: {snr_after_rec - snr_before_rec:.1f} dB)")
        
        plt.suptitle('FFT Analysis: Before and After Bandpass Filtering', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig('bandpass_filtered_fft_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: bandpass_filtered_fft_analysis.png")
        
        plt.show()


if __name__ == "__main__":
    analyzer = BandpassFilteredFFT()
    analyzer.plot_bandpass_filtered_fft()
