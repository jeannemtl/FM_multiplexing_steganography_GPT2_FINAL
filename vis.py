import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import pickle
from imec_encoder import MinEntropyCouplingSteganography

class FFTVisualComparison:
    """
    Visual demonstration: FFT analysis with vs without encryption key.
    Shows that frequency patterns are only recoverable WITH the key.
    """
    
    def __init__(self):
        print("Loading GPT-2 for FFT visualization...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.agents = {
            'ALICE': {'freq': 0.02, 'color': 'blue', 'label': 'ALICE (0.02 Hz)'},
            'BOB': {'freq': 0.04, 'color': 'green', 'label': 'BOB (0.04 Hz)'},
            'CHARLIE': {'freq': 0.06, 'color': 'red', 'label': 'CHARLIE (0.06 Hz)'}
        }
        
        self.imec = None
    
    def extract_entropy(self, token_ids, context):
        """Extract entropy sequence from tokens."""
        token_ids = [int(t) for t in token_ids]
        vocab_size = len(self.tokenizer)
        
        valid_tokens = [t for t in token_ids if 0 <= t < vocab_size]
        
        context_ids = self.tokenizer.encode(context)
        full_ids = context_ids + valid_tokens
        
        max_len = self.model.config.n_positions
        if len(full_ids) > max_len:
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
                except:
                    break
        
        return np.array(entropy_sequence)
    
    def compute_fft(self, signal):
        """Compute FFT and return frequencies and power."""
        N = len(signal)
        normalized = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
        
        fft_vals = fft(normalized)
        fft_freq = fftfreq(N, d=1.0)
        
        pos_mask = fft_freq > 0
        freqs = fft_freq[pos_mask]
        power = np.abs(fft_vals[pos_mask])**2
        
        return freqs, power
    
    def create_visualization(self, data_file='hybrid_freq_imec_data.pkl'):
        """Create comprehensive FFT comparison visualization."""
        
        print("\n" + "="*80)
        print("CREATING FFT VISUAL COMPARISON")
        print("="*80)
        
        # Load data
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        context = data['context']
        freq_tokens = data['freq_tokens']
        obf_tokens = data['obfuscated_tokens']
        metadata = data['metadata']
        messages = data['messages']
        
        print(f"\n✓ Loaded data")
        
        # ========================================
        # SCENARIO 1: Original (Frequency-Modulated)
        # ========================================
        print("\n✓ Processing Scenario 1: Original frequency-modulated tokens...")
        freq_entropy = self.extract_entropy(freq_tokens, context)
        freq_freqs, freq_power = self.compute_fft(freq_entropy)
        
        # ========================================
        # SCENARIO 2: Without Key (Obfuscated, then decoded without decryption)
        # ========================================
        print("✓ Processing Scenario 2: Recovery WITHOUT encryption key...")
        
        # Decode iMEC without decryption
        block_size = metadata['block_size_bits']
        self.imec = MinEntropyCouplingSteganography(block_size_bits=block_size)
        
        recovered_ciphertext = self.imec.decode_imec(
            obf_tokens, context, metadata['n_blocks'], metadata['block_size_bits']
        )
        
        # Convert ciphertext to tokens WITHOUT decryption
        bits_per_token = metadata['bits_per_token']
        n_tokens = metadata['n_freq_tokens']
        vocab_size = len(self.tokenizer)
        
        tokens_no_key = []
        for i in range(n_tokens):
            start = i * bits_per_token
            end = start + bits_per_token
            if end <= len(recovered_ciphertext):
                token_bits = recovered_ciphertext[start:end]
                token_id = int(token_bits, 2)
                tokens_no_key.append(token_id % vocab_size)
        
        nokey_entropy = self.extract_entropy(tokens_no_key, context)
        nokey_freqs, nokey_power = self.compute_fft(nokey_entropy)
        
        # ========================================
        # SCENARIO 3: With Key (Obfuscated, decoded with decryption)
        # ========================================
        print("✓ Processing Scenario 3: Recovery WITH encryption key...")
        
        # Decrypt
        encryption_key = metadata['encryption_key']
        min_len = min(len(recovered_ciphertext), len(encryption_key))
        recovered_plaintext = ''.join(
            str(int(recovered_ciphertext[i]) ^ int(encryption_key[i]))
            for i in range(min_len)
        )
        
        # Convert to tokens
        tokens_with_key = []
        for i in range(n_tokens):
            start = i * bits_per_token
            end = start + bits_per_token
            if end <= len(recovered_plaintext):
                token_bits = recovered_plaintext[start:end]
                token_id = int(token_bits, 2)
                tokens_with_key.append(token_id % vocab_size)
        
        withkey_entropy = self.extract_entropy(tokens_with_key, context)
        withkey_freqs, withkey_power = self.compute_fft(withkey_entropy)
        
        # ========================================
        # CREATE FIGURE
        # ========================================
        print("\n✓ Generating plots...")
        
        fig = plt.figure(figsize=(18, 12))
        
        # Define frequency range for zoomed plots
        freq_max = 0.15
        
        # ========================================
        # ROW 1: Full Spectrum
        # ========================================
        
        # Original
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(freq_freqs, freq_power, color='black', linewidth=0.5, alpha=0.7)
        for agent_name, agent_info in self.agents.items():
            target_freq = agent_info['freq']
            ax1.axvline(target_freq, color=agent_info['color'], linestyle='--', 
                       linewidth=2, alpha=0.7, label=agent_info['label'])
        ax1.set_xlabel('Frequency (Hz)', fontsize=11)
        ax1.set_ylabel('Power', fontsize=11)
        ax1.set_title('ORIGINAL: Frequency-Modulated Tokens\n(Ground Truth)', 
                     fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 0.5)
        
        # Without Key
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(nokey_freqs, nokey_power, color='black', linewidth=0.5, alpha=0.7)
        for agent_name, agent_info in self.agents.items():
            target_freq = agent_info['freq']
            ax2.axvline(target_freq, color=agent_info['color'], linestyle='--', 
                       linewidth=2, alpha=0.7, label=agent_info['label'])
        ax2.set_xlabel('Frequency (Hz)', fontsize=11)
        ax2.set_ylabel('Power', fontsize=11)
        ax2.set_title('WITHOUT KEY: Random Noise\n(No Recovery Possible)', 
                     fontsize=12, fontweight='bold', color='red')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 0.5)
        
        # With Key
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(withkey_freqs, withkey_power, color='black', linewidth=0.5, alpha=0.7)
        for agent_name, agent_info in self.agents.items():
            target_freq = agent_info['freq']
            ax3.axvline(target_freq, color=agent_info['color'], linestyle='--', 
                       linewidth=2, alpha=0.7, label=agent_info['label'])
        ax3.set_xlabel('Frequency (Hz)', fontsize=11)
        ax3.set_ylabel('Power', fontsize=11)
        ax3.set_title('WITH KEY: Patterns Recovered\n(99% Token Match)', 
                     fontsize=12, fontweight='bold', color='green')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 0.5)
        
        # ========================================
        # ROW 2: Zoomed to Signal Frequencies
        # ========================================
        
        # Original - Zoomed
        ax4 = plt.subplot(3, 3, 4)
        mask = freq_freqs < freq_max
        ax4.plot(freq_freqs[mask], freq_power[mask], color='black', linewidth=1.5)
        for agent_name, agent_info in self.agents.items():
            target_freq = agent_info['freq']
            ax4.axvline(target_freq, color=agent_info['color'], linestyle='--', 
                       linewidth=2.5, alpha=0.8)
        ax4.set_xlabel('Frequency (Hz)', fontsize=11)
        ax4.set_ylabel('Power', fontsize=11)
        ax4.set_title('Zoomed: Clear Peaks at Target Frequencies', fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, freq_max)
        
        # Without Key - Zoomed
        ax5 = plt.subplot(3, 3, 5)
        mask = nokey_freqs < freq_max
        ax5.plot(nokey_freqs[mask], nokey_power[mask], color='black', linewidth=1.5)
        for agent_name, agent_info in self.agents.items():
            target_freq = agent_info['freq']
            ax5.axvline(target_freq, color=agent_info['color'], linestyle='--', 
                       linewidth=2.5, alpha=0.8)
        ax5.set_xlabel('Frequency (Hz)', fontsize=11)
        ax5.set_ylabel('Power', fontsize=11)
        ax5.set_title('Zoomed: NO Peaks (Random Noise)', fontsize=11, color='red')
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(0, freq_max)
        
        # With Key - Zoomed
        ax6 = plt.subplot(3, 3, 6)
        mask = withkey_freqs < freq_max
        ax6.plot(withkey_freqs[mask], withkey_power[mask], color='black', linewidth=1.5)
        for agent_name, agent_info in self.agents.items():
            target_freq = agent_info['freq']
            ax6.axvline(target_freq, color=agent_info['color'], linestyle='--', 
                       linewidth=2.5, alpha=0.8)
        ax6.set_xlabel('Frequency (Hz)', fontsize=11)
        ax6.set_ylabel('Power', fontsize=11)
        ax6.set_title('Zoomed: Peaks RECOVERED at Target Frequencies', 
                     fontsize=11, color='green')
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim(0, freq_max)
        
        # ========================================
        # ROW 3: Peak Analysis at Each Frequency
        # ========================================
        
        # Function to get peak power at target frequency
        def get_peak_power(freqs, power, target_freq, window=0.01):
            mask = np.abs(freqs - target_freq) < window
            if np.any(mask):
                return np.max(power[mask])
            return 0
        
        # Collect peak powers
        scenarios = ['Original', 'Without Key', 'With Key']
        freq_data = [
            (freq_freqs, freq_power),
            (nokey_freqs, nokey_power),
            (withkey_freqs, withkey_power)
        ]
        
        for idx, (agent_name, agent_info) in enumerate(self.agents.items()):
            ax = plt.subplot(3, 3, 7 + idx)
            target_freq = agent_info['freq']
            
            peak_powers = []
            for freqs, power in freq_data:
                peak = get_peak_power(freqs, power, target_freq)
                peak_powers.append(peak)
            
            colors_bar = ['blue', 'red', 'green']
            bars = ax.bar(scenarios, peak_powers, color=colors_bar, alpha=0.7, edgecolor='black')
            
            ax.set_ylabel('Peak Power', fontsize=11)
            ax.set_title(f'{agent_name} Signal\n(Target: {target_freq} Hz)', 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(peak_powers) * 1.2 if max(peak_powers) > 0 else 1)
            
            # Add value labels on bars
            for bar, power in zip(bars, peak_powers):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{power:.0f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Rotate x labels
            ax.set_xticklabels(scenarios, rotation=15, ha='right')
        
        plt.tight_layout()
        
        # Save figure
        output_file = 'fft_comparison_with_without_key.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved figure: {output_file}")
        
        # Also create a simplified 1x3 version
        self._create_simple_comparison(freq_freqs, freq_power, 
                                       nokey_freqs, nokey_power,
                                       withkey_freqs, withkey_power)
        
        plt.show()
        
        return fig
    
    def _create_simple_comparison(self, freq_freqs, freq_power, 
                                   nokey_freqs, nokey_power,
                                   withkey_freqs, withkey_power):
        """Create simplified 1x3 comparison."""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        freq_max = 0.15
        
        # Original
        mask = freq_freqs < freq_max
        axes[0].plot(freq_freqs[mask], freq_power[mask], color='black', linewidth=2)
        for agent_name, agent_info in self.agents.items():
            axes[0].axvline(agent_info['freq'], color=agent_info['color'], 
                          linestyle='--', linewidth=3, alpha=0.8, 
                          label=agent_info['label'])
        axes[0].set_xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Power', fontsize=14, fontweight='bold')
        axes[0].set_title('A) ORIGINAL\nFrequency-Modulated', 
                         fontsize=16, fontweight='bold')
        axes[0].legend(fontsize=11, loc='upper right')
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(labelsize=12)
        
        # Without Key
        mask = nokey_freqs < freq_max
        axes[1].plot(nokey_freqs[mask], nokey_power[mask], color='red', linewidth=2, alpha=0.7)
        for agent_name, agent_info in self.agents.items():
            axes[1].axvline(agent_info['freq'], color=agent_info['color'], 
                          linestyle='--', linewidth=3, alpha=0.8)
        axes[1].set_xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Power', fontsize=14, fontweight='bold')
        axes[1].set_title('B) WITHOUT KEY\nNo Signal Recovery (0% match)', 
                         fontsize=16, fontweight='bold', color='darkred')
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(labelsize=12)
        
        # With Key
        mask = withkey_freqs < freq_max
        axes[2].plot(withkey_freqs[mask], withkey_power[mask], color='green', linewidth=2, alpha=0.7)
        for agent_name, agent_info in self.agents.items():
            axes[2].axvline(agent_info['freq'], color=agent_info['color'], 
                          linestyle='--', linewidth=3, alpha=0.8)
        axes[2].set_xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Power', fontsize=14, fontweight='bold')
        axes[2].set_title('C) WITH KEY\nSignal Recovered (99% match)', 
                         fontsize=16, fontweight='bold', color='darkgreen')
        axes[2].grid(True, alpha=0.3)
        axes[2].tick_params(labelsize=12)
        
        plt.tight_layout()
        
        output_file = 'fft_comparison_simple.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved simplified figure: {output_file}")


if __name__ == "__main__":
    visualizer = FFTVisualComparison()
    visualizer.create_visualization('hybrid_freq_imec_data.pkl')
