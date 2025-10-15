import numpy as np
import matplotlib.pyplot as plt
from scipy import fft, signal, stats
import torch
import pickle
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class HybridIMECSecurityTest:
    """Test FFT security of your Frequency Multiplexing + iMEC hybrid system"""
    
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.embedding_dim = self.model.config.n_embd
        
    def load_hybrid_data(self, pkl_path='hybrid_freq_imec_data.pkl'):
        """Load your hybrid frequency + iMEC encoded data"""
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print("Loaded hybrid data with keys:", list(data.keys()))
        print("\nData structure:")
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: array shape {value.shape}")
            elif isinstance(value, list):
                print(f"  {key}: list length {len(value)}")
            elif isinstance(value, dict):
                print(f"  {key}: dict with keys {list(value.keys())}")
            else:
                print(f"  {key}: {type(value).__name__}")
        
        return data
    
    def get_embeddings_from_tokens(self, tokens):
        """Convert tokens to embeddings"""
        # Handle string input (needs tokenization first)
        if isinstance(tokens, str):
            tokens = self.tokenizer.encode(tokens, return_tensors='pt')
        elif isinstance(tokens, list):
            tokens = torch.tensor(tokens).unsqueeze(0)
        elif isinstance(tokens, np.ndarray):
            tokens = torch.tensor(tokens).unsqueeze(0)
        
        with torch.no_grad():
            embeddings = self.model.transformer.wte(tokens)
        
        return embeddings.squeeze().numpy()
    
    def analyze_frequency_channels(self, hybrid_data):
        """Analyze the raw frequency-multiplexed bias signals BEFORE iMEC"""
        
        if 'bias_signals' not in hybrid_data:
            print("⚠ No bias_signals found - skipping frequency analysis")
            return None
        
        bias_signals_dict = hybrid_data['bias_signals']
        print(f"\n{'='*80}")
        print("STEP 1: RAW FREQUENCY CHANNELS (Before iMEC Steganography)")
        print(f"{'='*80}")
        
        # Convert dict to array - stack all agent signals
        if isinstance(bias_signals_dict, dict):
            print(f"Found {len(bias_signals_dict)} agents: {list(bias_signals_dict.keys())}")
            bias_signals = np.column_stack([bias_signals_dict[agent] for agent in bias_signals_dict.keys()])
            print(f"Combined bias signals shape: {bias_signals.shape}")
        else:
            bias_signals = bias_signals_dict
            print(f"Bias signals shape: {bias_signals.shape}")
        
        # FFT of each channel
        n_channels = bias_signals.shape[1] if bias_signals.ndim > 1 else 1
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time domain - all channels
        axes[0, 0].plot(bias_signals)
        axes[0, 0].set_title('Frequency Channels (Time Domain)', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Position')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlim([0, min(len(bias_signals), 500)])  # Show first 500 points
        
        # FFT of each channel
        fft_result = np.fft.fft(bias_signals, axis=0)
        power = np.abs(fft_result) ** 2
        freqs = np.fft.fftfreq(len(bias_signals))
        
        half_len = len(freqs) // 2
        for i in range(min(n_channels, 8)):  # Plot up to 8 channels
            if bias_signals.ndim > 1:
                axes[0, 1].semilogy(freqs[:half_len], power[:half_len, i], 
                                   label=f'Ch {i}', alpha=0.7, linewidth=2)
            else:
                axes[0, 1].semilogy(freqs[:half_len], power[:half_len], 
                                   label='Signal', linewidth=2)
                break
        
        axes[0, 1].set_title('FFT: Clear Carrier Frequencies Visible', 
                            fontsize=12, fontweight='bold', color='red')
        axes[0, 1].set_xlabel('Frequency')
        axes[0, 1].set_ylabel('Power (log scale)')
        axes[0, 1].legend(loc='upper right', fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 2D FFT visualization
        if bias_signals.ndim > 1:
            fft_2d = np.fft.fft2(bias_signals)
            power_2d = np.abs(fft_2d) ** 2
            im = axes[1, 0].imshow(np.log10(power_2d + 1), aspect='auto', 
                                  cmap='hot', interpolation='nearest')
            axes[1, 0].set_title('2D FFT Heatmap', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Channel')
            axes[1, 0].set_ylabel('Frequency Bin')
            plt.colorbar(im, ax=axes[1, 0], label='Log Power')
        else:
            axes[1, 0].text(0.5, 0.5, 'Single channel\n(no 2D view)', 
                           ha='center', va='center', fontsize=12)
            axes[1, 0].axis('off')
        
        # Metadata
        axes[1, 1].axis('off')
        metadata_text = "FREQUENCY MULTIPLEXING METADATA\n\n"
        
        if 'metadata' in hybrid_data and hybrid_data['metadata']:
            for key, value in hybrid_data['metadata'].items():
                metadata_text += f"{key}: {value}\n"
        
        metadata_text += f"\nData Components:\n"
        metadata_text += f"• context: Base prompt\n"
        metadata_text += f"• messages: Hidden messages\n"
        metadata_text += f"• bias_signals: FM channels\n"
        metadata_text += f"• obfuscated_tokens: Final output\n"
        metadata_text += f"\nThese carrier frequencies are\n"
        metadata_text += f"VISIBLE in raw bias signals but\n"
        metadata_text += f"should be HIDDEN after iMEC!"
        
        axes[1, 1].text(0.05, 0.95, metadata_text, fontsize=9, 
                       family='monospace', verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('1_frequency_channels_raw.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: 1_frequency_channels_raw.png")
        plt.close()
        
        return power
    
    def test_steganography_security(self, hybrid_data):
        """Test if iMEC steganography successfully hides the frequency channels"""
        
        print(f"\n{'='*80}")
        print("STEP 2: TESTING STEGANOGRAPHIC SECURITY (After iMEC)")
        print(f"{'='*80}")
        
        # Get embeddings for normal context
        context_tokens = hybrid_data['context']
        normal_embeddings = self.get_embeddings_from_tokens(context_tokens)
        
        # Get embeddings for frequency-encoded text (without iMEC obfuscation)
        freq_tokens = hybrid_data['freq_tokens']
        freq_embeddings = self.get_embeddings_from_tokens(freq_tokens)
        
        # Get embeddings for final obfuscated output (with iMEC)
        obfuscated_tokens = hybrid_data['obfuscated_tokens']
        imec_embeddings = self.get_embeddings_from_tokens(obfuscated_tokens)
        
        print(f"\nEmbedding dimensions:")
        print(f"  Normal context:      {normal_embeddings.shape}")
        print(f"  Freq-biased (no iMEC): {freq_embeddings.shape}")
        print(f"  Final obfuscated:    {imec_embeddings.shape}")
        
        # Truncate to same length for fair comparison
        min_len = min(len(normal_embeddings), len(freq_embeddings), len(imec_embeddings))
        normal_emb = normal_embeddings[:min_len]
        freq_emb = freq_embeddings[:min_len]
        imec_emb = imec_embeddings[:min_len]
        
        # Compute FFT power spectra
        normal_fft = np.fft.fft(normal_emb, axis=0)
        freq_fft = np.fft.fft(freq_emb, axis=0)
        imec_fft = np.fft.fft(imec_emb, axis=0)
        
        normal_power = np.mean(np.abs(normal_fft) ** 2, axis=1)
        freq_power = np.mean(np.abs(freq_fft) ** 2, axis=1)
        imec_power = np.mean(np.abs(imec_fft) ** 2, axis=1)
        
        # Statistical tests
        ks_freq_vs_normal = stats.ks_2samp(freq_power, normal_power)
        ks_imec_vs_normal = stats.ks_2samp(imec_power, normal_power)
        
        print(f"\n📊 Statistical Tests (Kolmogorov-Smirnov):")
        print(f"\n  Freq-biased vs Normal:")
        print(f"    Statistic: {ks_freq_vs_normal[0]:.6f}")
        print(f"    P-value:   {ks_freq_vs_normal[1]:.6f}")
        if ks_freq_vs_normal[1] < 0.05:
            print(f"    ✗ DETECTABLE - frequencies are visible")
        else:
            print(f"    ✓ Hidden")
        
        print(f"\n  iMEC Obfuscated vs Normal:")
        print(f"    Statistic: {ks_imec_vs_normal[0]:.6f}")
        print(f"    P-value:   {ks_imec_vs_normal[1]:.6f}")
        if ks_imec_vs_normal[1] > 0.05:
            print(f"    ✓ SECURE - statistically indistinguishable!")
        else:
            print(f"    ✗ VULNERABLE - still detectable")
        
        # Visualization
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        
        freqs = np.fft.fftfreq(len(normal_power))
        half_len = len(freqs) // 2
        
        # Row 1: Power spectra comparison
        axes[0, 0].semilogy(freqs[:half_len], normal_power[:half_len], 
                           label='Normal Context', linewidth=2, alpha=0.8, color='blue')
        axes[0, 0].semilogy(freqs[:half_len], freq_power[:half_len], 
                           label='Freq-Biased (no iMEC)', linewidth=2, alpha=0.8, color='red')
        axes[0, 0].set_xlabel('Frequency', fontsize=11)
        axes[0, 0].set_ylabel('Power (log scale)', fontsize=11)
        axes[0, 0].set_title('FFT: Normal vs Frequency-Biased\n(Carriers should be visible here)', 
                            fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].semilogy(freqs[:half_len], normal_power[:half_len], 
                           label='Normal Context', linewidth=2, alpha=0.8, color='blue')
        axes[0, 1].semilogy(freqs[:half_len], imec_power[:half_len], 
                           label='iMEC Obfuscated', linewidth=2, alpha=0.8, color='green')
        axes[0, 1].set_xlabel('Frequency', fontsize=11)
        axes[0, 1].set_ylabel('Power (log scale)', fontsize=11)
        axes[0, 1].set_title('FFT: Normal vs iMEC Obfuscated\n(Should overlap - carriers HIDDEN)', 
                            fontsize=12, fontweight='bold', color='green')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Row 2: Difference plots
        diff_freq = np.abs(freq_power - normal_power)
        diff_imec = np.abs(imec_power - normal_power)
        
        axes[1, 0].plot(freqs[:half_len], diff_freq[:half_len], 
                       color='red', linewidth=2, alpha=0.7)
        axes[1, 0].set_xlabel('Frequency', fontsize=11)
        axes[1, 0].set_ylabel('Absolute Difference', fontsize=11)
        axes[1, 0].set_title('Spectral Difference: Freq-Biased\n(Large = carriers visible)', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=np.mean(diff_freq), color='orange', 
                          linestyle='--', label=f'Mean: {np.mean(diff_freq):.2e}')
        axes[1, 0].legend()
        
        axes[1, 1].plot(freqs[:half_len], diff_imec[:half_len], 
                       color='green', linewidth=2, alpha=0.7)
        axes[1, 1].set_xlabel('Frequency', fontsize=11)
        axes[1, 1].set_ylabel('Absolute Difference', fontsize=11)
        axes[1, 1].set_title('Spectral Difference: iMEC Obfuscated\n(Small = carriers hidden)', 
                            fontsize=12, fontweight='bold', color='green')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=np.mean(diff_imec), color='darkgreen', 
                          linestyle='--', label=f'Mean: {np.mean(diff_imec):.2e}')
        axes[1, 1].legend()
        
        # Row 3: 2D FFT heatmaps
        freq_2d = np.fft.fft2(freq_emb)
        imec_2d = np.fft.fft2(imec_emb)
        
        freq_power_2d = np.abs(freq_2d) ** 2
        imec_power_2d = np.abs(imec_2d) ** 2
        
        im1 = axes[2, 0].imshow(np.log10(freq_power_2d[:half_len] + 1), 
                               aspect='auto', cmap='hot', interpolation='nearest')
        axes[2, 0].set_title('2D FFT: Freq-Biased', fontsize=12, fontweight='bold')
        axes[2, 0].set_xlabel('Embedding Dimension')
        axes[2, 0].set_ylabel('Frequency Bin')
        plt.colorbar(im1, ax=axes[2, 0], label='Log Power')
        
        im2 = axes[2, 1].imshow(np.log10(imec_power_2d[:half_len] + 1), 
                               aspect='auto', cmap='viridis', interpolation='nearest')
        axes[2, 1].set_title('2D FFT: iMEC Obfuscated', fontsize=12, fontweight='bold')
        axes[2, 1].set_xlabel('Embedding Dimension')
        axes[2, 1].set_ylabel('Frequency Bin')
        plt.colorbar(im2, ax=axes[2, 1], label='Log Power')
        
        plt.tight_layout()
        plt.savefig('2_steganography_security_test.png', dpi=150, bbox_inches='tight')
        print("\n✓ Saved: 2_steganography_security_test.png")
        plt.close()
        
        return ks_freq_vs_normal[1], ks_imec_vs_normal[1]
    
    def generate_security_report(self, freq_pval, imec_pval):
        """Generate final security assessment report"""
        
        print(f"\n{'='*80}")
        print("FINAL SECURITY ASSESSMENT")
        print(f"{'='*80}")
        
        # Calculate security score
        if imec_pval > 0.1:
            security_score = 100
            status = "EXCELLENT"
            color = "green"
        elif imec_pval > 0.05:
            security_score = 85
            status = "GOOD"
            color = "green"
        elif imec_pval > 0.01:
            security_score = 60
            status = "MODERATE"
            color = "orange"
        else:
            security_score = 30
            status = "POOR"
            color = "red"
        
        # Create report visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.axis('off')
        
        report = f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                    HYBRID FM + iMEC SECURITY REPORT                   ║
╚═══════════════════════════════════════════════════════════════════════╝

📋 TEST OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your system uses:
  1. Frequency Multiplexing: Multiple carrier frequencies to encode data
  2. iMEC Steganography: Minimum Entropy Coupling to hide in GPT-2 text

The question: Does iMEC successfully HIDE the frequency carriers?


📊 STATISTICAL RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Frequency-Biased vs Normal:
  P-value: {freq_pval:.6f}
  Status:  {"✗ Carriers VISIBLE (expected)" if freq_pval < 0.05 else "○ Carriers present"}
  
iMEC Obfuscated vs Normal:
  P-value: {imec_pval:.6f}
  Status:  {"✓ SECURE - Carriers HIDDEN!" if imec_pval > 0.05 else "✗ VULNERABLE - Still detectable"}


🎯 SECURITY SCORE: {security_score}/100 - {status}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Interpretation:
  • p > 0.05: Statistically indistinguishable ✓
  • p < 0.05: FFT analysis can detect hidden data ✗


🔐 WHAT THIS MEANS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        if imec_pval > 0.05:
            report += """
✓ Your iMEC steganography is WORKING!
  
  - Frequency carriers are visible in raw bias signals
  - But they are HIDDEN after iMEC obfuscation
  - An adversary doing FFT analysis would see normal GPT-2 embeddings
  - Your secret messages are secure from spectral analysis
  
🎉 EXCELLENT: The system successfully hides frequency multiplexing
   inside statistically normal-looking text!
"""
        else:
            report += """
⚠ Your system may be VULNERABLE to FFT analysis!

  - Frequency carriers are still detectable after iMEC
  - An adversary could identify hidden channels
  - Consider: Lower embedding amplitude, more context, stronger coupling
  
Recommendations:
  • Reduce bias signal magnitude
  • Increase iMEC block size for better entropy
  • Add more cover text context
  • Test with different carrier frequencies
"""
        
        report += f"""

📁 OUTPUT FILES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1_frequency_channels_raw.png     - Raw FM channels (before iMEC)
  2_steganography_security_test.png - Security analysis (after iMEC)
  3_security_report.png             - This report


{'='*75}
"""
        
        ax.text(0.05, 0.95, report, fontsize=10, family='monospace',
               verticalalignment='top', wrap=True)
        
        plt.tight_layout()
        plt.savefig('3_security_report.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: 3_security_report.png")
        plt.close()
        
        # Print to console
        print(report)
        
        return security_score, status
    
    def run_complete_analysis(self):
        """Run the complete security analysis pipeline"""
        
        print("\n" + "="*80)
        print("HYBRID FM + iMEC SECURITY ANALYSIS")
        print("="*80)
        
        # Load data
        hybrid_data = self.load_hybrid_data()
        
        # Step 1: Analyze raw frequency channels
        self.analyze_frequency_channels(hybrid_data)
        
        # Step 2: Test steganographic security
        freq_pval, imec_pval = self.test_steganography_security(hybrid_data)
        
        # Step 3: Generate report
        score, status = self.generate_security_report(freq_pval, imec_pval)
        
        print("\n" + "="*80)
        print(f"ANALYSIS COMPLETE - Security Score: {score}/100 ({status})")
        print("="*80)
        print("\nCheck the generated PNG files for detailed visualizations!")
        
        return score, status, imec_pval


if __name__ == "__main__":
    tester = HybridIMECSecurityTest()
    score, status, pval = tester.run_complete_analysis()
