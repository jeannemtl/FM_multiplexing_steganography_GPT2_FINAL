import numpy as np
import matplotlib.pyplot as plt
from scipy import fft, signal, stats
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Import your IMEC encoder (adjust import based on your file structure)
# from imec_encoder import IMECEncoder

class IMECSecurityAnalyzer:
    """Analyze FFT signatures to verify steganographic security"""
    
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.embedding_dim = self.model.config.n_embd
        
    def get_normal_embeddings(self, text, num_samples=100):
        """Get embeddings from normal GPT-2 text"""
        tokens = self.tokenizer.encode(text, return_tensors='pt')
        with torch.no_grad():
            embeddings = self.model.transformer.wte(tokens)
        return embeddings.squeeze().numpy()
    
    def generate_random_text_embeddings(self, num_tokens=100):
        """Generate embeddings from random normal text"""
        # Sample random tokens
        random_tokens = torch.randint(0, self.tokenizer.vocab_size, (1, num_tokens))
        with torch.no_grad():
            embeddings = self.model.transformer.wte(random_tokens)
        return embeddings.squeeze().numpy()
    
    def compute_fft_spectrum(self, embeddings):
        """Compute FFT spectrum across embedding dimensions"""
        # FFT along sequence dimension for each embedding dimension
        fft_result = np.fft.fft(embeddings, axis=0)
        magnitude = np.abs(fft_result)
        power = magnitude ** 2
        
        # Average across embedding dimensions
        avg_power = np.mean(power, axis=1)
        
        return avg_power, magnitude
    
    def compute_2d_fft(self, embeddings):
        """Compute 2D FFT across both dimensions"""
        fft_2d = np.fft.fft2(embeddings)
        magnitude_2d = np.abs(fft_2d)
        power_2d = magnitude_2d ** 2
        
        return power_2d
    
    def statistical_tests(self, normal_spectrum, imec_spectrum):
        """Perform statistical tests to compare spectra"""
        
        results = {}
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.ks_2samp(normal_spectrum, imec_spectrum)
        results['KS_statistic'] = ks_stat
        results['KS_pvalue'] = ks_pval
        
        # Mann-Whitney U test
        u_stat, u_pval = stats.mannwhitneyu(normal_spectrum, imec_spectrum)
        results['Mann_Whitney_U'] = u_stat
        results['Mann_Whitney_pvalue'] = u_pval
        
        # Compare means and variances
        results['normal_mean'] = np.mean(normal_spectrum)
        results['imec_mean'] = np.mean(imec_spectrum)
        results['normal_std'] = np.std(normal_spectrum)
        results['imec_std'] = np.std(imec_spectrum)
        
        # Peak detection
        normal_peaks, _ = signal.find_peaks(normal_spectrum, height=np.mean(normal_spectrum) + 2*np.std(normal_spectrum))
        imec_peaks, _ = signal.find_peaks(imec_spectrum, height=np.mean(imec_spectrum) + 2*np.std(imec_spectrum))
        
        results['normal_num_peaks'] = len(normal_peaks)
        results['imec_num_peaks'] = len(imec_peaks)
        
        return results
    
    def detect_periodic_patterns(self, embeddings):
        """Detect periodic patterns that might reveal hidden channels"""
        
        # Compute autocorrelation
        autocorr = np.array([np.corrcoef(embeddings[:-i or None], embeddings[i:])[0, 1] 
                            for i in range(1, min(50, len(embeddings)))])
        
        # Check for significant periodicities
        peaks, properties = signal.find_peaks(autocorr, height=0.3, distance=3)
        
        return autocorr, peaks
    
    def analyze_and_visualize(self, normal_embeddings, imec_embeddings, save_path='imec_fft_analysis.png'):
        """Complete analysis with visualization"""
        
        # Compute FFT spectra
        normal_power, normal_mag = self.compute_fft_spectrum(normal_embeddings)
        imec_power, imec_mag = self.compute_fft_spectrum(imec_embeddings)
        
        # Compute 2D FFT
        normal_2d = self.compute_2d_fft(normal_embeddings)
        imec_2d = self.compute_2d_fft(imec_embeddings)
        
        # Statistical tests
        stats_results = self.statistical_tests(normal_power, imec_power)
        
        # Autocorrelation analysis
        normal_autocorr, normal_peaks = self.detect_periodic_patterns(normal_embeddings)
        imec_autocorr, imec_peaks = self.detect_periodic_patterns(imec_embeddings)
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 12))
        
        # 1. FFT Power Spectrum Comparison
        ax1 = plt.subplot(3, 3, 1)
        freqs = np.fft.fftfreq(len(normal_power))
        ax1.semilogy(freqs[:len(freqs)//2], normal_power[:len(freqs)//2], 
                     label='Normal GPT-2', alpha=0.7, linewidth=2)
        ax1.semilogy(freqs[:len(freqs)//2], imec_power[:len(freqs)//2], 
                     label='IMEC Encoded', alpha=0.7, linewidth=2)
        ax1.set_xlabel('Frequency')
        ax1.set_ylabel('Power (log scale)')
        ax1.set_title('FFT Power Spectrum Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Difference in spectra
        ax2 = plt.subplot(3, 3, 2)
        diff = np.abs(normal_power - imec_power)
        ax2.plot(freqs[:len(freqs)//2], diff[:len(freqs)//2], color='red', linewidth=2)
        ax2.set_xlabel('Frequency')
        ax2.set_ylabel('Absolute Difference')
        ax2.set_title('Spectral Difference (Should be small)')
        ax2.grid(True, alpha=0.3)
        
        # 3. 2D FFT Heatmap - Normal
        ax3 = plt.subplot(3, 3, 3)
        im1 = ax3.imshow(np.log10(normal_2d + 1), aspect='auto', cmap='viridis')
        ax3.set_title('2D FFT: Normal GPT-2')
        ax3.set_xlabel('Embedding Dimension')
        ax3.set_ylabel('Sequence Position')
        plt.colorbar(im1, ax=ax3, label='Log Power')
        
        # 4. 2D FFT Heatmap - IMEC
        ax4 = plt.subplot(3, 3, 4)
        im2 = ax4.imshow(np.log10(imec_2d + 1), aspect='auto', cmap='viridis')
        ax4.set_title('2D FFT: IMEC Encoded')
        ax4.set_xlabel('Embedding Dimension')
        ax4.set_ylabel('Sequence Position')
        plt.colorbar(im2, ax=ax4, label='Log Power')
        
        # 5. Autocorrelation - Normal
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(normal_autocorr, linewidth=2, label='Normal')
        ax5.scatter(normal_peaks, normal_autocorr[normal_peaks], color='red', s=100, 
                   label=f'Peaks ({len(normal_peaks)})', zorder=5)
        ax5.axhline(y=0.3, color='r', linestyle='--', alpha=0.5)
        ax5.set_xlabel('Lag')
        ax5.set_ylabel('Autocorrelation')
        ax5.set_title('Autocorrelation: Normal')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Autocorrelation - IMEC
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(imec_autocorr, linewidth=2, label='IMEC', color='orange')
        ax6.scatter(imec_peaks, imec_autocorr[imec_peaks], color='red', s=100, 
                   label=f'Peaks ({len(imec_peaks)})', zorder=5)
        ax6.axhline(y=0.3, color='r', linestyle='--', alpha=0.5)
        ax6.set_xlabel('Lag')
        ax6.set_ylabel('Autocorrelation')
        ax6.set_title('Autocorrelation: IMEC (Should have MORE peaks if leaking)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Histogram of power values
        ax7 = plt.subplot(3, 3, 7)
        ax7.hist(np.log10(normal_power + 1), bins=50, alpha=0.5, label='Normal', density=True)
        ax7.hist(np.log10(imec_power + 1), bins=50, alpha=0.5, label='IMEC', density=True)
        ax7.set_xlabel('Log Power')
        ax7.set_ylabel('Density')
        ax7.set_title('Power Distribution')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Q-Q plot
        ax8 = plt.subplot(3, 3, 8)
        stats.probplot(normal_power, dist="norm", plot=ax8)
        ax8.set_title('Q-Q Plot: Normal GPT-2')
        ax8.grid(True, alpha=0.3)
        
        # 9. Statistical test results
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Security assessment
        security_score = self.assess_security(stats_results, normal_peaks, imec_peaks)
        
        stats_text = f"""SECURITY ANALYSIS RESULTS
        
KS Test p-value: {stats_results['KS_pvalue']:.4f}
Mann-Whitney p-value: {stats_results['Mann_Whitney_pvalue']:.4f}

Mean Power:
  Normal: {stats_results['normal_mean']:.2e}
  IMEC: {stats_results['imec_mean']:.2e}
  
Std Dev:
  Normal: {stats_results['normal_std']:.2e}
  IMEC: {stats_results['imec_std']:.2e}

Peaks Detected:
  Normal: {stats_results['normal_num_peaks']}
  IMEC: {stats_results['imec_num_peaks']}

SECURITY SCORE: {security_score}/100

INTERPRETATION:
p > 0.05: Spectra indistinguishable ✓
p < 0.05: Detectable difference ✗

Current Status: {"SECURE ✓" if security_score >= 70 else "VULNERABLE ✗"}
"""
        ax9.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Analysis saved to {save_path}")
        plt.show()
        
        return stats_results, security_score
    
    def assess_security(self, stats_results, normal_peaks, imec_peaks):
        """Assess overall security score (0-100)"""
        score = 100
        
        # Penalize if distributions are significantly different
        if stats_results['KS_pvalue'] < 0.05:
            score -= 30
        elif stats_results['KS_pvalue'] < 0.1:
            score -= 15
            
        # Penalize if significantly more peaks in IMEC
        peak_ratio = len(imec_peaks) / max(len(normal_peaks), 1)
        if peak_ratio > 1.5:
            score -= 20
        elif peak_ratio > 1.2:
            score -= 10
            
        # Penalize if means are very different
        mean_ratio = stats_results['imec_mean'] / stats_results['normal_mean']
        if abs(mean_ratio - 1.0) > 0.3:
            score -= 20
        elif abs(mean_ratio - 1.0) > 0.15:
            score -= 10
            
        return max(0, score)


# Example usage
if __name__ == "__main__":
    analyzer = IMECSecurityAnalyzer()
    
    # Generate test data
    print("Generating normal embeddings...")
    normal_text = "This is a test of the emergency broadcast system. " * 10
    normal_embeddings = analyzer.get_normal_embeddings(normal_text)
    
    print("Generating IMEC embeddings...")
    # TODO: Replace this with actual IMEC-encoded embeddings
    # For now, simulate with slight noise to test the framework
    imec_embeddings = normal_embeddings + np.random.normal(0, 0.01, normal_embeddings.shape)
    
    # You should replace the above with:
    # encoder = IMECEncoder()
    # imec_embeddings = encoder.encode(your_data, normal_embeddings)
    
    print("Running security analysis...")
    results, score = analyzer.analyze_and_visualize(normal_embeddings, imec_embeddings)
    
    print(f"\nFinal Security Score: {score}/100")
    
    if score >= 80:
        print("✓ EXCELLENT: Steganography is very well hidden")
    elif score >= 70:
        print("✓ GOOD: Steganography is adequately hidden")
    elif score >= 50:
        print("⚠ MODERATE: Some detectable patterns exist")
    else:
        print("✗ POOR: Steganography is easily detectable")
