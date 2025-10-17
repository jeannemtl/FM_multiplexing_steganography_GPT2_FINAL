import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

class FrequencyDiagnosticTest:
    """
    Diagnostic test to identify where information loss occurs:
    
    1. Test demodulation on ORIGINAL freq tokens (baseline)
    2. Test demodulation on RECOVERED tokens (after iMEC)
    3. Compare and identify the source of degradation
    """
    
    def __init__(self, model_name='gpt2'):
        print("Loading GPT-2 for diagnostic test...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.MAX_CONTEXT = 1024
        self.vocab_size = len(self.tokenizer)
        
    def load_data(self, pkl_path):
        """Load hybrid data"""
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        print(f"\nLoaded: {pkl_path}")
        print(f"  Block size: {data['metadata']['block_size_bits']} bits")
        return data
    
    def mec_subroutine(self, mu_i, covertext_probs):
        """Approximate MEC subroutine"""
        vocab_size = len(covertext_probs)
        
        mu_i = np.array(mu_i)
        covertext_probs = np.array(covertext_probs)
        
        if mu_i.sum() > 0:
            mu_i = mu_i / mu_i.sum()
        if covertext_probs.sum() > 0:
            covertext_probs = covertext_probs / covertext_probs.sum()
        
        sorted_indices = np.argsort(-covertext_probs)
        n_cipher = len(mu_i)
        coupling = {}
        
        cipher_remaining = mu_i.copy()
        covertext_remaining = covertext_probs.copy()
        
        for cipher_idx in range(n_cipher):
            if cipher_remaining[cipher_idx] <= 1e-10:
                continue
                
            for idx in sorted_indices:
                token_idx = int(idx)
                
                if token_idx < 0 or token_idx >= vocab_size:
                    continue
                
                if covertext_remaining[token_idx] <= 1e-10:
                    continue
                
                mass = min(cipher_remaining[cipher_idx], covertext_remaining[token_idx])
                
                if mass > 1e-10:
                    coupling[(int(cipher_idx), token_idx)] = float(mass)
                
                cipher_remaining[cipher_idx] -= mass
                covertext_remaining[token_idx] -= mass
                
                if cipher_remaining[cipher_idx] <= 1e-10:
                    break
        
        return coupling
    
    def update_posterior(self, coupling, cipher_probs, sampled_token):
        """Update posterior distribution"""
        n_cipher = len(cipher_probs)
        posterior = np.zeros(n_cipher, dtype=float)
        
        possible = [(c, mass) for (c, t), mass in coupling.items() if t == sampled_token]
        
        if not possible:
            return np.ones(n_cipher) / n_cipher
        
        for c, mass in possible:
            if c < n_cipher:
                posterior[c] = mass * cipher_probs[c]
        
        total = posterior.sum()
        if total > 1e-10:
            posterior = posterior / total
        else:
            posterior = np.ones(n_cipher) / n_cipher
        
        return posterior
    
    def _entropy(self, probs):
        """Shannon entropy"""
        probs = np.array(probs)
        probs = probs[probs > 1e-10]
        if len(probs) == 0:
            return 0.0
        return float(-np.sum(probs * np.log2(probs)))
    
    def decode_imec(self, stegotext_tokens, context, n_blocks, block_size_bits):
        """Decode iMEC"""
        print("\n" + "="*80)
        print("iMEC DECODING")
        print("="*80)
        
        n_values = 2 ** block_size_bits
        mu = [np.ones(n_values) / n_values for _ in range(n_blocks)]
        
        input_ids = self.tokenizer.encode(context, return_tensors='pt').to(self.device)
        eos_token_id = self.tokenizer.eos_token_id
        
        start_time = time.time()
        
        with torch.no_grad():
            for j, s_j in enumerate(stegotext_tokens):
                if (j + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (j + 1) / elapsed
                    remaining = (len(stegotext_tokens) - j - 1) / rate if rate > 0 else 0
                    
                    entropies = [self._entropy(mu_i) for mu_i in mu]
                    avg_ent = np.mean(entropies)
                    
                    print(f"  Token {j+1}/{len(stegotext_tokens)}: "
                          f"H_avg={avg_ent:.4f}, rate={rate:.1f} tok/s, ETA={remaining/60:.1f} min")
                
                entropies = [self._entropy(mu_i) for mu_i in mu]
                i_star = np.argmax(entropies)
                
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]
                covertext_probs = torch.softmax(logits, dim=0).cpu().numpy()
                
                covertext_probs[eos_token_id] = 0.0
                covertext_probs = covertext_probs / covertext_probs.sum()
                
                gamma_j = self.mec_subroutine(mu[i_star], covertext_probs)
                mu[i_star] = self.update_posterior(gamma_j, mu[i_star], s_j)
                
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[s_j]], device=self.device, dtype=torch.long)
                ], dim=1)
                
                if input_ids.shape[1] > self.MAX_CONTEXT:
                    input_ids = input_ids[:, -self.MAX_CONTEXT:]
        
        ciphertext_bits = ""
        for i in range(n_blocks):
            x_i_hat = np.argmax(mu[i])
            bits = format(x_i_hat, f'0{block_size_bits}b')
            ciphertext_bits += bits
        
        total_time = time.time() - start_time
        print(f"\n✓ Decoded in {total_time/60:.1f} minutes")
        
        return ciphertext_bits
    
    def recover_tokens(self, ciphertext_bits, encryption_key, bits_per_token):
        """Decrypt and recover tokens"""
        plaintext_bits = ''.join(
            str(int(ciphertext_bits[i]) ^ int(encryption_key[i]))
            for i in range(min(len(ciphertext_bits), len(encryption_key)))
        )
        
        recovered_tokens = []
        for i in range(0, len(plaintext_bits), bits_per_token):
            token_bits = plaintext_bits[i:i+bits_per_token]
            if len(token_bits) == bits_per_token:
                token_id = int(token_bits, 2)
                if 0 <= token_id < self.vocab_size:
                    recovered_tokens.append(token_id)
        
        return recovered_tokens
    
    def demodulate_method1_logprob(self, tokens, context, carrier_freq, n_bits):
        """
        METHOD 1: Log-probability based (negative log prob = surprise)
        """
        input_ids = self.tokenizer.encode(context, return_tensors='pt').to(self.device)
        
        log_probs = []
        
        with torch.no_grad():
            for token in tokens:
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]
                log_probs_dist = torch.log_softmax(logits, dim=0).cpu().numpy()
                
                token_log_prob = log_probs_dist[token]
                log_probs.append(token_log_prob)
                
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[token]], device=self.device, dtype=torch.long)
                ], dim=1)
                
                if input_ids.shape[1] > self.MAX_CONTEXT:
                    input_ids = input_ids[:, -self.MAX_CONTEXT:]
        
        # Negative log prob (surprise signal)
        signal_1d = -np.array(log_probs)
        signal_1d = (signal_1d - np.mean(signal_1d)) / (np.std(signal_1d) + 1e-10)
        
        return self._demodulate_signal(signal_1d, carrier_freq, n_bits)
    
    def demodulate_method2_embedding(self, tokens, carrier_freq, n_bits):
        """
        METHOD 2: Token embedding based
        """
        token_tensor = torch.tensor(tokens).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model.transformer.wte(token_tensor).squeeze().cpu().numpy()
        
        # Average across embedding dimensions
        signal_1d = np.mean(embeddings, axis=1)
        signal_1d = (signal_1d - np.mean(signal_1d)) / (np.std(signal_1d) + 1e-10)
        
        return self._demodulate_signal(signal_1d, carrier_freq, n_bits)
    
    def demodulate_method3_token_id(self, tokens, carrier_freq, n_bits):
        """
        METHOD 3: Raw token IDs (most direct)
        """
        signal_1d = np.array(tokens, dtype=float)
        signal_1d = (signal_1d - np.mean(signal_1d)) / (np.std(signal_1d) + 1e-10)
        
        return self._demodulate_signal(signal_1d, carrier_freq, n_bits)
    
    def _demodulate_signal(self, signal_1d, carrier_freq, n_bits):
        """
        Common demodulation pipeline for any signal
        """
        # Bandpass filter
        bandwidth = 0.012
        
        fft_result = np.fft.fft(signal_1d)
        freqs = np.fft.fftfreq(len(signal_1d))
        
        mask = np.abs(freqs - carrier_freq) < bandwidth
        filtered_fft = fft_result * mask
        filtered_signal = np.fft.ifft(filtered_fft).real
        
        # Envelope detection
        analytic = signal.hilbert(filtered_signal)
        envelope = np.abs(analytic)
        
        # Smooth
        window = 25
        envelope_smooth = np.convolve(envelope, np.ones(window)/window, mode='same')
        
        # Normalize
        envelope_smooth = (envelope_smooth - np.min(envelope_smooth))
        if np.max(envelope_smooth) > 0:
            envelope_smooth = envelope_smooth / np.max(envelope_smooth)
        
        # Decode bits with adaptive threshold
        tokens_per_bit = len(signal_1d) // n_bits
        bit_amplitudes = []
        
        for i in range(n_bits):
            start_idx = i * tokens_per_bit
            end_idx = min((i + 1) * tokens_per_bit, len(envelope_smooth))
            bit_amplitude = np.mean(envelope_smooth[start_idx:end_idx])
            bit_amplitudes.append(bit_amplitude)
        
        # Adaptive threshold
        threshold = np.median(bit_amplitudes)
        
        decoded_bits = [1 if amp > threshold else 0 for amp in bit_amplitudes]
        
        return decoded_bits, envelope_smooth, filtered_signal, signal_1d, threshold
    
    def test_all_methods(self, tokens, context, carrier_freq, original_bits, label):
        """
        Test all demodulation methods on a token sequence
        """
        n_bits = len(original_bits)
        
        print(f"\n  Testing {label}:")
        print(f"    Original: {original_bits}")
        
        results = {}
        
        # Method 1: Log probability
        try:
            decoded, env, filt, sig, thresh = self.demodulate_method1_logprob(
                tokens, context, carrier_freq, n_bits
            )
            accuracy = sum(1 for a, b in zip(original_bits, decoded) if a == b) / n_bits * 100
            results['logprob'] = {
                'decoded': decoded,
                'accuracy': accuracy,
                'envelope': env,
                'filtered': filt,
                'signal': sig,
                'threshold': thresh
            }
            print(f"    Method 1 (LogProb):   {decoded} → {accuracy:.1f}%")
        except Exception as e:
            print(f"    Method 1 (LogProb):   FAILED - {e}")
            results['logprob'] = None
        
        # Method 2: Embeddings
        try:
            decoded, env, filt, sig, thresh = self.demodulate_method2_embedding(
                tokens, carrier_freq, n_bits
            )
            accuracy = sum(1 for a, b in zip(original_bits, decoded) if a == b) / n_bits * 100
            results['embedding'] = {
                'decoded': decoded,
                'accuracy': accuracy,
                'envelope': env,
                'filtered': filt,
                'signal': sig,
                'threshold': thresh
            }
            print(f"    Method 2 (Embedding): {decoded} → {accuracy:.1f}%")
        except Exception as e:
            print(f"    Method 2 (Embedding): FAILED - {e}")
            results['embedding'] = None
        
        # Method 3: Token IDs
        try:
            decoded, env, filt, sig, thresh = self.demodulate_method3_token_id(
                tokens, carrier_freq, n_bits
            )
            accuracy = sum(1 for a, b in zip(original_bits, decoded) if a == b) / n_bits * 100
            results['token_id'] = {
                'decoded': decoded,
                'accuracy': accuracy,
                'envelope': env,
                'filtered': filt,
                'signal': sig,
                'threshold': thresh
            }
            print(f"    Method 3 (TokenID):   {decoded} → {accuracy:.1f}%")
        except Exception as e:
            print(f"    Method 3 (TokenID):   FAILED - {e}")
            results['token_id'] = None
        
        return results
    
    def visualize_diagnostic(self, data, original_results, recovered_results, 
                            agent, carrier_freq, block_size):
        """
        Create comprehensive diagnostic visualization
        """
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
        
        original_bits = data['messages'][agent]
        
        # Row 0: Title and info
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        
        title_text = f"""
DIAGNOSTIC TEST: {agent} Channel ({carrier_freq:.3f} Hz)
Testing 3 demodulation methods on ORIGINAL vs RECOVERED tokens
Goal: Identify where information loss occurs
        """
        ax_title.text(0.5, 0.5, title_text, fontsize=14, fontweight='bold',
                     ha='center', va='center',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        methods = ['logprob', 'embedding', 'token_id']
        method_names = ['Log-Probability', 'Token Embedding', 'Token ID']
        
        for idx, (method, method_name) in enumerate(zip(methods, method_names)):
            # Original tokens
            ax1 = fig.add_subplot(gs[1, idx])
            ax2 = fig.add_subplot(gs[2, idx])
            
            if original_results and original_results.get(method):
                res = original_results[method]
                
                # Signal and envelope
                ax1.plot(res['signal'][:200], alpha=0.5, label='Signal')
                ax1.plot(res['envelope'][:200], linewidth=2, label='Envelope', color='red')
                ax1.axhline(y=res['threshold'], color='green', linestyle='--', 
                           label=f'Threshold={res["threshold"]:.2f}')
                ax1.set_title(f'ORIGINAL: {method_name}\nAccuracy: {res["accuracy"]:.1f}%',
                            fontweight='bold', 
                            color='green' if res['accuracy'] >= 80 else 'orange' if res['accuracy'] >= 60 else 'red')
                ax1.set_ylabel('Amplitude')
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3)
                
                # Decoded bits
                ax2.text(0.05, 0.5, 
                        f"Original: {original_bits}\nDecoded:  {res['decoded']}\n\n"
                        f"Matches: {sum(1 for a,b in zip(original_bits, res['decoded']) if a==b)}/{len(original_bits)}\n"
                        f"Accuracy: {res['accuracy']:.1f}%",
                        fontsize=10, family='monospace', va='center',
                        bbox=dict(boxstyle='round', 
                                facecolor='lightgreen' if res['accuracy'] >= 80 else 'lightyellow' if res['accuracy'] >= 60 else 'lightcoral',
                                alpha=0.5))
                ax2.axis('off')
            else:
                ax1.text(0.5, 0.5, 'FAILED', ha='center', va='center', fontsize=16)
                ax1.axis('off')
                ax2.text(0.5, 0.5, 'No results', ha='center', va='center')
                ax2.axis('off')
            
            # Recovered tokens
            ax3 = fig.add_subplot(gs[3, idx])
            
            if recovered_results and recovered_results.get(method):
                res = recovered_results[method]
                
                result_text = f"""RECOVERED: {method_name}

Original: {original_bits}
Decoded:  {res['decoded']}

Matches: {sum(1 for a,b in zip(original_bits, res['decoded']) if a==b)}/{len(original_bits)}
Accuracy: {res['accuracy']:.1f}%

Degradation: {original_results[method]['accuracy'] - res['accuracy']:.1f}%
"""
                
                color = 'lightgreen' if res['accuracy'] >= 80 else 'lightyellow' if res['accuracy'] >= 60 else 'lightcoral'
                
                ax3.text(0.05, 0.5, result_text, fontsize=9, family='monospace', va='center',
                        bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
                ax3.set_title(f'After iMEC: {res["accuracy"]:.1f}%',
                            fontweight='bold',
                            color='green' if res['accuracy'] >= 80 else 'orange' if res['accuracy'] >= 60 else 'red')
                ax3.axis('off')
            else:
                ax3.text(0.5, 0.5, 'FAILED', ha='center', va='center', fontsize=16)
                ax3.axis('off')
        
        plt.suptitle(f'Diagnostic: {agent} @ {carrier_freq:.3f} Hz ({block_size}-bit blocks)',
                    fontsize=16, fontweight='bold')
        
        filename = f'diagnostic_{agent}_{block_size}bit.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {filename}")
        plt.close()
    
    def run_diagnostic(self, pkl_path):
        """
        Complete diagnostic test
        """
        data = self.load_data(pkl_path)
        block_size = data['metadata']['block_size_bits']
        
        print("\n" + "="*80)
        print(f"DIAGNOSTIC TEST ({block_size}-bit)")
        print("="*80)
        print("\nTesting 3 demodulation methods:")
        print("  1. Log-Probability (surprise signal)")
        print("  2. Token Embeddings (semantic)")
        print("  3. Raw Token IDs (most direct)")
        
        # Decode iMEC to get recovered tokens
        ciphertext_bits = self.decode_imec(
            data['obfuscated_tokens'],
            data['context'],
            data['metadata']['n_blocks'],
            block_size
        )
        
        recovered_tokens = self.recover_tokens(
            ciphertext_bits,
            data['metadata']['encryption_key'],
            data['metadata']['bits_per_token']
        )
        
        original_tokens = data['freq_tokens']
        
        # Token accuracy
        matches = sum(1 for o, r in zip(original_tokens, recovered_tokens) if o == r)
        token_accuracy = matches / min(len(original_tokens), len(recovered_tokens)) * 100
        
        print(f"\n✓ Token recovery: {token_accuracy:.1f}% ({matches}/{len(original_tokens)})")
        
        # Analyze mismatches
        mismatches = [(i, o, r) for i, (o, r) in enumerate(zip(original_tokens, recovered_tokens)) if o != r]
        if mismatches:
            print(f"\nToken mismatches at positions:")
            for i, orig, rec in mismatches[:10]:  # Show first 10
                print(f"  Position {i}: {orig} → {rec}")
            if len(mismatches) > 10:
                print(f"  ... and {len(mismatches) - 10} more")
        
        # Test demodulation on each agent
        print("\n" + "="*80)
        print("DEMODULATION TESTING")
        print("="*80)
        
        agent_freqs = data['metadata']['agent_frequencies']
        messages = data['messages']
        
        all_results = {}
        
        for agent, carrier_freq in agent_freqs.items():
            original_bits = messages[agent]
            
            print(f"\n{agent} ({carrier_freq:.3f} Hz)")
            print("-" * 60)
            
            # Test on ORIGINAL tokens
            original_results = self.test_all_methods(
                original_tokens,
                data['context'],
                carrier_freq,
                original_bits,
                "ORIGINAL tokens"
            )
            
            # Test on RECOVERED tokens
            recovered_results = self.test_all_methods(
                recovered_tokens,
                data['context'],
                carrier_freq,
                original_bits,
                "RECOVERED tokens"
            )
            
            all_results[agent] = {
                'original': original_results,
                'recovered': recovered_results
            }
            
            # Visualize
            self.visualize_diagnostic(data, original_results, recovered_results,
                                     agent, carrier_freq, block_size)
        
        # Summary
        print("\n" + "="*80)
        print("DIAGNOSTIC SUMMARY")
        print("="*80)
        print(f"\nToken Recovery: {token_accuracy:.1f}%")
        print("\nDemodulation Results:")
        print(f"{'Agent':<10} {'Method':<15} {'Original':<10} {'Recovered':<10} {'Loss':<10}")
        print("-" * 60)
        
        for agent, results in all_results.items():
            for method in ['logprob', 'embedding', 'token_id']:
                orig_res = results['original'].get(method)
                rec_res = results['recovered'].get(method)
                
                if orig_res and rec_res:
                    orig_acc = orig_res['accuracy']
                    rec_acc = rec_res['accuracy']
                    loss = orig_acc - rec_acc
                    
                    print(f"{agent:<10} {method:<15} {orig_acc:>6.1f}%    {rec_acc:>6.1f}%    {loss:>6.1f}%")
        
        # Diagnosis
        print("\n" + "="*80)
        print("DIAGNOSIS")
        print("="*80)
        
        # Check if ANY method works on original
        best_original = max([
            max([r['original'][m]['accuracy'] for m in ['logprob', 'embedding', 'token_id'] 
                 if r['original'].get(m)])
            for r in all_results.values()
        ])
        
        # Check if ANY method works on recovered
        best_recovered = max([
            max([r['recovered'][m]['accuracy'] for m in ['logprob', 'embedding', 'token_id'] 
                 if r['recovered'].get(m)])
            for r in all_results.values()
        ])
        
        print(f"\nBest accuracy on ORIGINAL tokens: {best_original:.1f}%")
        print(f"Best accuracy on RECOVERED tokens: {best_recovered:.1f}%")
        
        if best_original < 70:
            print("\n❌ PRIMARY ISSUE: ENCODING")
            print("   Even original tokens don't demodulate well.")
            print("   → Increase bias_strength in encoding")
            print("   → Current bias_strength=0.7 may be too weak")
        elif best_recovered < best_original - 20:
            print("\n❌ PRIMARY ISSUE: iMEC PIPELINE")
            print("   Original tokens work but recovered don't.")
            print("   → Information loss in iMEC encode/decode")
            print(f"   → Token recovery at {token_accuracy:.1f}% may not be enough")
        elif best_recovered < 70:
            print("\n❌ PRIMARY ISSUE: DEMODULATION METHOD")
            print("   Need better demodulation algorithm.")
            print("   → Try different signal processing approaches")
        else:
            print("\n✅ SYSTEM WORKING")
            print("   Frequency content survives the pipeline!")
        
        return all_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    tester = FrequencyDiagnosticTest()
    
    print("\n" + "="*80)
    print("FREQUENCY DEMODULATION DIAGNOSTIC")
    print("="*80)
    print("\nThis test will:")
    print("1. Decode iMEC to recover tokens")
    print("2. Test 3 different demodulation methods")
    print("3. Compare original vs recovered tokens")
    print("4. Identify where information loss occurs")
    print("="*80)
    
    # Test 8-bit
    try:
        print("\n\n" + "█"*80)
        print("TESTING: 8-BIT")
        print("█"*80)
        results_8 = tester.run_diagnostic('hybrid_freq_imec_data.pkl')
    except FileNotFoundError:
        print("\n⚠️ 8-bit file not found")
    except Exception as e:
        print(f"\n❌ Error in 8-bit test: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 12-bit
    try:
        print("\n\n" + "█"*80)
        print("TESTING: 12-BIT")
        print("█"*80)
        results_12 = tester.run_diagnostic('hybrid_freq_imec_data_12bit.pkl')
    except FileNotFoundError:
        print("\n⚠️ 12-bit file not found")
    except Exception as e:
        print(f"\n❌ Error in 12-bit test: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    print("\nCheck the generated diagnostic_*.png files for detailed visualizations!")
