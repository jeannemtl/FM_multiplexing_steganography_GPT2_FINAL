import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
from imec_encoder import MinEntropyCouplingSteganography

class FrequencyRecoveryVisualization:
    """
    Show that frequency-encoded messages are:
    - VISIBLE in original tokens
    - INVISIBLE without decryption key
    - VISIBLE WITH decryption key
    """
    
    def __init__(self):
        print("Loading GPT-2 for frequency recovery visualization...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.agents = {
            'ALICE': {'freq': 0.02, 'color': 'blue'},
            'BOB': {'freq': 0.04, 'color': 'green'},
            'CHARLIE': {'freq': 0.06, 'color': 'red'}
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
    
    def bandpass_filter(self, signal, center_freq, bandwidth=0.015):
        """Apply bandpass filter."""
        fs = 1.0
        nyquist = fs / 2
        low = max(0.001, min((center_freq - bandwidth/2) / nyquist, 0.999))
        high = max(0.001, min((center_freq + bandwidth/2) / nyquist, 0.999))
        
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        
        return filtered
    
    def decode_ask_bits(self, filtered_signal, n_bits):
        """Decode ASK bits using K-means clustering."""
        signal_length = len(filtered_signal)
        tokens_per_bit = signal_length // n_bits
        
        amplitudes = []
        for i in range(n_bits):
            start = i * tokens_per_bit
            end = min((i + 1) * tokens_per_bit, signal_length)
            window = filtered_signal[start:end]
            rms_amplitude = np.sqrt(np.mean(window**2))
            amplitudes.append(rms_amplitude)
        
        amplitudes = np.array(amplitudes)
        
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(amplitudes.reshape(-1, 1))
        
        centers = kmeans.cluster_centers_.flatten()
        if centers[0] > centers[1]:
            bits = 1 - labels
        else:
            bits = labels
        
        return bits
    
    def decode_messages_from_tokens(self, token_ids, context, original_messages):
        """Decode all three agent messages from token sequence."""
        
        # Extract entropy
        entropy_seq = self.extract_entropy(token_ids, context)
        
        if len(entropy_seq) == 0:
            return None
        
        # Normalize
        normalized = (entropy_seq - np.mean(entropy_seq)) / (np.std(entropy_seq) + 1e-10)
        
        results = {}
        
        for agent_name, agent_info in self.agents.items():
            freq = agent_info['freq']
            
            # Filter at agent's frequency
            filtered = self.bandpass_filter(normalized, freq)
            
            # Decode bits
            n_bits = len(original_messages[agent_name])
            decoded_bits = self.decode_ask_bits(filtered, n_bits)
            
            # Compare to original
            original_bits = np.array(original_messages[agent_name])
            matches = np.sum(decoded_bits == original_bits)
            accuracy = matches / len(original_bits) * 100
            ber = 100 - accuracy
            
            results[agent_name] = {
                'decoded_bits': decoded_bits,
                'original_bits': original_bits,
                'accuracy': accuracy,
                'ber': ber
            }
        
        return results
    
    def create_visualization(self, data_file='hybrid_freq_imec_data.pkl'):
        """Create comprehensive frequency recovery visualization."""
        
        print("\n" + "="*80)
        print("FREQUENCY MESSAGE RECOVERY VISUALIZATION")
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
        print(f"  Original messages:")
        for agent, bits in messages.items():
            print(f"    {agent}: {bits}")
        
        # ========================================
        # SCENARIO 1: Original tokens
        # ========================================
        print("\n✓ Decoding from ORIGINAL tokens...")
        original_decoded = self.decode_messages_from_tokens(freq_tokens, context, messages)
        
        # ========================================
        # SCENARIO 2: WITHOUT key
        # ========================================
        print("✓ Decoding WITHOUT encryption key...")
        
        # Initialize iMEC
        block_size = metadata['block_size_bits']
        self.imec = MinEntropyCouplingSteganography(block_size_bits=block_size)
        
        # Decode iMEC
        recovered_ciphertext = self.imec.decode_imec(
            obf_tokens, context, metadata['n_blocks'], metadata['block_size_bits']
        )
        
        # Convert to tokens WITHOUT decryption
        bits_per_token = metadata['bits_per_token']
        n_tokens = metadata['n_freq_tokens']
        vocab_size = len(self.tokenizer)
        
        tokens_no_key = []
        for i in range(n_tokens):
            start = i * bits_per_token
            end = start + bits_per_token
            if end <= len(recovered_ciphertext):
                token_bits = recovered_ciphertext[start:end]
                token_id = int(token_bits, 2) % vocab_size
                tokens_no_key.append(token_id)
        
        nokey_decoded = self.decode_messages_from_tokens(tokens_no_key, context, messages)
        
        # ========================================
        # SCENARIO 3: WITH key
        # ========================================
        print("✓ Decoding WITH encryption key...")
        
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
                token_id = int(token_bits, 2) % vocab_size
                tokens_with_key.append(token_id)
        
        withkey_decoded = self.decode_messages_from_tokens(tokens_with_key, context, messages)
        
        # ========================================
        # CREATE VISUALIZATION
        # ========================================
        print("\n✓ Creating visualization...")
        
        fig = plt.figure(figsize=(20, 12))
        
        # ========================================
        # ROW 1: Bit comparison for each agent
        # ========================================
        
        for idx, agent_name in enumerate(['ALICE', 'BOB', 'CHARLIE']):
            ax = plt.subplot(3, 4, idx + 1)
            
            original_bits = original_decoded[agent_name]['original_bits']
            n_bits = len(original_bits)
            x = np.arange(n_bits)
            
            # Plot as binary signals
            ax.step(x, original_bits, where='post', linewidth=3, 
                   color='black', label='Original Message')
            ax.step(x, original_decoded[agent_name]['decoded_bits'], 
                   where='post', linewidth=2, linestyle='--',
                   color='blue', alpha=0.7, label='Decoded from Original')
            
            ax.set_ylim(-0.2, 1.3)
            ax.set_xlabel('Bit Position', fontsize=11, fontweight='bold')
            ax.set_ylabel('Bit Value', fontsize=11, fontweight='bold')
            ax.set_title(f'{agent_name}: Original Tokens\n'
                        f'Accuracy: {original_decoded[agent_name]["accuracy"]:.1f}%',
                        fontsize=12, fontweight='bold', color='blue')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_yticks([0, 1])
        
        # ========================================
        # ROW 2: WITHOUT key
        # ========================================
        
        for idx, agent_name in enumerate(['ALICE', 'BOB', 'CHARLIE']):
            ax = plt.subplot(3, 4, idx + 5)
            
            original_bits = nokey_decoded[agent_name]['original_bits']
            decoded_bits = nokey_decoded[agent_name]['decoded_bits']
            n_bits = len(original_bits)
            x = np.arange(n_bits)
            
            # Plot as binary signals
            ax.step(x, original_bits, where='post', linewidth=3, 
                   color='black', label='Original Message')
            ax.step(x, decoded_bits, where='post', linewidth=2, 
                   linestyle='--', color='red', alpha=0.7, 
                   label='Decoded WITHOUT Key')
            
            ax.set_ylim(-0.2, 1.3)
            ax.set_xlabel('Bit Position', fontsize=11, fontweight='bold')
            ax.set_ylabel('Bit Value', fontsize=11, fontweight='bold')
            ax.set_title(f'{agent_name}: WITHOUT Key\n'
                        f'Accuracy: {nokey_decoded[agent_name]["accuracy"]:.1f}% (BER: {nokey_decoded[agent_name]["ber"]:.1f}%)',
                        fontsize=12, fontweight='bold', color='red')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_yticks([0, 1])
            
            # Add "RANDOM" annotation if BER near 50%
            if 40 < nokey_decoded[agent_name]["ber"] < 60:
                ax.text(n_bits/2, 0.5, 'RANDOM\nNOISE', 
                       fontsize=14, fontweight='bold', color='red',
                       ha='center', va='center', alpha=0.5,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # ========================================
        # ROW 3: WITH key
        # ========================================
        
        for idx, agent_name in enumerate(['ALICE', 'BOB', 'CHARLIE']):
            ax = plt.subplot(3, 4, idx + 9)
            
            original_bits = withkey_decoded[agent_name]['original_bits']
            decoded_bits = withkey_decoded[agent_name]['decoded_bits']
            n_bits = len(original_bits)
            x = np.arange(n_bits)
            
            # Plot as binary signals
            ax.step(x, original_bits, where='post', linewidth=3, 
                   color='black', label='Original Message')
            ax.step(x, decoded_bits, where='post', linewidth=2, 
                   linestyle='--', color='green', alpha=0.7, 
                   label='Decoded WITH Key')
            
            ax.set_ylim(-0.2, 1.3)
            ax.set_xlabel('Bit Position', fontsize=11, fontweight='bold')
            ax.set_ylabel('Bit Value', fontsize=11, fontweight='bold')
            ax.set_title(f'{agent_name}: WITH Key\n'
                        f'Accuracy: {withkey_decoded[agent_name]["accuracy"]:.1f}% (BER: {withkey_decoded[agent_name]["ber"]:.1f}%)',
                        fontsize=12, fontweight='bold', color='green')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_yticks([0, 1])
            
            # Add "RECOVERED" annotation if BER < 30%
            if withkey_decoded[agent_name]["ber"] < 30:
                ax.text(n_bits/2, 0.5, 'MESSAGE\nRECOVERED', 
                       fontsize=12, fontweight='bold', color='green',
                       ha='center', va='center', alpha=0.5,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # ========================================
        # COLUMN 4: Summary
        # ========================================
        
        ax_summary = plt.subplot(3, 4, 4)
        ax_summary.axis('off')
        
        summary_text = f"""
SCENARIO 1:
ORIGINAL TOKENS

✓ Messages embedded via
  frequency modulation
  
Decoding Results:
  ALICE: {original_decoded['ALICE']['accuracy']:.0f}%
  BOB: {original_decoded['BOB']['accuracy']:.0f}%
  CHARLIE: {original_decoded['CHARLIE']['accuracy']:.0f}%

━━━━━━━━━━━━━━━━━━━━━━
Messages are VISIBLE
        """
        
        ax_summary.text(0.5, 0.5, summary_text,
                       transform=ax_summary.transAxes,
                       fontsize=11, family='monospace',
                       verticalalignment='center',
                       horizontalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        ax_summary2 = plt.subplot(3, 4, 8)
        ax_summary2.axis('off')
        
        summary_text2 = f"""
SCENARIO 2:
WITHOUT KEY

❌ iMEC decoded but
   NOT decrypted
   
Decoding Results:
  ALICE: {nokey_decoded['ALICE']['accuracy']:.0f}%
  BOB: {nokey_decoded['BOB']['accuracy']:.0f}%
  CHARLIE: {nokey_decoded['CHARLIE']['accuracy']:.0f}%

BER ≈ 50% = RANDOM

━━━━━━━━━━━━━━━━━━━━━━
Messages are HIDDEN
        """
        
        ax_summary2.text(0.5, 0.5, summary_text2,
                        transform=ax_summary2.transAxes,
                        fontsize=11, family='monospace',
                        verticalalignment='center',
                        horizontalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        
        ax_summary3 = plt.subplot(3, 4, 12)
        ax_summary3.axis('off')
        
        summary_text3 = f"""
SCENARIO 3:
WITH KEY

✅ iMEC decoded AND
   decrypted with key
   
Decoding Results:
  ALICE: {withkey_decoded['ALICE']['accuracy']:.0f}%
  BOB: {withkey_decoded['BOB']['accuracy']:.0f}%
  CHARLIE: {withkey_decoded['CHARLIE']['accuracy']:.0f}%

BER < 50% = RECOVERED

━━━━━━━━━━━━━━━━━━━━━━
Messages are VISIBLE
        """
        
        ax_summary3.text(0.5, 0.5, summary_text3,
                        transform=ax_summary3.transAxes,
                        fontsize=11, family='monospace',
                        verticalalignment='center',
                        horizontalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        plt.suptitle('FREQUENCY-ENCODED MESSAGE RECOVERY\n'
                    'Can You Decode the Hidden Messages?',
                    fontsize=18, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save
        output_file = 'message_recovery_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {output_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(f"\nORIGINAL (before encryption):")
        for agent in ['ALICE', 'BOB', 'CHARLIE']:
            print(f"  {agent}: {original_decoded[agent]['accuracy']:.1f}% accuracy")
        
        print(f"\nWITHOUT KEY (encrypted, no decryption):")
        for agent in ['ALICE', 'BOB', 'CHARLIE']:
            print(f"  {agent}: {nokey_decoded[agent]['ber']:.1f}% BER", end="")
            if 40 < nokey_decoded[agent]['ber'] < 60:
                print(" ← RANDOM (hidden ✓)")
            else:
                print()
        
        print(f"\nWITH KEY (encrypted + decrypted):")
        for agent in ['ALICE', 'BOB', 'CHARLIE']:
            print(f"  {agent}: {withkey_decoded[agent]['accuracy']:.1f}% accuracy", end="")
            if withkey_decoded[agent]['accuracy'] > 60:
                print(" ← RECOVERED (✓)")
            else:
                print()
        
        return fig


if __name__ == "__main__":
    viz = FrequencyRecoveryVisualization()
    viz.create_visualization('hybrid_freq_imec_data.pkl')
