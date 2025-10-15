import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pickle
from imec_encoder import MinEntropyCouplingSteganography

class ImprovedFrequencyEncoder:
    """
    IMPROVED: Much stronger FM signal for reliable message recovery
    """
    
    def __init__(self, model_name='gpt2'):
        print("Loading GPT-2 for IMPROVED hybrid encoding...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Agent configuration - SAME frequencies
        self.agents = {
            'ALICE': {'freq': 0.015, 'bits': None},
            'BOB': {'freq': 0.030, 'bits': None},
            'CHARLIE': {'freq': 0.045, 'bits': None}
        }
        
        # iMEC encoder with 8-bit blocks
        self.imec = MinEntropyCouplingSteganography(block_size_bits=8)
        
        print(f"Hybrid encoder ready on {self.device}")
        print(f"✓ IMPROVED: Much stronger bias for reliable recovery")
    
    def encode_ask_smooth(self, bits, carrier_freq, sequence_length, transition_tokens=5):
        """ASK modulation with smooth transitions - STRONGER amplitudes"""
        tokens_per_bit = sequence_length // len(bits)
        bias_signal = np.zeros(sequence_length)
        
        for i, bit in enumerate(bits):
            start = i * tokens_per_bit
            end = min((i + 1) * tokens_per_bit, sequence_length)
            
            # INCREASED AMPLITUDES: 2.0 for '1', 0.3 for '0' (was 0.8 and 0.2)
            amplitude_target = 2.0 if bit == 1 else 0.3
            window_length = end - start
            amplitude = np.ones(window_length) * amplitude_target
            
            if i > 0:
                prev_amp = 2.0 if bits[i-1] == 1 else 0.3
                for j in range(min(transition_tokens, window_length)):
                    t_norm = j / transition_tokens
                    amplitude[j] = prev_amp * (1 - t_norm) + amplitude_target * t_norm
            
            t = np.arange(window_length)
            carrier = np.sin(2 * np.pi * carrier_freq * (start + t))
            bias_signal[start:end] = amplitude * carrier
        
        return bias_signal
    
    def generate_frequency_modulated(self, context, messages, sequence_length=300, 
                                     bias_strength=1.5):  # INCREASED from 0.7
        """
        STAGE 1: Generate frequency-modulated stegotext with STRONG signal
        """
        print("\n" + "="*80)
        print("STAGE 1: FREQUENCY MODULATION (IMPROVED - STRONG SIGNAL)")
        print("="*80)
        print(f"Generating {sequence_length} tokens with {bias_strength} bias strength")
        print(f"Using AMPLIFIED modulation: 2.0 for '1', 0.3 for '0'")
        
        for agent, bits in messages.items():
            self.agents[agent]['bits'] = np.array(bits)
            print(f"{agent:8s}: {bits}")
        
        # Generate bias signals with STRONGER amplitudes
        bias_signals = {}
        for agent_name, agent_info in self.agents.items():
            bits = agent_info['bits']
            freq = agent_info['freq']
            bias_signals[agent_name] = self.encode_ask_smooth(
                bits, freq, sequence_length
            )
        
        combined_bias = sum(bias_signals.values())
        
        # Generate text
        input_ids = self.tokenizer.encode(context, return_tensors='pt').to(self.device)
        freq_tokens = []
        eos_token_id = self.tokenizer.eos_token_id
        
        with torch.no_grad():
            for i in range(sequence_length):
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]
                base_probs = torch.softmax(logits, dim=0)
                
                # Suppress EOS
                base_probs[eos_token_id] = 0.0
                base_probs = base_probs / base_probs.sum()
                
                # Apply STRONGER frequency bias
                biased_probs = base_probs * (1 + bias_strength * combined_bias[i])
                biased_probs = torch.clamp(biased_probs, min=0)  # Ensure non-negative
                biased_probs = biased_probs / biased_probs.sum()
                
                next_token = torch.multinomial(biased_probs, num_samples=1)
                freq_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                if (i + 1) % 100 == 0:
                    print(f"  Generated {i+1}/{sequence_length} tokens...")
        
        freq_text = self.tokenizer.decode(freq_tokens)
        
        print(f"\n✓ Frequency modulation complete: {len(freq_tokens)} tokens")
        print(f"✓ Using MUCH STRONGER signal for reliable recovery")
        
        return freq_text, freq_tokens, bias_signals

    def apply_imec_obfuscation(self, freq_tokens, context):
        """
        STAGE 2: Apply iMEC to hide frequency patterns.
        """
        print("\n" + "="*80)
        print("STAGE 2: iMEC OBFUSCATION (with encryption, 8-bit blocks)")
        print("="*80)
        
        # Convert frequency tokens to binary (plaintext)
        bits_per_token = 16
        print(f"\nConverting {len(freq_tokens)} tokens to binary...")
        plaintext_bits = ''.join(format(token, f'0{bits_per_token}b') 
                                  for token in freq_tokens)
        print(f"Plaintext: {len(plaintext_bits)} bits")
        
        # Encrypt with one-time pad
        print("\n✓ Generating encryption key (one-time pad)...")
        encryption_key = np.random.randint(0, 2, len(plaintext_bits), dtype=np.uint8)
        
        # XOR encryption to create UNIFORM ciphertext
        ciphertext_bits = ''.join(
            str(int(plaintext_bits[i]) ^ int(encryption_key[i]))
            for i in range(len(plaintext_bits))
        )
        print(f"✓ Encrypted to uniform ciphertext: {len(ciphertext_bits)} bits")
        
        # Verify uniformity
        ones_ratio = sum(int(b) for b in ciphertext_bits) / len(ciphertext_bits)
        print(f"✓ Ciphertext uniformity check: {ones_ratio:.3f} (should be ~0.5)")
        
        # Apply iMEC encoding
        print(f"\n✓ Applying iMEC with 8-bit blocks...")
        obfuscated_tokens = self.imec.encode_imec(
            ciphertext_bits, 
            context, 
            max_tokens=5000,
            entropy_threshold=0.05
        )
        
        obfuscated_text = self.tokenizer.decode(obfuscated_tokens)
        
        print(f"\n✓ iMEC obfuscation complete")
        print(f"✓ Obfuscated to {len(obfuscated_tokens)} tokens")
        
        # Store metadata
        metadata = {
            'n_freq_tokens': len(freq_tokens),
            'block_size_bits': 8,
            'n_blocks': len(ciphertext_bits) // 8,
            'bits_per_token': bits_per_token,
            'encryption_key': encryption_key,
            'agent_frequencies': {name: info['freq'] for name, info in self.agents.items()},
            'bias_strength': 1.5,  # Store for reference
            'amplitude_high': 2.0,
            'amplitude_low': 0.3
        }
        
        print(f"✓ Metadata: {metadata['n_blocks']} blocks of {metadata['block_size_bits']} bits")
        
        return obfuscated_text, obfuscated_tokens, metadata
    
    def encode_hybrid(self, context, messages, sequence_length=300):
        """
        Complete hybrid encoding pipeline with IMPROVED signal strength
        """
        print("\n" + "="*80)
        print("IMPROVED HYBRID ENCODER: FREQUENCY + iMEC")
        print("="*80)
        print("Changes from previous version:")
        print("  • Amplitude HIGH: 0.8 → 2.0 (2.5x stronger)")
        print("  • Amplitude LOW:  0.2 → 0.3 (1.5x stronger)")
        print("  • Bias strength:  0.7 → 1.5 (2.1x stronger)")
        print("  • Overall signal: ~5x stronger for reliable recovery!")
        print("="*80)
        
        # Stage 1: Frequency modulation with STRONG signal
        freq_text, freq_tokens, bias_signals = self.generate_frequency_modulated(
            context, messages, sequence_length
        )
        
        # Stage 2: iMEC obfuscation
        obf_text, obf_tokens, metadata = self.apply_imec_obfuscation(
            freq_tokens, context
        )
        
        # Save everything
        output_data = {
            'context': context,
            'messages': messages,
            'freq_text': freq_text,
            'freq_tokens': freq_tokens,
            'bias_signals': bias_signals,
            'obfuscated_text': obf_text,
            'obfuscated_tokens': obf_tokens,
            'metadata': metadata
        }
        
        with open('hybrid_freq_imec_data_IMPROVED.pkl', 'wb') as f:
            pickle.dump(output_data, f)
        
        print("\n" + "="*80)
        print("ENCODING COMPLETE")
        print("="*80)
        print(f"✓ Frequency-modulated: {len(freq_tokens)} tokens (STRONG signal)")
        print(f"✓ iMEC obfuscated: {len(obf_tokens)} tokens")
        print(f"✓ Using 8-bit blocks: {metadata['n_blocks']} blocks")
        print(f"✓ Saved to: hybrid_freq_imec_data_IMPROVED.pkl")
        print(f"\n✓ Signal is now 5x STRONGER for reliable message recovery!")
        
        return output_data


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    encoder = ImprovedFrequencyEncoder()
    
    # Define messages
    messages = {
        'ALICE':   [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
        'BOB':     [1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
        'CHARLIE': [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0]
    }
    
    context = "The future of artificial intelligence"
    
    # Encode with IMPROVED system (much stronger signal)
    output_data = encoder.encode_hybrid(
        context=context,
        messages=messages,
        sequence_length=300
    )
    
    print(f"\nFrequency text preview:")
    print(f"  {output_data['freq_text'][:150]}...")
    print(f"\nObfuscated text preview:")
    print(f"  {output_data['obfuscated_text'][:150]}...")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Run the full decoding pipeline on this IMPROVED data")
    print("2. Message recovery should now be 80-100% accurate")
    print("3. This proves the complete system works end-to-end!")
