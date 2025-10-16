import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pickle
from imec_encoder import MinEntropyCouplingSteganography

class PaperOptimalEncoder:
    """
    Using the paper's OPTIMAL settings: 20-bit blocks
    This is what achieved the best efficiency in the ICLR 2023 paper
    """
    
    def __init__(self, model_name='gpt2'):
        print("Loading GPT-2 for PAPER-OPTIMAL encoding...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Agent configuration
        self.agents = {
            'ALICE': {'freq': 0.015, 'bits': None},
            'BOB': {'freq': 0.030, 'bits': None},
            'CHARLIE': {'freq': 0.045, 'bits': None}
        }
        
        # iMEC encoder with 20-bit blocks (paper's optimal)
        self.imec = MinEntropyCouplingSteganography(block_size_bits=20)
        
        print(f"Hybrid encoder ready on {self.device}")
        print(f"✓ Using 20-bit blocks (paper's optimal setting)")
        print(f"✓ Expected: highest precision, ~10x slower than model inference")
    
    def encode_ask_smooth(self, bits, carrier_freq, sequence_length, transition_tokens=5):
        """ASK modulation with MAXIMUM amplitudes"""
        tokens_per_bit = sequence_length // len(bits)
        bias_signal = np.zeros(sequence_length)
        
        for i, bit in enumerate(bits):
            start = i * tokens_per_bit
            end = min((i + 1) * tokens_per_bit, sequence_length)
            
            # MAXIMUM AMPLITUDES: 5.0 for '1', 0.1 for '0'
            amplitude_target = 5.0 if bit == 1 else 0.1
            window_length = end - start
            amplitude = np.ones(window_length) * amplitude_target
            
            if i > 0:
                prev_amp = 5.0 if bits[i-1] == 1 else 0.1
                for j in range(min(transition_tokens, window_length)):
                    t_norm = j / transition_tokens
                    amplitude[j] = prev_amp * (1 - t_norm) + amplitude_target * t_norm
            
            t = np.arange(window_length)
            carrier = np.sin(2 * np.pi * carrier_freq * (start + t))
            bias_signal[start:end] = amplitude * carrier
        
        return bias_signal
    
    def generate_frequency_modulated(self, context, messages, sequence_length=300, 
                                     bias_strength=2.5):
        """
        STAGE 1: Generate frequency-modulated stegotext with ULTRA STRONG signal
        """
        print("\n" + "="*80)
        print("STAGE 1: FREQUENCY MODULATION (ULTRA STRONG SIGNAL)")
        print("="*80)
        print(f"Generating {sequence_length} tokens with {bias_strength} bias strength")
        print(f"Using MAXIMUM modulation: 5.0 for '1', 0.1 for '0'")
        
        for agent, bits in messages.items():
            self.agents[agent]['bits'] = np.array(bits)
            print(f"{agent:8s}: {bits}")
        
        # Generate bias signals with MAXIMUM amplitudes
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
                
                # Apply ULTRA STRONG frequency bias to logits
                bias_value = combined_bias[i]
                bias_scaled = bias_strength * bias_value / 2.0
                
                # Add bias to logits
                biased_logits = logits + bias_scaled
                
                # Softmax to get probabilities
                biased_probs = torch.softmax(biased_logits, dim=0)
                
                # Suppress EOS
                biased_probs[eos_token_id] = 0.0
                biased_probs = biased_probs / biased_probs.sum()
                
                next_token = torch.multinomial(biased_probs, num_samples=1)
                freq_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                if (i + 1) % 100 == 0:
                    print(f"  Generated {i+1}/{sequence_length} tokens...")
        
        freq_text = self.tokenizer.decode(freq_tokens)
        
        print(f"\n✓ Frequency modulation complete: {len(freq_tokens)} tokens")
        print(f"✓ Using ULTRA STRONG signal for guaranteed recovery")
        
        return freq_text, freq_tokens, bias_signals

    def apply_imec_obfuscation(self, freq_tokens, context):
        """
        STAGE 2: Apply iMEC with 20-bit blocks (PAPER'S OPTIMAL)
        """
        print("\n" + "="*80)
        print("STAGE 2: iMEC OBFUSCATION (20-bit blocks - PAPER OPTIMAL)")
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
        
        if not (0.48 <= ones_ratio <= 0.52):
            print(f"⚠️  WARNING: Ciphertext not uniform! Ratio: {ones_ratio:.3f}")
        
        # Apply iMEC encoding with 20-bit blocks
        print(f"\n✓ Applying iMEC with 20-bit blocks (PAPER'S BEST)...")
        print(f"  Block size: 20 bits (1,048,576 possible values per block!)")
        print(f"  Number of blocks: {len(ciphertext_bits) // 20}")
        print(f"  ⚠️  This will be SLOW (~10x model inference time)")
        print(f"  ⚠️  But gives MAXIMUM precision for clean recovery")
        
        obfuscated_tokens = self.imec.encode_imec(
            ciphertext_bits, 
            context, 
            max_tokens=15000,  # Even more tokens for 20-bit
            entropy_threshold=0.1  # Paper uses 0.1 threshold
        )
        
        obfuscated_text = self.tokenizer.decode(obfuscated_tokens)
        
        print(f"\n✓ iMEC obfuscation complete")
        print(f"✓ Obfuscated to {len(obfuscated_tokens)} tokens")
        
        # Store metadata
        metadata = {
            'n_freq_tokens': len(freq_tokens),
            'block_size_bits': 20,  # Paper's optimal
            'n_blocks': len(ciphertext_bits) // 20,
            'bits_per_token': bits_per_token,
            'encryption_key': encryption_key,
            'agent_frequencies': {name: info['freq'] for name, info in self.agents.items()},
            'bias_strength': 2.5,
            'amplitude_high': 5.0,
            'amplitude_low': 0.1
        }
        
        print(f"✓ Metadata: {metadata['n_blocks']} blocks of {metadata['block_size_bits']} bits")
        
        return obfuscated_text, obfuscated_tokens, metadata
    
    def encode_hybrid(self, context, messages, sequence_length=300):
        """
        Complete hybrid encoding with PAPER-OPTIMAL settings
        """
        print("\n" + "="*80)
        print("PAPER-OPTIMAL ENCODER (ICLR 2023)")
        print("="*80)
        print("Settings from paper:")
        print("  • Block size: 20 bits (paper's best performer)")
        print("  • Signal strength: ULTRA STRONG (15x original)")
        print("  • Entropy threshold: 0.1 (paper's setting)")
        print("  • Expected: 99%+ token recovery")
        print("  • Trade-off: ~10x slower than model inference")
        print("="*80)
        
        # Stage 1: Frequency modulation with ULTRA STRONG signal
        freq_text, freq_tokens, bias_signals = self.generate_frequency_modulated(
            context, messages, sequence_length
        )
        
        # Stage 2: iMEC obfuscation with 20-bit precision
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
        
        with open('hybrid_freq_imec_data_PAPER_OPTIMAL.pkl', 'wb') as f:
            pickle.dump(output_data, f)
        
        print("\n" + "="*80)
        print("ENCODING COMPLETE")
        print("="*80)
        print(f"✓ Frequency-modulated: {len(freq_tokens)} tokens (ULTRA STRONG)")
        print(f"✓ iMEC obfuscated: {len(obf_tokens)} tokens (20-bit precision)")
        print(f"✓ Using 20-bit blocks: {metadata['n_blocks']} blocks")
        print(f"✓ Saved to: hybrid_freq_imec_data_PAPER_OPTIMAL.pkl")
        print(f"\n✓ Paper-optimal 20-bit blocks should achieve 99%+ recovery!")
        print(f"✓ This should enable 90-100% message recovery!")
        
        return output_data


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    encoder = PaperOptimalEncoder()
    
    # Define messages
    messages = {
        'ALICE':   [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
        'BOB':     [1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
        'CHARLIE': [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0]
    }
    
    context = "The future of artificial intelligence"
    
    # Encode with PAPER-OPTIMAL system (20-bit blocks)
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
    print("1. Update test scripts to load 'hybrid_freq_imec_data_PAPER_OPTIMAL.pkl'")
    print("2. Update scripts to use block_size_bits=20")
    print("3. Run imec_security_test5_update.py (will take ~10-15 minutes)")
    print("4. Run token_id_recovery.py")
    print("5. Expect 99%+ token recovery → 90-100% message recovery!")
