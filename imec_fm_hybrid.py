import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pickle
from imec_encoder import MinEntropyCouplingSteganography

class FrequencyiMECHybridEncoder:
    """
    Complete hybrid: Frequency modulation + iMEC obfuscation
    
    Pipeline:
    1. Encode 3 messages via frequency modulation
    2. Convert frequency-modulated tokens to binary
    3. Encrypt with one-time pad → uniform ciphertext
    4. Apply iMEC with 8-bit blocks (better convergence)
    5. Decode: iMEC → decrypt → recover tokens → FFT analysis → extract messages
    """
    
    def __init__(self, model_name='gpt2'):
        print("Loading GPT-2 for hybrid encoding...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Agent configuration
        self.agents = {
            'ALICE': {'freq': 0.02, 'bits': None},
            'BOB': {'freq': 0.04, 'bits': None},
            'CHARLIE': {'freq': 0.06, 'bits': None}
        }
        
        # iMEC encoder with 8-bit blocks (256 values vs 65536)
        # This improves convergence dramatically
        self.imec = MinEntropyCouplingSteganography(block_size_bits=8)
        
        print(f"Hybrid encoder ready on {self.device}")
        print(f"Using 8-bit iMEC blocks for better convergence")
    
    def encode_ask_smooth(self, bits, carrier_freq, sequence_length, transition_tokens=5):
        """ASK modulation with smooth transitions."""
        tokens_per_bit = sequence_length // len(bits)
        bias_signal = np.zeros(sequence_length)
        
        for i, bit in enumerate(bits):
            start = i * tokens_per_bit
            end = min((i + 1) * tokens_per_bit, sequence_length)
            
            amplitude_target = 0.8 if bit == 1 else 0.2
            window_length = end - start
            amplitude = np.ones(window_length) * amplitude_target
            
            if i > 0:
                prev_amp = 0.8 if bits[i-1] == 1 else 0.2
                for j in range(min(transition_tokens, window_length)):
                    t_norm = j / transition_tokens
                    amplitude[j] = prev_amp * (1 - t_norm) + amplitude_target * t_norm
            
            t = np.arange(window_length)
            carrier = np.sin(2 * np.pi * carrier_freq * (start + t))
            bias_signal[start:end] = amplitude * carrier
        
        return bias_signal
    
    def generate_frequency_modulated(self, context, messages, sequence_length=100, 
                                     bias_strength=0.5):
        """
        STAGE 1: Generate frequency-modulated stegotext.
        """
        print("\n" + "="*80)
        print("STAGE 1: FREQUENCY MODULATION")
        print("="*80)
        
        for agent, bits in messages.items():
            self.agents[agent]['bits'] = np.array(bits)
            print(f"{agent:8s}: {bits}")
        
        # Generate bias signals
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
                
                # Apply frequency bias
                biased_probs = base_probs * (1 + bias_strength * combined_bias[i])
                biased_probs = biased_probs / biased_probs.sum()
                
                next_token = torch.multinomial(biased_probs, num_samples=1)
                freq_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                if (i + 1) % 200 == 0:
                    print(f"  Generated {i+1} tokens...")
        
        freq_text = self.tokenizer.decode(freq_tokens)
        
        print(f"\n✓ Frequency modulation complete: {len(freq_tokens)} tokens")
        
        return freq_text, freq_tokens, bias_signals

    def apply_imec_obfuscation(self, freq_tokens, context):
        """
        STAGE 2: Apply iMEC to hide frequency patterns.
        CORRECTED: Use 8-bit blocks for better convergence.
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
        
        # ========================================
        # CRITICAL: Encrypt with one-time pad
        # ========================================
        print("\n✓ Generating encryption key (one-time pad)...")
        encryption_key = np.random.randint(0, 2, len(plaintext_bits), dtype=np.uint8)
        
        # XOR encryption to create UNIFORM ciphertext
        ciphertext_bits = ''.join(
            str(int(plaintext_bits[i]) ^ int(encryption_key[i]))
            for i in range(len(plaintext_bits))
        )
        print(f"✓ Encrypted to uniform ciphertext: {len(ciphertext_bits)} bits")
        
        # Verify uniformity (should be ~50% ones)
        ones_ratio = sum(int(b) for b in ciphertext_bits) / len(ciphertext_bits)
        print(f"✓ Ciphertext uniformity check: {ones_ratio:.3f} (should be ~0.5)")
        
        if not (0.48 <= ones_ratio <= 0.52):
            print(f"⚠️  WARNING: Ciphertext not uniform! Ratio: {ones_ratio:.3f}")
        
        # Apply iMEC encoding to UNIFORM ciphertext with 8-bit blocks
        print(f"\n✓ Applying iMEC with 8-bit blocks...")
        print(f"  Block size: 8 bits (256 possible values per block)")
        print(f"  Number of blocks: {len(ciphertext_bits) // 8}")
        
        obfuscated_tokens = self.imec.encode_imec(
            ciphertext_bits, 
            context, 
            max_tokens=3000,  # Increased for 8-bit blocks (need more tokens)
            entropy_threshold=0.05  # Slightly higher threshold for faster convergence
        )
        
        obfuscated_text = self.tokenizer.decode(obfuscated_tokens)
        
        print(f"\n✓ iMEC obfuscation complete")
        print(f"✓ Obfuscated to {len(obfuscated_tokens)} tokens")
        
        # Store metadata INCLUDING encryption key and 8-bit block size
        metadata = {
            'n_freq_tokens': len(freq_tokens),
            'block_size_bits': 8,  # CHANGED from 16 to 8
            'n_blocks': len(ciphertext_bits) // 8,  # Will be 200 blocks
            'bits_per_token': bits_per_token,
            'encryption_key': encryption_key
        }
        
        print(f"✓ Metadata: {metadata['n_blocks']} blocks of {metadata['block_size_bits']} bits")
        
        return obfuscated_text, obfuscated_tokens, metadata
    
    def imec_decode_tokens(self, obfuscated_tokens, context, metadata):
        """
        STAGE 3: Decode iMEC AND decrypt to recover frequency-modulated tokens.
        """
        print("\n" + "="*80)
        print("STAGE 3: iMEC DECODING + DECRYPTION")
        print("="*80)
        
        # Step 1: iMEC decode (stegotext → ciphertext)
        print("\n✓ Applying iMEC decoding...")
        print(f"  Using {metadata['n_blocks']} blocks of {metadata['block_size_bits']} bits")
        
        recovered_ciphertext = self.imec.decode_imec(
            obfuscated_tokens,
            context,
            metadata['n_blocks'],
            metadata['block_size_bits']
        )
        print(f"✓ Recovered ciphertext: {len(recovered_ciphertext)} bits")
        
        # ========================================
        # CRITICAL: Decrypt with one-time pad
        # ========================================
        print("\n✓ Decrypting with one-time pad...")
        encryption_key = metadata['encryption_key']
        
        # XOR decryption (ciphertext → plaintext)
        min_len = min(len(recovered_ciphertext), len(encryption_key))
        recovered_plaintext = ''.join(
            str(int(recovered_ciphertext[i]) ^ int(encryption_key[i]))
            for i in range(min_len)
        )
        print(f"✓ Decrypted plaintext: {len(recovered_plaintext)} bits")
        
        # Step 2: Convert binary back to tokens
        bits_per_token = metadata.get('bits_per_token', 16)
        n_tokens = metadata['n_freq_tokens']
        recovered_tokens = []
        invalid_count = 0
        
        vocab_size = len(self.tokenizer)
        
        for i in range(n_tokens):
            start = i * bits_per_token
            end = start + bits_per_token
            
            if end <= len(recovered_plaintext):
                token_bits = recovered_plaintext[start:end]
                token_id = int(token_bits, 2)
                
                # Validate token ID
                if 0 <= token_id < vocab_size:
                    recovered_tokens.append(token_id)
                else:
                    # This indicates bit errors in iMEC recovery
                    invalid_count += 1
                    if invalid_count <= 5:
                        print(f"⚠️  Invalid token {token_id} at position {i} (vocab size: {vocab_size})")
                    # Use modulo to wrap into valid range
                    recovered_tokens.append(token_id % vocab_size)
            else:
                print(f"⚠️  Insufficient bits for token {i}")
                break
        
        if invalid_count > 0:
            print(f"\n⚠️  WARNING: {invalid_count}/{n_tokens} tokens were out of range!")
            print(f"    Using modulo fallback for invalid tokens.")
        
        print(f"\n✓ Recovered {len(recovered_tokens)} tokens")
        
        return recovered_tokens
    
    def encode_hybrid(self, context, messages, sequence_length=100):
        """
        Complete hybrid encoding pipeline.
        """
        print("\n" + "="*80)
        print("HYBRID ENCODER: FREQUENCY + iMEC (8-bit blocks)")
        print("="*80)
        
        # Stage 1: Frequency modulation
        freq_text, freq_tokens, bias_signals = self.generate_frequency_modulated(
            context, messages, sequence_length
        )
        
        # Stage 2: iMEC obfuscation with 8-bit blocks
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
        
        with open('hybrid_freq_imec_data.pkl', 'wb') as f:
            pickle.dump(output_data, f)
        
        print("\n" + "="*80)
        print("ENCODING COMPLETE")
        print("="*80)
        print(f"✓ Frequency-modulated: {len(freq_tokens)} tokens")
        print(f"✓ iMEC obfuscated: {len(obf_tokens)} tokens")
        print(f"✓ Using 8-bit blocks: {metadata['n_blocks']} blocks")
        print(f"✓ Saved to: hybrid_freq_imec_data.pkl")
        
        return output_data


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    encoder = FrequencyiMECHybridEncoder()
    
    # Define messages
    messages = {
        'ALICE':   [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
        'BOB':     [1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
        'CHARLIE': [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0]
    }
    
    context = "The future of artificial intelligence"
    
    # Encode with hybrid system (8-bit blocks)
    output_data = encoder.encode_hybrid(
        context=context,
        messages=messages,
        sequence_length=100
    )
    
    print(f"\nFrequency text preview:")
    print(f"  {output_data['freq_text'][:150]}...")
    print(f"\nObfuscated text preview:")
    print(f"  {output_data['obfuscated_text'][:150]}...")
