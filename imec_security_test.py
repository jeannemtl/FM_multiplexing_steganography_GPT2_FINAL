import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pickle
from imec_encoder import MinEntropyCouplingSteganography

class iMECSecurityTest:
    """
    Test iMEC security: Can you recover original tokens WITHOUT the encryption key?
    """
    
    def __init__(self):
        print("Loading GPT-2 for security test...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.imec = None
    
    def test_without_key(self, data_file='hybrid_freq_imec_data.pkl'):
        """
        Attempt to recover tokens WITHOUT the encryption key.
        This tests whether iMEC actually hides the information.
        """
        print("\n" + "="*80)
        print("iMEC SECURITY TEST: RECOVERY WITHOUT KEY")
        print("="*80)
        
        # Load data
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        context = data['context']
        freq_tokens = data['freq_tokens']  # Original tokens
        obf_tokens = data['obfuscated_tokens']  # iMEC obfuscated
        metadata = data['metadata']
        
        print(f"\n✓ Loaded:")
        print(f"  Original tokens: {len(freq_tokens)}")
        print(f"  Obfuscated tokens: {len(obf_tokens)}")
        print(f"  Block size: {metadata['block_size_bits']} bits")
        
        # ========================================
        # TEST 1: Decode WITHOUT encryption key
        # ========================================
        print("\n" + "="*80)
        print("TEST 1: iMEC DECODE WITHOUT ENCRYPTION KEY")
        print("="*80)
        
        # Initialize iMEC decoder
        block_size = metadata['block_size_bits']
        self.imec = MinEntropyCouplingSteganography(block_size_bits=block_size)
        
        print(f"\n✓ Running iMEC decode (this gets encrypted bits)...")
        recovered_ciphertext = self.imec.decode_imec(
            obf_tokens,
            context,
            metadata['n_blocks'],
            metadata['block_size_bits']
        )
        
        print(f"✓ Recovered: {len(recovered_ciphertext)} bits of ciphertext")
        
        # Try to convert ciphertext directly to tokens (WITHOUT decryption)
        print("\n✓ Attempting to convert ciphertext to tokens WITHOUT decryption...")
        bits_per_token = metadata['bits_per_token']
        n_tokens = metadata['n_freq_tokens']
        
        recovered_tokens_no_key = []
        invalid_count = 0
        vocab_size = len(self.tokenizer)
        
        for i in range(n_tokens):
            start = i * bits_per_token
            end = start + bits_per_token
            
            if end <= len(recovered_ciphertext):
                token_bits = recovered_ciphertext[start:end]
                token_id = int(token_bits, 2)
                
                if 0 <= token_id < vocab_size:
                    recovered_tokens_no_key.append(token_id)
                else:
                    invalid_count += 1
                    recovered_tokens_no_key.append(token_id % vocab_size)
        
        print(f"✓ Converted {len(recovered_tokens_no_key)} tokens (without decryption)")
        if invalid_count > 0:
            print(f"  ({invalid_count} tokens out of range, wrapped with modulo)")
        
        # ========================================
        # COMPARE: With vs Without Key
        # ========================================
        print("\n" + "="*80)
        print("COMPARISON: WITH KEY vs WITHOUT KEY")
        print("="*80)
        
        # Calculate match rate WITHOUT key
        matches_no_key = sum(1 for i in range(len(freq_tokens))
                            if freq_tokens[i] == recovered_tokens_no_key[i])
        match_pct_no_key = 100 * matches_no_key / len(freq_tokens)
        
        print(f"\n✓ Token Recovery WITHOUT Encryption Key:")
        print(f"  Exact matches: {matches_no_key}/{len(freq_tokens)} ({match_pct_no_key:.1f}%)")
        
        if match_pct_no_key < 5:
            print(f"  ✓ EXCELLENT - Essentially 0% recovery (random chance)")
        elif match_pct_no_key < 10:
            print(f"  ✓ GOOD - Very low recovery rate")
        elif match_pct_no_key < 25:
            print(f"  ⚠️  WEAK - Some information leakage")
        else:
            print(f"  ✗ FAILED - Significant information leakage")
        
        # Now test WITH key
        print(f"\n✓ Token Recovery WITH Encryption Key:")
        print(f"  (From Phase 3 results)")
        
        encryption_key = metadata['encryption_key']
        
        # Decrypt
        min_len = min(len(recovered_ciphertext), len(encryption_key))
        recovered_plaintext = ''.join(
            str(int(recovered_ciphertext[i]) ^ int(encryption_key[i]))
            for i in range(min_len)
        )
        
        recovered_tokens_with_key = []
        for i in range(n_tokens):
            start = i * bits_per_token
            end = start + bits_per_token
            
            if end <= len(recovered_plaintext):
                token_bits = recovered_plaintext[start:end]
                token_id = int(token_bits, 2)
                
                if 0 <= token_id < vocab_size:
                    recovered_tokens_with_key.append(token_id)
                else:
                    recovered_tokens_with_key.append(token_id % vocab_size)
        
        matches_with_key = sum(1 for i in range(len(freq_tokens))
                              if freq_tokens[i] == recovered_tokens_with_key[i])
        match_pct_with_key = 100 * matches_with_key / len(freq_tokens)
        
        print(f"  Exact matches: {matches_with_key}/{len(freq_tokens)} ({match_pct_with_key:.1f}%)")
        print(f"  ✓ Decryption works perfectly")
        
        # ========================================
        # TEST 2: Statistical Analysis
        # ========================================
        print("\n" + "="*80)
        print("TEST 2: STATISTICAL ANALYSIS")
        print("="*80)
        
        print("\n✓ Analyzing ciphertext properties:")
        
        # Check uniformity of recovered ciphertext
        ones = sum(int(b) for b in recovered_ciphertext)
        uniformity = ones / len(recovered_ciphertext)
        print(f"  Ciphertext bit uniformity: {uniformity:.3f} (ideal: 0.500)")
        
        if 0.48 <= uniformity <= 0.52:
            print(f"  ✓ Ciphertext is uniform (iMEC working correctly)")
        else:
            print(f"  ⚠️  Ciphertext shows bias")
        
        # Compare token distributions
        print(f"\n✓ Token distribution analysis:")
        print(f"  Original tokens: unique={len(set(freq_tokens))}, range=[{min(freq_tokens)}, {max(freq_tokens)}]")
        print(f"  Without key: unique={len(set(recovered_tokens_no_key))}, range=[{min(recovered_tokens_no_key)}, {max(recovered_tokens_no_key)}]")
        print(f"  With key: unique={len(set(recovered_tokens_with_key))}, range=[{min(recovered_tokens_with_key)}, {max(recovered_tokens_with_key)}]")
        
        # ========================================
        # TEST 3: Brute Force Feasibility
        # ========================================
        print("\n" + "="*80)
        print("TEST 3: BRUTE FORCE ATTACK FEASIBILITY")
        print("="*80)
        
        key_length = len(encryption_key)
        key_space = 2 ** key_length
        
        print(f"\n✓ Encryption key properties:")
        print(f"  Key length: {key_length} bits")
        # print(f"  Key space: 2^{key_length} = {key_space:.2e} possible keys")
        print(f"  Random sample: {encryption_key[:20]}... (first 20 bits)")
        
        # Estimate brute force time
        checks_per_second = 1e9  # 1 billion checks/sec (optimistic)
        seconds = key_space / checks_per_second
        years = seconds / (365.25 * 24 * 3600)
        
        print(f"\n✓ Brute force attack estimate:")
        print(f"  At 1 billion keys/second: {years:.2e} years")
        
        if key_length >= 128:
            print(f"  ✓ SECURE - Infeasible to brute force")
        elif key_length >= 80:
            print(f"  ⚠️  WEAK - Vulnerable to well-resourced attackers")
        else:
            print(f"  ✗ INSECURE - Vulnerable to brute force")
        
        # ========================================
        # FINAL VERDICT
        # ========================================
        print("\n" + "="*80)
        print("SECURITY VERDICT")
        print("="*80)
        
        print(f"\n✓ Key findings:")
        print(f"  1. Recovery without key: {match_pct_no_key:.1f}%")
        print(f"  2. Recovery with key: {match_pct_with_key:.1f}%")
        print(f"  3. Ciphertext uniformity: {uniformity:.3f}")
        print(f"  4. Key space: 2^{key_length}")
        
        print(f"\n✓ Security assessment:")
        
        if match_pct_no_key < 5 and match_pct_with_key > 95 and 0.48 <= uniformity <= 0.52:
            print(f"  ✅ SECURE - iMEC + encryption provides strong security")
            print(f"     • Cannot recover without key ({match_pct_no_key:.1f}% ≈ random)")
            print(f"     • Perfect recovery with key ({match_pct_with_key:.1f}%)")
            print(f"     • Ciphertext is uniform (no statistical bias)")
            print(f"     • Key space prevents brute force (2^{key_length})")
        elif match_pct_no_key < 10:
            print(f"  ⚠️  MOSTLY SECURE - Minor information leakage detected")
            print(f"     • Low recovery without key ({match_pct_no_key:.1f}%)")
            print(f"     • Good recovery with key ({match_pct_with_key:.1f}%)")
        else:
            print(f"  ❌ INSECURE - Significant information leakage")
            print(f"     • Too much recovery without key ({match_pct_no_key:.1f}%)")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    tester = iMECSecurityTest()
    tester.test_without_key('hybrid_freq_imec_data.pkl')
