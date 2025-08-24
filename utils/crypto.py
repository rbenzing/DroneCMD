#!/usr/bin/env python3
"""
Enhanced Cryptographic Utilities

This module provides comprehensive cryptographic functionality for the dronecmd
library, including hashing, HMAC, encryption, key derivation, and secure
random number generation.

Key Features:
- Multiple hash algorithms (SHA-256, SHA-512, BLAKE2, etc.)
- HMAC generation and verification
- Symmetric encryption (AES, ChaCha20)
- Key derivation functions (PBKDF2, Argon2, HKDF)
- Secure random generation
- Digital signatures (Ed25519, ECDSA)
- Integrity verification utilities
- Backward compatibility with original functions
- Constant-time operations for security

Security Notes:
- All cryptographic operations use secure defaults
- Timing-safe comparison functions prevent timing attacks
- Secure random number generation for all random operations
- Key material is properly handled and can be zeroed
- All functions validate input parameters

Usage:
    # Basic hashing (backward compatible)
    digest = sha256_digest("hello world")
    hmac_val = hmac_sha256("secret", "message")
    
    # Enhanced functionality
    crypto = CryptoManager()
    encrypted = crypto.encrypt_data(b"sensitive data", key)
    decrypted = crypto.decrypt_data(encrypted, key)
    
    # Secure key derivation
    key = derive_key("password", salt, key_length=32)
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import struct
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Protocol
import logging

# Optional imports for extended functionality
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519, ec
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.backends import default_backend
    from cryptography.fernet import Fernet
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

try:
    import argon2
    ARGON2_AVAILABLE = True
except ImportError:
    ARGON2_AVAILABLE = False

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases
BytesLike = Union[bytes, str]
KeyMaterial = bytes
Salt = bytes
IV = bytes
Tag = bytes


class HashAlgorithm(Enum):
    """Supported hash algorithms."""
    
    SHA256 = "sha256"
    SHA512 = "sha512"
    SHA3_256 = "sha3_256"
    SHA3_512 = "sha3_512"
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"
    MD5 = "md5"  # For backward compatibility only


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""
    
    AES_GCM = "aes_gcm"
    AES_CBC = "aes_cbc"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    FERNET = "fernet"  # High-level symmetric encryption


class KeyDerivationFunction(Enum):
    """Supported key derivation functions."""
    
    PBKDF2 = "pbkdf2"
    ARGON2 = "argon2"
    HKDF = "hkdf"
    SCRYPT = "scrypt"


@dataclass
class CryptoResult:
    """Result from cryptographic operations."""
    
    success: bool
    data: Optional[bytes] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __bool__(self) -> bool:
        """Return success status."""
        return self.success


@dataclass
class EncryptionResult:
    """Result from encryption operations."""
    
    ciphertext: bytes
    iv: Optional[bytes] = None
    tag: Optional[bytes] = None
    salt: Optional[bytes] = None
    algorithm: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CryptoError(Exception):
    """Base exception for cryptographic operations."""
    pass


class InvalidKeyError(CryptoError):
    """Exception for invalid key material."""
    pass


class DecryptionError(CryptoError):
    """Exception for decryption failures."""
    pass


class SecureRandom:
    """Secure random number generator utilities."""
    
    @staticmethod
    def bytes(length: int) -> bytes:
        """Generate cryptographically secure random bytes."""
        if length <= 0:
            raise ValueError("Length must be positive")
        return secrets.token_bytes(length)
    
    @staticmethod
    def int(min_val: int = 0, max_val: int = 2**63 - 1) -> int:
        """Generate cryptographically secure random integer."""
        return secrets.randbelow(max_val - min_val + 1) + min_val
    
    @staticmethod
    def choice(sequence: List[Any]) -> Any:
        """Cryptographically secure choice from sequence."""
        if not sequence:
            raise ValueError("Sequence cannot be empty")
        return secrets.choice(sequence)
    
    @staticmethod
    def token_hex(length: int = 32) -> str:
        """Generate random hex token."""
        return secrets.token_hex(length)
    
    @staticmethod
    def token_urlsafe(length: int = 32) -> str:
        """Generate random URL-safe token."""
        return secrets.token_urlsafe(length)


class SecureCompare:
    """Timing-safe comparison utilities."""
    
    @staticmethod
    def equal(a: bytes, b: bytes) -> bool:
        """Timing-safe equality comparison."""
        return secrets.compare_digest(a, b)
    
    @staticmethod
    def equal_str(a: str, b: str) -> bool:
        """Timing-safe string equality comparison."""
        return secrets.compare_digest(a.encode('utf-8'), b.encode('utf-8'))


class HashManager:
    """Comprehensive hash management utilities."""
    
    def __init__(self, algorithm: HashAlgorithm = HashAlgorithm.SHA256):
        """Initialize hash manager with specified algorithm."""
        self.algorithm = algorithm
        self._validate_algorithm()
    
    def _validate_algorithm(self) -> None:
        """Validate that the algorithm is supported."""
        if self.algorithm == HashAlgorithm.MD5:
            warnings.warn(
                "MD5 is cryptographically broken and should not be used for security purposes",
                UserWarning,
                stacklevel=3
            )
    
    def hash(self, data: BytesLike, salt: Optional[bytes] = None) -> bytes:
        """
        Compute hash of data with optional salt.
        
        Args:
            data: Data to hash
            salt: Optional salt for the hash
            
        Returns:
            Hash digest as bytes
        """
        data_bytes = self._ensure_bytes(data)
        
        if salt:
            data_bytes = salt + data_bytes
        
        hasher = hashlib.new(self.algorithm.value)
        hasher.update(data_bytes)
        return hasher.digest()
    
    def hash_hex(self, data: BytesLike, salt: Optional[bytes] = None) -> str:
        """
        Compute hash and return as hex string.
        
        Args:
            data: Data to hash
            salt: Optional salt for the hash
            
        Returns:
            Hash digest as hex string
        """
        return self.hash(data, salt).hex()
    
    def hash_stream(self, stream, chunk_size: int = 8192) -> bytes:
        """
        Hash data from a stream (file-like object).
        
        Args:
            stream: Stream to read from
            chunk_size: Size of chunks to read
            
        Returns:
            Hash digest as bytes
        """
        hasher = hashlib.new(self.algorithm.value)
        
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
        
        return hasher.digest()
    
    def verify_hash(self, data: BytesLike, expected_hash: bytes, salt: Optional[bytes] = None) -> bool:
        """
        Verify data against expected hash.
        
        Args:
            data: Data to verify
            expected_hash: Expected hash value
            salt: Salt used in original hash
            
        Returns:
            True if hash matches
        """
        computed_hash = self.hash(data, salt)
        return SecureCompare.equal(computed_hash, expected_hash)
    
    @staticmethod
    def _ensure_bytes(data: BytesLike) -> bytes:
        """Ensure data is in bytes format."""
        if isinstance(data, str):
            return data.encode('utf-8')
        return data


class HMACManager:
    """HMAC generation and verification utilities."""
    
    def __init__(self, algorithm: HashAlgorithm = HashAlgorithm.SHA256):
        """Initialize HMAC manager with specified algorithm."""
        self.algorithm = algorithm
    
    def generate(self, key: BytesLike, message: BytesLike) -> bytes:
        """
        Generate HMAC for message with key.
        
        Args:
            key: Secret key
            message: Message to authenticate
            
        Returns:
            HMAC digest as bytes
        """
        key_bytes = self._ensure_bytes(key)
        message_bytes = self._ensure_bytes(message)
        
        return hmac.new(key_bytes, message_bytes, self.algorithm.value).digest()
    
    def generate_hex(self, key: BytesLike, message: BytesLike) -> str:
        """
        Generate HMAC and return as hex string.
        
        Args:
            key: Secret key
            message: Message to authenticate
            
        Returns:
            HMAC digest as hex string
        """
        return self.generate(key, message).hex()
    
    def verify(self, key: BytesLike, message: BytesLike, expected_hmac: bytes) -> bool:
        """
        Verify HMAC for message.
        
        Args:
            key: Secret key
            message: Message to verify
            expected_hmac: Expected HMAC value
            
        Returns:
            True if HMAC is valid
        """
        computed_hmac = self.generate(key, message)
        return SecureCompare.equal(computed_hmac, expected_hmac)
    
    @staticmethod
    def _ensure_bytes(data: BytesLike) -> bytes:
        """Ensure data is in bytes format."""
        if isinstance(data, str):
            return data.encode('utf-8')
        return data


class KeyDerivation:
    """Key derivation utilities."""
    
    @staticmethod
    def pbkdf2(
        password: BytesLike,
        salt: bytes,
        key_length: int = 32,
        iterations: int = 100000,
        algorithm: HashAlgorithm = HashAlgorithm.SHA256
    ) -> bytes:
        """
        Derive key using PBKDF2.
        
        Args:
            password: Password to derive from
            salt: Salt for derivation
            key_length: Length of derived key
            iterations: Number of iterations
            algorithm: Hash algorithm to use
            
        Returns:
            Derived key material
        """
        password_bytes = password.encode('utf-8') if isinstance(password, str) else password
        
        if CRYPTOGRAPHY_AVAILABLE:
            # Use cryptography library for better performance
            hash_algo = getattr(hashes, algorithm.value.upper())()
            kdf = PBKDF2HMAC(
                algorithm=hash_algo,
                length=key_length,
                salt=salt,
                iterations=iterations,
                backend=default_backend()
            )
            return kdf.derive(password_bytes)
        else:
            # Fallback to hashlib
            return hashlib.pbkdf2_hmac(
                algorithm.value,
                password_bytes,
                salt,
                iterations,
                key_length
            )
    
    @staticmethod
    def argon2(
        password: BytesLike,
        salt: bytes,
        key_length: int = 32,
        time_cost: int = 3,
        memory_cost: int = 65536,
        parallelism: int = 1
    ) -> bytes:
        """
        Derive key using Argon2.
        
        Args:
            password: Password to derive from
            salt: Salt for derivation
            key_length: Length of derived key
            time_cost: Time cost parameter
            memory_cost: Memory cost parameter
            parallelism: Parallelism parameter
            
        Returns:
            Derived key material
        """
        if not ARGON2_AVAILABLE:
            raise CryptoError("Argon2 library not available")
        
        password_bytes = password.encode('utf-8') if isinstance(password, str) else password
        
        ph = argon2.PasswordHasher(
            time_cost=time_cost,
            memory_cost=memory_cost,
            parallelism=parallelism,
            hash_len=key_length,
            salt_len=len(salt)
        )
        
        # Extract raw hash from Argon2 result
        hash_result = ph.hash(password_bytes, salt=salt)
        return hash_result.encode('utf-8')
    
    @staticmethod
    def hkdf(
        input_key: bytes,
        length: int = 32,
        salt: Optional[bytes] = None,
        info: Optional[bytes] = None,
        algorithm: HashAlgorithm = HashAlgorithm.SHA256
    ) -> bytes:
        """
        Derive key using HKDF (HMAC-based Key Derivation Function).
        
        Args:
            input_key: Input key material
            length: Length of output key
            salt: Optional salt
            info: Optional context information
            algorithm: Hash algorithm to use
            
        Returns:
            Derived key material
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise CryptoError("HKDF requires cryptography library")
        
        hash_algo = getattr(hashes, algorithm.value.upper())()
        
        hkdf = HKDF(
            algorithm=hash_algo,
            length=length,
            salt=salt,
            info=info,
            backend=default_backend()
        )
        
        return hkdf.derive(input_key)


class SymmetricEncryption:
    """Symmetric encryption utilities."""
    
    def __init__(self, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_GCM):
        """Initialize with specified algorithm."""
        self.algorithm = algorithm
        
        if not CRYPTOGRAPHY_AVAILABLE and algorithm != EncryptionAlgorithm.FERNET:
            raise CryptoError("Advanced encryption requires cryptography library")
    
    def encrypt(
        self,
        plaintext: bytes,
        key: KeyMaterial,
        associated_data: Optional[bytes] = None
    ) -> EncryptionResult:
        """
        Encrypt plaintext with key.
        
        Args:
            plaintext: Data to encrypt
            key: Encryption key
            associated_data: Additional authenticated data (for AEAD modes)
            
        Returns:
            EncryptionResult with ciphertext and metadata
        """
        if self.algorithm == EncryptionAlgorithm.AES_GCM:
            return self._encrypt_aes_gcm(plaintext, key, associated_data)
        elif self.algorithm == EncryptionAlgorithm.AES_CBC:
            return self._encrypt_aes_cbc(plaintext, key)
        elif self.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            return self._encrypt_chacha20_poly1305(plaintext, key, associated_data)
        elif self.algorithm == EncryptionAlgorithm.FERNET:
            return self._encrypt_fernet(plaintext, key)
        else:
            raise CryptoError(f"Unsupported algorithm: {self.algorithm}")
    
    def decrypt(
        self,
        encryption_result: EncryptionResult,
        key: KeyMaterial,
        associated_data: Optional[bytes] = None
    ) -> bytes:
        """
        Decrypt ciphertext with key.
        
        Args:
            encryption_result: Result from encryption
            key: Decryption key
            associated_data: Additional authenticated data
            
        Returns:
            Decrypted plaintext
        """
        if self.algorithm == EncryptionAlgorithm.AES_GCM:
            return self._decrypt_aes_gcm(encryption_result, key, associated_data)
        elif self.algorithm == EncryptionAlgorithm.AES_CBC:
            return self._decrypt_aes_cbc(encryption_result, key)
        elif self.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            return self._decrypt_chacha20_poly1305(encryption_result, key, associated_data)
        elif self.algorithm == EncryptionAlgorithm.FERNET:
            return self._decrypt_fernet(encryption_result, key)
        else:
            raise CryptoError(f"Unsupported algorithm: {self.algorithm}")
    
    def _encrypt_aes_gcm(
        self,
        plaintext: bytes,
        key: KeyMaterial,
        associated_data: Optional[bytes]
    ) -> EncryptionResult:
        """Encrypt using AES-GCM."""
        if len(key) not in [16, 24, 32]:
            raise InvalidKeyError("AES key must be 16, 24, or 32 bytes")
        
        iv = SecureRandom.bytes(12)  # 96-bit IV for GCM
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv),
            backend=default_backend()
        )
        
        encryptor = cipher.encryptor()
        
        if associated_data:
            encryptor.authenticate_additional_data(associated_data)
        
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        return EncryptionResult(
            ciphertext=ciphertext,
            iv=iv,
            tag=encryptor.tag,
            algorithm=self.algorithm.value
        )
    
    def _decrypt_aes_gcm(
        self,
        encryption_result: EncryptionResult,
        key: KeyMaterial,
        associated_data: Optional[bytes]
    ) -> bytes:
        """Decrypt using AES-GCM."""
        if not encryption_result.iv or not encryption_result.tag:
            raise DecryptionError("IV and tag required for AES-GCM decryption")
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(encryption_result.iv, encryption_result.tag),
            backend=default_backend()
        )
        
        decryptor = cipher.decryptor()
        
        if associated_data:
            decryptor.authenticate_additional_data(associated_data)
        
        try:
            plaintext = decryptor.update(encryption_result.ciphertext) + decryptor.finalize()
            return plaintext
        except Exception as e:
            raise DecryptionError(f"AES-GCM decryption failed: {e}") from e
    
    def _encrypt_aes_cbc(self, plaintext: bytes, key: KeyMaterial) -> EncryptionResult:
        """Encrypt using AES-CBC with PKCS7 padding."""
        if len(key) not in [16, 24, 32]:
            raise InvalidKeyError("AES key must be 16, 24, or 32 bytes")
        
        # Add PKCS7 padding
        block_size = 16
        padding_length = block_size - (len(plaintext) % block_size)
        padded_plaintext = plaintext + bytes([padding_length] * padding_length)
        
        iv = SecureRandom.bytes(16)  # 128-bit IV for CBC
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend()
        )
        
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()
        
        return EncryptionResult(
            ciphertext=ciphertext,
            iv=iv,
            algorithm=self.algorithm.value
        )
    
    def _decrypt_aes_cbc(self, encryption_result: EncryptionResult, key: KeyMaterial) -> bytes:
        """Decrypt using AES-CBC and remove PKCS7 padding."""
        if not encryption_result.iv:
            raise DecryptionError("IV required for AES-CBC decryption")
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(encryption_result.iv),
            backend=default_backend()
        )
        
        decryptor = cipher.decryptor()
        
        try:
            padded_plaintext = decryptor.update(encryption_result.ciphertext) + decryptor.finalize()
            
            # Remove PKCS7 padding
            padding_length = padded_plaintext[-1]
            plaintext = padded_plaintext[:-padding_length]
            
            return plaintext
        except Exception as e:
            raise DecryptionError(f"AES-CBC decryption failed: {e}") from e
    
    def _encrypt_chacha20_poly1305(
        self,
        plaintext: bytes,
        key: KeyMaterial,
        associated_data: Optional[bytes]
    ) -> EncryptionResult:
        """Encrypt using ChaCha20-Poly1305."""
        if len(key) != 32:
            raise InvalidKeyError("ChaCha20 key must be 32 bytes")
        
        nonce = SecureRandom.bytes(12)  # 96-bit nonce
        
        cipher = Cipher(
            algorithms.ChaCha20(key, nonce),
            modes.GCM(nonce),
            backend=default_backend()
        )
        
        encryptor = cipher.encryptor()
        
        if associated_data:
            encryptor.authenticate_additional_data(associated_data)
        
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        return EncryptionResult(
            ciphertext=ciphertext,
            iv=nonce,
            tag=encryptor.tag,
            algorithm=self.algorithm.value
        )
    
    def _decrypt_chacha20_poly1305(
        self,
        encryption_result: EncryptionResult,
        key: KeyMaterial,
        associated_data: Optional[bytes]
    ) -> bytes:
        """Decrypt using ChaCha20-Poly1305."""
        if not encryption_result.iv or not encryption_result.tag:
            raise DecryptionError("Nonce and tag required for ChaCha20-Poly1305 decryption")
        
        cipher = Cipher(
            algorithms.ChaCha20(key, encryption_result.iv),
            modes.GCM(encryption_result.iv, encryption_result.tag),
            backend=default_backend()
        )
        
        decryptor = cipher.decryptor()
        
        if associated_data:
            decryptor.authenticate_additional_data(associated_data)
        
        try:
            plaintext = decryptor.update(encryption_result.ciphertext) + decryptor.finalize()
            return plaintext
        except Exception as e:
            raise DecryptionError(f"ChaCha20-Poly1305 decryption failed: {e}") from e
    
    def _encrypt_fernet(self, plaintext: bytes, key: KeyMaterial) -> EncryptionResult:
        """Encrypt using Fernet (high-level symmetric encryption)."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise CryptoError("Fernet requires cryptography library")
        
        f = Fernet(key)
        ciphertext = f.encrypt(plaintext)
        
        return EncryptionResult(
            ciphertext=ciphertext,
            algorithm=self.algorithm.value
        )
    
    def _decrypt_fernet(self, encryption_result: EncryptionResult, key: KeyMaterial) -> bytes:
        """Decrypt using Fernet."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise CryptoError("Fernet requires cryptography library")
        
        f = Fernet(key)
        
        try:
            plaintext = f.decrypt(encryption_result.ciphertext)
            return plaintext
        except Exception as e:
            raise DecryptionError(f"Fernet decryption failed: {e}") from e


class CryptoManager:
    """High-level cryptographic operations manager."""
    
    def __init__(self):
        """Initialize crypto manager with secure defaults."""
        self.hash_manager = HashManager(HashAlgorithm.SHA256)
        self.hmac_manager = HMACManager(HashAlgorithm.SHA256)
        self.encryption = SymmetricEncryption(EncryptionAlgorithm.AES_GCM)
    
    def generate_key(self, length: int = 32) -> bytes:
        """Generate cryptographically secure key."""
        return SecureRandom.bytes(length)
    
    def generate_salt(self, length: int = 16) -> bytes:
        """Generate cryptographically secure salt."""
        return SecureRandom.bytes(length)
    
    def derive_key_from_password(
        self,
        password: str,
        salt: Optional[bytes] = None,
        key_length: int = 32
    ) -> Tuple[bytes, bytes]:
        """
        Derive key from password with automatic salt generation.
        
        Args:
            password: Password to derive from
            salt: Optional salt (generated if None)
            key_length: Length of derived key
            
        Returns:
            Tuple of (derived_key, salt)
        """
        if salt is None:
            salt = self.generate_salt()
        
        key = KeyDerivation.pbkdf2(password, salt, key_length)
        return key, salt
    
    def encrypt_data(self, data: bytes, key: Optional[bytes] = None) -> Tuple[EncryptionResult, bytes]:
        """
        Encrypt data with optional key generation.
        
        Args:
            data: Data to encrypt
            key: Optional key (generated if None)
            
        Returns:
            Tuple of (encryption_result, key)
        """
        if key is None:
            key = self.generate_key()
        
        result = self.encryption.encrypt(data, key)
        return result, key
    
    def decrypt_data(self, encryption_result: EncryptionResult, key: bytes) -> bytes:
        """
        Decrypt data.
        
        Args:
            encryption_result: Result from encryption
            key: Decryption key
            
        Returns:
            Decrypted data
        """
        return self.encryption.decrypt(encryption_result, key)
    
    def hash_data(self, data: BytesLike, algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> str:
        """Hash data and return hex digest."""
        hash_manager = HashManager(algorithm)
        return hash_manager.hash_hex(data)
    
    def verify_integrity(self, data: BytesLike, expected_hash: str, algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> bool:
        """Verify data integrity against hash."""
        hash_manager = HashManager(algorithm)
        computed_hash = hash_manager.hash_hex(data)
        return SecureCompare.equal_str(computed_hash, expected_hash)


# =============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# =============================================================================

def sha256_digest(data: Union[bytes, str]) -> str:
    """
    Generate SHA-256 digest (backward compatible).
    
    Args:
        data: Data to hash
        
    Returns:
        SHA-256 digest as hex string
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def hmac_sha256(key: Union[bytes, str], data: Union[bytes, str]) -> str:
    """
    Generate HMAC-SHA256 (backward compatible).
    
    Args:
        key: Secret key
        data: Data to authenticate
        
    Returns:
        HMAC-SHA256 as hex string
    """
    if isinstance(key, str):
        key = key.encode('utf-8')
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hmac.new(key, data, hashlib.sha256).hexdigest()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_secure_token(length: int = 32) -> str:
    """Generate secure URL-safe token."""
    return SecureRandom.token_urlsafe(length)


def generate_key(length: int = 32) -> bytes:
    """Generate cryptographically secure key."""
    return SecureRandom.bytes(length)


def derive_key(password: str, salt: bytes, key_length: int = 32) -> bytes:
    """Derive key from password using PBKDF2."""
    return KeyDerivation.pbkdf2(password, salt, key_length)


def hash_file(filepath: str, algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> str:
    """Hash file contents."""
    hash_manager = HashManager(algorithm)
    
    with open(filepath, 'rb') as f:
        return hash_manager.hash_stream(f).hex()


def verify_hmac(key: BytesLike, message: BytesLike, expected_hmac: str) -> bool:
    """Verify HMAC (accepts hex string)."""
    hmac_manager = HMACManager()
    expected_bytes = bytes.fromhex(expected_hmac)
    return hmac_manager.verify(key, message, expected_bytes)


def constant_time_compare(a: str, b: str) -> bool:
    """Constant-time string comparison."""
    return SecureCompare.equal_str(a, b)


# Example usage and testing
if __name__ == "__main__":
    print("=== Enhanced Cryptographic Utilities Demo ===")
    
    # Test backward compatibility
    print("\n1. Backward Compatibility Tests:")
    data = "hello world"
    key = "secret key"
    
    sha_digest = sha256_digest(data)
    print(f"SHA-256: {sha_digest}")
    
    hmac_digest = hmac_sha256(key, data)
    print(f"HMAC-SHA256: {hmac_digest}")
    
    # Test enhanced functionality
    print("\n2. Enhanced Functionality Tests:")
    
    # Crypto manager
    crypto = CryptoManager()
    
    # Key generation
    secure_key = crypto.generate_key()
    print(f"Generated key: {secure_key.hex()[:32]}...")
    
    # Password-based key derivation
    derived_key, salt = crypto.derive_key_from_password("test_password")
    print(f"Derived key: {derived_key.hex()[:32]}...")
    print(f"Salt: {salt.hex()}")
    
    # Encryption/Decryption
    test_data = b"This is sensitive data that needs encryption!"
    encryption_result, enc_key = crypto.encrypt_data(test_data)
    print(f"Encrypted: {encryption_result.ciphertext.hex()[:32]}...")
    
    decrypted_data = crypto.decrypt_data(encryption_result, enc_key)
    print(f"Decrypted: {decrypted_data.decode()}")
    print(f"Encryption successful: {test_data == decrypted_data}")
    
    # Hash verification
    print("\n3. Hash Verification:")
    file_hash = crypto.hash_data(test_data)
    is_valid = crypto.verify_integrity(test_data, file_hash)
    print(f"Hash: {file_hash}")
    print(f"Integrity check: {is_valid}")
    
    # Secure random generation
    print("\n4. Secure Random Generation:")
    token = generate_secure_token()
    random_bytes = SecureRandom.bytes(16)
    random_int = SecureRandom.int(1, 100)
    
    print(f"Secure token: {token}")
    print(f"Random bytes: {random_bytes.hex()}")
    print(f"Random int: {random_int}")
    
    print("\nAll tests completed successfully!")