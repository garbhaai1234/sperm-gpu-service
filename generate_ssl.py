"""
Generate a self-signed SSL certificate for the GPU VM.
Required because mobile browsers need HTTPS to access the camera (getUserMedia).

Usage:
    python3 generate_ssl.py

Creates:
    ssl/cert.pem
    ssl/key.pem

Then run:
    uvicorn main:app --host 0.0.0.0 --port 5002 --ssl-keyfile ssl/key.pem --ssl-certfile ssl/cert.pem
"""

import os
import subprocess
import sys

SSL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ssl")
CERT_PATH = os.path.join(SSL_DIR, "cert.pem")
KEY_PATH = os.path.join(SSL_DIR, "key.pem")

# Change this to your GPU VM's public IP
PUBLIC_IP = "34.19.196.129"


def main():
    os.makedirs(SSL_DIR, exist_ok=True)

    if os.path.exists(CERT_PATH) and os.path.exists(KEY_PATH):
        print(f"SSL certs already exist at {SSL_DIR}/")
        print("Delete them to regenerate.")
        return

    cmd = [
        "openssl",
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-keyout",
        KEY_PATH,
        "-out",
        CERT_PATH,
        "-days",
        "365",
        "-nodes",
        "-subj",
        f"/CN={PUBLIC_IP}/O=Garbha AI/C=IN",
        "-addext",
        f"subjectAltName=IP:{PUBLIC_IP},IP:127.0.0.1,DNS:localhost",
    ]

    print(f"Generating self-signed SSL cert for {PUBLIC_IP}...")
    try:
        subprocess.run(cmd, check=True)
        print(f"\nDone! Created:\n  {CERT_PATH}\n  {KEY_PATH}")
        print(f"\nStart the server with:")
        print(
            f"  uvicorn main:app --host 0.0.0.0 --port 5002 --ssl-keyfile {KEY_PATH} --ssl-certfile {CERT_PATH}"
        )
        print(f"\nOn phone: tap 'Advanced' > 'Proceed' to accept the self-signed cert.")
    except FileNotFoundError:
        print("openssl not found, trying Python fallback...")
        python_fallback()


def python_fallback():
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import datetime
        import ipaddress

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.COMMON_NAME, PUBLIC_IP),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Garbha AI"),
            ]
        )
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow())
            .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
            .add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.IPAddress(ipaddress.IPv4Address(PUBLIC_IP)),
                        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                    ]
                ),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )
        with open(KEY_PATH, "wb") as f:
            f.write(
                key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.TraditionalOpenSSL,
                    serialization.NoEncryption(),
                )
            )
        with open(CERT_PATH, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        print(f"Done! Created: {CERT_PATH}, {KEY_PATH}")
    except ImportError:
        print("Install: pip install cryptography")
        sys.exit(1)


if __name__ == "__main__":
    main()
