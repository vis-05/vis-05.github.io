from flask import Flask, render_template, request, redirect, url_for
from Crypto.Cipher import AES
from cryptography.hazmat.primitives import hashes

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from PIL import Image
import stepic
import cv2
import numpy as np
from math import log10, sqrt

app = Flask(__name__)

# Your functions for key generation, AES encryption, PSNR calculation, etc. (mentioned in the previous code)
from hashlib import sha256
import base64
from Crypto import Random
from Crypto.Cipher import AES
import pandas as pd
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization, hashes
def gen_key():
    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend())

    public_key = private_key.public_key()
    return private_key, public_key


def save_pvkey(pk, filename):
    pem = pk.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    )
    with open(filename, 'wb') as pem_out:
        pem_out.write(pem)

def save_pukey(pk, filename):
    pem = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    with open(filename, 'wb') as pem_out:
        pem_out.write(pem)

private_key, public_key = gen_key()

save_pvkey(private_key, 'private_key')
save_pukey(public_key, 'public_key')
print("private key and public key generated.")
from hashlib import sha256
import base64
from Crypto import Random
from Crypto.Cipher import AES
import pandas as pd
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from PIL import Image
import stepic
DNA_data = { "words":["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W",
                   "X","Y","Z"," ",",",".",":","0","1","2","3","4","5","6","7","8","9"],
            "DNA_code": ["CGA","CCA","GTT","TTG","GGC","GGT","TTT","CGC","ATG","AGT","AAG","TGC","TCC","TCT","GGA","GTG",
                         "AAC","TCA","ACG","TTC","CTG","CCT","CCG","CTA","AAA","CTT","ACC","TCG","GAT","GCT","ACT","TAG",
                         "ATA","GCA","GAG","AGA","TTA","ACA","AGG","GCG"]
           }

DNA_df = pd.DataFrame.from_dict(DNA_data)
print(DNA_df)
#block size =16
#AES-128
BS = 16
pad = lambda s: bytes(s + (BS - len(s) % BS) * chr(BS - len(s) % BS), 'utf-8')
class AESCipher:
    def __init__(self, key):
        self.key = bytes(key, 'utf-8')
    def encrypt( self, raw ):
        raw = pad(raw)
        iv = Random.new().read( AES.block_size )
        cipher = AES.new(self.key, AES.MODE_CBC, iv )
        return base64.b64encode( iv + cipher.encrypt( raw ) )
cipher = AESCipher('LKHlhb899Y09olUi')

def load_pvkey(filename):
    with open(filename, 'rb') as pem_in:
        pemlines = pem_in.read()

    private_key = load_pem_private_key(pemlines, None, default_backend())
    return private_key
import cv2
import numpy as np
from math import log10,sqrt
def PSNR(original, steg):
    mse = np.mean((original - steg) ** 2)
    if(mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

@app.route('/')
def index():
    return render_template('Encrypt.html')

@app.route('/process_data', methods=['POST'])
def process_data():
    if request.method == 'POST':
        # Fetch form data
        name = request.form['Name']
        gender = request.form['Gender']
        birthdate = request.form['Birthdate']
        ssn = request.form['ssn']
        history = request.form['history']
        diagnosis = request.form['Diagnosis']

        # Message creation using form data
        
        
        # Generating the DNA encoded message as per your previous code
        message = f"Name:{name},Gender:{gender},Birthdate:{birthdate},SSN:{ssn},MedicalHistory:{history},Diagnosis: {diagnosis}"
        DNA_crypto_message = ""
        word = message.upper()
        print(word)

        for i in word:
            DNA_crypto_message += str(DNA_df.loc[ DNA_df['words'] == i.upper(), 'DNA_code' ].iloc[0])

        # AES encryption
        cipher = AESCipher('LKHlhb899Y09olUi')
        AES_encrypted_message = cipher.encrypt(DNA_crypto_message)

        # Loading private key for signature generation
        private_key = load_pvkey("private_key")
        
        # Signature generation
        signature = private_key.sign(
            AES_encrypted_message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        # Combining AES encrypted message and signature
        secret_msg = AES_encrypted_message + b"SIGNATURE" + signature
        
        # Embedding in an image
        im = Image.open('C:\\Users\\Dell\\OneDrive\\Documents\\InfoSec\\EHR.jpg')
        im1 = stepic.encode(im, secret_msg)
        im1.save('C:\\Users\\Dell\\OneDrive\\Documents\\InfoSec\\static\\encoded_image.png', 'PNG')
        
        # Calculating PSNR
        original = cv2.imread("C:\\Users\\Dell\\OneDrive\\Documents\\InfoSec\\EHR.jpg")
        steg = cv2.imread("C:\\Users\\Dell\\OneDrive\\Documents\\InfoSec\\static\\encoded_image.png")
        value = PSNR(original, steg)
        encoded_image_path='encoded_image.png'

        return render_template('Result.html', psnr_value=value, encoded_image=encoded_image_path)

    return "Error processing data"

if __name__ == '__main__':
    app.run(debug=True)