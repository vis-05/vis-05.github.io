from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import stepic
from io import BytesIO
from cryptography.hazmat.primitives import serialization, hashes
from hashlib import sha256
import base64
from Crypto.Cipher import AES
import pandas as pd
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.exceptions import InvalidSignature

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('Decrypt.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    if request.method == 'POST':
        # Get uploaded image
        uploaded_image = request.files['image']
        
        if uploaded_image:
            image = Image.open(uploaded_image)
            stego_image = stepic.decode(image)

            ind_sep = stego_image.find('SIGNATURE')
            message = bytes(stego_image[:ind_sep], 'utf-8')
            signature = bytes(stego_image[ind_sep+9:], 'latin1')

            # Load public key
            def load_pukey(filename):
                with open(filename, 'rb') as pem_in:
                    pemlines = pem_in.read()
                public_key = load_pem_public_key(pemlines, default_backend())
                return public_key

            public_key = load_pukey("public_key")

            try:
                public_key.verify(
                    signature,
                    message,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                print(message)
            except InvalidSignature:
                print('Invalid!')

            # Decrypting the message
            BS = 16
            unpad = lambda s: s[0:-s[-1]]

            class AESCipher:
                def __init__(self, key):
                    self.key = bytes(key, 'utf-8')

                def decrypt(self, enc):
                    enc = base64.b64decode(enc)
                    iv = enc[:16]
                    cipher = AES.new(self.key, AES.MODE_CBC, iv)
                    return unpad(cipher.decrypt(enc[16:])).decode('utf8')

            cipher = AESCipher('LKHlhb899Y09olUi')
            AES_decrypted = cipher.decrypt(message)

            # Conversion to original message
            DNA_data = {
                "words": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
                           "X", "Y", "Z", " ", ",", ".", ":", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                "DNA_code": ["CGA", "CCA", "GTT", "TTG", "GGC", "GGT", "TTT", "CGC", "ATG", "AGT", "AAG", "TGC", "TCC", "TCT", "GGA", "GTG",
                             "AAC", "TCA", "ACG", "TTC", "CTG", "CCT", "CCG", "CTA", "AAA", "CTT", "ACC", "TCG", "GAT", "GCT", "ACT", "TAG",
                             "ATA", "GCA", "GAG", "AGA", "TTA", "ACA", "AGG", "GCG"]
            }

            DNA_df = pd.DataFrame.from_dict(DNA_data)

            l = [AES_decrypted[i:i+3] for i in range(0, len(AES_decrypted), 3)]

            original_message = ""
            for i in l:
                original_message += str(DNA_df.loc[DNA_df['DNA_code'] == i, 'words'].iloc[0])

            # Prepare the message to be displayed
            original_message = original_message.lower()
            return render_template('ResultD.html', original_message=original_message)

        return "Error processing image"

if __name__ == '__main__':
    app.run(debug=True)