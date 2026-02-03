import pandas as pd
from cryptography.fernet import Fernet
import os

# --- Key Management ---
def generate_key():
    """Generates a new encryption key and saves it to a file."""
    key = Fernet.generate_key()
    with open("secret.key", "wb") as key_file:
        key_file.write(key)
    print("Generated a new encryption key and saved it as 'secret.key'.")

def load_key():
    """Loads the encryption key from the 'secret.key' file."""
    if not os.path.exists("secret.key"):
        raise FileNotFoundError("Encryption key ('secret.key') not found. Please generate one first.")
    return open("secret.key", "rb").read()

# --- Encryption ---
def encrypt_excel(input_file, output_file, key):
    """Encrypts an Excel file."""
    fernet = Fernet(key)
    try:
        df = pd.read_excel(input_file)
        # Convert the entire DataFrame to a string, then encode to bytes
        data_to_encrypt = df.to_csv(index=False).encode()
        encrypted_data = fernet.encrypt(data_to_encrypt)
        with open(output_file, "wb") as f:
            f.write(encrypted_data)
        print(f"Successfully encrypted '{input_file}' to '{output_file}'.")
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An error occurred during encryption: {e}")

# --- Decryption ---
def decrypt_excel(input_file, output_file, key):
    """Decrypts an Excel file and saves it with original data plus additions."""
    fernet = Fernet(key)
    try:
        with open(input_file, "rb") as f:
            encrypted_data = f.read()
        decrypted_data = fernet.decrypt(encrypted_data)
        # Convert the decrypted bytes back to a string and then to a DataFrame
        from io import StringIO
        df = pd.read_csv(StringIO(decrypted_data.decode()))
        # Here you would add your new data from the LLM
        # For demonstration, we'll just add a new column
        df['LLM_Addition'] = "New Data"
        df.to_excel(output_file, index=False)
        print(f"Successfully decrypted '{input_file}' and saved it as '{output_file}' with additions.")
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An error occurred during decryption: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # --- Step 1: Generate the key (only needs to be done once) ---
    if not os.path.exists("secret.key"):
        generate_key()

    # --- Step 2: Load the key ---
    try:
        encryption_key = load_key()

        # --- Step 3: Encrypt your Excel file ---
        original_excel = "original_data.xlsx"  # Your original Excel file
        encrypted_output = "encrypted_data.bin"
        # Create a dummy Excel file for demonstration
        if not os.path.exists(original_excel):
            pd.DataFrame({'A': [1, 2], 'B': [3, 4]}).to_excel(original_excel, index=False)
        encrypt_excel(original_excel, encrypted_output, encryption_key)

        # --- Step 4: Work with the LLM (using the encrypted file) ---
        print("\n--- Now you can use the encrypted file with your LLM ---\n")
        # In a real-world scenario, you would pass 'encrypted_output' to your LLM

        # --- Step 5: Decrypt the file to get the final version ---
        final_decrypted_excel = "final_decrypted_data.xlsx"
        decrypt_excel(encrypted_output, final_decrypted_excel, encryption_key)

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")