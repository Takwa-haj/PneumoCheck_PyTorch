"""
download_data.py — PneumoCheck (PyTorch)
Téléchargement automatique du dataset depuis Kaggle.
"""
import os, sys, zipfile

DATASET  = 'paultimothymooney/chest-xray-pneumonia'
DEST_DIR = 'data/chest_xray'
ZIP_FILE = 'chest-xray-pneumonia.zip'


def download():
    try:
        import kaggle
    except ImportError:
        print("Installation de kaggle CLI...")
        os.system(f'{sys.executable} -m pip install kaggle -q')
        import kaggle

    os.makedirs('data', exist_ok=True)
    print(f"Téléchargement de '{DATASET}'...")
    os.system(f'kaggle datasets download -d {DATASET} -p data/')

    zip_path = os.path.join('data', ZIP_FILE)
    if os.path.exists(zip_path):
        print(f"Extraction dans {DEST_DIR}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall('data/')
        os.remove(zip_path)
        print("✅ Dataset prêt dans data/chest_xray/")
    else:
        print("❌ ZIP non trouvé. Vérifiez ~/.kaggle/kaggle.json")
        sys.exit(1)

    # Vérification
    for split in ['train', 'val', 'test']:
        path = os.path.join(DEST_DIR, split)
        if os.path.exists(path):
            total = sum(len(os.listdir(os.path.join(path, cls)))
                        for cls in ['NORMAL', 'BACTERIA', 'VIRUS']
                        if os.path.exists(os.path.join(path, cls)))
            print(f"  {split:5s} : {total} images")

if __name__ == '__main__':
    download()
