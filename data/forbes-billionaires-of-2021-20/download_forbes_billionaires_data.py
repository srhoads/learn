# os.system('pip install kaggle')

import zipfile 
os.system('kaggle datasets download alexanderbader/forbes-billionaires-of-2021-20/forbes_billionaires.csv')
archive = zipfile.ZipFile('forbes-billionaires-of-2021-20.zip')
filename = archive.filelist[0].filename
xlfile = archive.open(filename)
df = pd.read_csv(xlfile)