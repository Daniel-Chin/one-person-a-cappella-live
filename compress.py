import os

files = os.listdir()
files = [x for x in files if x.endswith('.wav')]
for fn in files:
    outfn = os.path.splitext(fn)[0] + '.mp3'
    os.system(f'ffmpeg -i {fn} {outfn}')
print('ok')
input('Enter...')
