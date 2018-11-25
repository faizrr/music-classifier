import pandas as pd
import numpy as np
import librosa
import glob
import os
import audioread
import concurrent.futures
import tables
import os

tracks_df = pd.read_csv('/storage/music/fma_metadata/tracks.csv', index_col=0, header=[0, 1])
genres_df = pd.read_csv('/storage/music/fma_metadata/genres.csv')

def wav2mfcc(file_path):
    wave, sr = librosa.load(file_path, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave)
    
    return mfcc

def get_file_path(tracks_df, genres_df, file_name):
    index = int(file_name)
    title = tracks_df.loc[index]['track']['genre_top']
    genre_id = genres_df[genres_df['title'] == title]['genre_id']
    s = 'output/' + str(int(genre_id))
    return s

def save_to_csv(file_path, file_name, mfcc_df):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    mfcc_df.to_csv(file_path + '/' + file_name + '.csv')
            
def save_mfcc(f):
    base = os.path.basename(f)
    file_name = os.path.splitext(base)[0]
    try:
        mfcc = wav2mfcc(f)
        file_to_save_path = get_file_path(tracks_df, genres_df, file_name)

        mfcc_df = pd.DataFrame(mfcc)
        save_to_csv(file_to_save_path, file_name, mfcc_df)
        
        return file_to_save_path + '/' + file_name
    except audioread.NoBackendError:
        print(f, ' was skipped')

files = glob.glob("/storage/music/fma_small/**/*.mp3")

with concurrent.futures.ProcessPoolExecutor() as executor:
    for file, mfcc_file_key in zip(files, executor.map(save_mfcc, files)):
        print(f"MFCC for {file} was saved as {mfcc_file_key}")