import librosa
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix


def extract_feature(file_name):
    X, sample_rate=librosa.load(file_name)
    stft=np.abs(librosa.stft(X))
    mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma=np.mean(librosa.feature.chroma_stft(S=stft,sr=sample_rate).T,axis=0)
    mel=np.mean(librosa.feature.melspectrogram(X,sr=sample_rate).T,axis=0)
    contrast=np.mean(librosa.feature.spectral_contrast(S=stft,sr=sample_rate).T,axis=0)
    tonnetz=np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                            sr=sample_rate).T,axis=0)
    return mfccs, chroma, mel, contrast, tonnetz

def parse_audio_files(parent_dir, sub_dirs, file_ext="*wav"):
    features, labels, =np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)): #create a file path
            try:
                mfccs, chroma, mel, contrast, tonnetz=extract_feature(fn)
            except Exception as e:
                print ("Error encountered while parsing file: "),fn
                continue
            ext_features=np.hstack([mfccs,chroma, mel,contrast,tonnetz])
            features=np.vstack([features,ext_features])
            labels=np.append(labels,fn.split('/')[2].split('-')[1])
    return np.array(features),np.array(labels,dtype=np.str)

def parse_one(myFile):
    features =np.empty((0,193))
    try:
        mfccs, chroma, mel, contrast, tonnetz=extract_feature(myFile)
    except Exception as e:
        print ("Error encountered while parsing file: "),myFile
        ext_features=np.hstack([mfccs,chroma, mel,contrast,tonnetz])
        features=np.vstack([features,ext_features])
    return np.array(features)

def do_stuff():
    parent_dir='allCropped'
    sub_dirs=['set1','set2']
    np.set_printoptions(threshold=np.inf)
    features,labels=parse_audio_files(parent_dir,sub_dirs)

    print("")

    x=features
    y=labels
    x_train, x_test, y_train, y_test=train_test_split(x,y)

    scaler=StandardScaler()
    scaler.fit(x_train)

    x_train=scaler.transform(x_train)
    x_test=scaler.transform(x_test)

    mlp=MLPClassifier(hidden_layer_sizes=(193,193,193),solver='lbfgs',momentum=0.90)
    mlp.fit(x_test,y_test)
    print("actual ripeness:")
    print(y_test)
    print("")

    predictions=mlp.predict(x_test)

    print("predictions")
    print(predictions)
    print(confusion_matrix(y_test,predictions))
