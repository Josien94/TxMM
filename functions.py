
import csv, itertools
import cv2
import os
from os import listdir
from os.path import isfile, join
import re
import pickle
import json
from sklearn.model_selection import train_test_split
import imblearn
from imblearn.over_sampling import SMOTE
from collections import Counter
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from PIL import Image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import pickle
from sklearn import tree
import graphviz 
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import plot_confusion_matrix


'''
Arugments: video (CV2 video), frame_number (int)

Resize frame [frame_number] of video [video]

Returns: frame_rgb (CV2 image)
'''
def get_frame_resized(video, frame_number):
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video.read()
    frame = cv2.resize(frame, (224, 224))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb


'''
Arugments: image (CV2 image)

Creates neural representation unsing the vgg19_without_last_model

Returns: numpy array containing the neural representation
'''
def neural_representations(image):
    return np.array(vgg19_without_last.predict(image)).flatten()

'''
Arugments: image_array (numpy array), format (tuple)

Crops image for purposes of the VGG19 model

Returns: numpy array containing the cropped image
'''
def convert_image_to_format(image_array, format=(224,224)):
    img = Image.fromarray(image_array)
    height,width = img.size
    #   # fit in the biggest possible square to crop this
    img = img.crop((400, 10, height , width))
    img = img.resize(format)
    return np.array(img)

'''
Arugments: title (string), fns (int) fns_op (list), y (list)

Plots distribution of the Palms Open sign over an x amount of frames.

Returns: Nothing
'''

def plot_distribution(title,fns,fns_OP, y = []):
    ys = []
    xs = []
    if(y == []):
        for fn in range(0,len(fns)):
            xs= xs +[fn]

            if(fn in fns_OP ):
                ys = ys+[1]
            else:
                ys = ys+[0]
    
    else:
        for fn in range(0,len(y)):
            xs = xs+[fn]
        ys = y
        print(len(ys))
        
    plt.bar(xs, ys)
    plt.xlabel('Frame number')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()

'''
Arugments: classifier(string), x_train (numpy array), y_train (numpty array), hyperparamsDictionary (dictionary), category (string)

Performs binary classification with classifier [classifier] to tune the hypermarapeters defined in [hyperparamsDictionary].
Tuning hyperparamters is performed with stratified 10-fold cross validation.
Computes the (average) Root Mean Square Error (RMSE) for each parameter value.

Returns: rmseAll (list)
'''
def tune_hyperparameters(classifier, X_train,y_train, hyperparamsDictionary,category):
    if(classifier == 'dt'):
        rmseAll = []
        
        #Cross validation to find optimal tree depth
        skf = StratifiedKFold(n_splits=10,shuffle=True)
        
        depths = hyperparamsDictionary['depths']
        for hp in depths:
            rmseTemp = []
            for train_index, test_index in skf.split(X_train, y_train):

                X_train_CV, X_test_CV = X_train[train_index], X_train[test_index]
                y_train_CV, y_test_CV = y_train[train_index], y_train[test_index]
               
                clf = tree.DecisionTreeClassifier(random_state =0,max_depth=hp)
                clf = clf.fit(X_train_CV, y_train_CV)
                ypred = clf.predict(X_test_CV)
                
                rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_test_CV, ypred))
                rmseTemp.append(rmse)
        
            rmseAll.append(np.mean(rmseTemp))
          
        return rmseAll

    
def extract_VGG_features(rel_path_videos, video_names, n):
    VGG_features_dictionary = dict()
    model = VGG19(weights='imagenet')
    #model.summary()
    vgg19_without_last = Model(inputs=model.inputs, outputs=model.get_layer('fc2').output)
    
    for i in range(0,n):
        path_video = get_path_video(rel_path_videos, video_names[i])
       
        video = cv2.VideoCapture(path_video)
        video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        VGG19_features = []
        for j in range(0,video_length):
            frame_rgb = get_frame_resized(video,j)
            #formatted_img = convert_image_to_format(frame_rgb)
            preprocessed_img = image.img_to_array(frame_rgb)
            preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
            preprocessed_img = preprocess_input(preprocessed_img)
           
            img_VGG19 = np.array(vgg19_without_last.predict(preprocessed_img)).flatten()#neural_representations(preprocessed_img)
            VGG19_features.append(img_VGG19)
        VGG_features_dictionary[video_names[i]] = np.array(VGG19_features)
    return VGG_features_dictionary

'''
Arugments: data (dictionary)

Parses the Open Pose data to the right coordinates format

Returns: parsed_frame(dictioanry)
'''
def parse_json_frame(data):
    try:
        data_person = data['people'][0]
        body_x = data_person['pose_keypoints_2d'][::3]
        body_y = data_person['pose_keypoints_2d'][1::3]
        body_c = data_person['pose_keypoints_2d'][2::3]


        face_x = data_person['face_keypoints_2d'][::3]
        face_y = data_person['face_keypoints_2d'][1::3]
        face_c = data_person['face_keypoints_2d'][2::3]

        handL_x = data_person['hand_left_keypoints_2d'][::3]
        handL_y = data_person['hand_left_keypoints_2d'][1::3]
        handL_c = data_person['hand_left_keypoints_2d'][2::3]

        handR_x = data_person['hand_right_keypoints_2d'][::3]
        handR_y = data_person['hand_right_keypoints_2d'][1::3]
        handR_c = data_person['hand_right_keypoints_2d'][2::3]

        parsed_frame = {'body_xs': body_x,
                        'body_ys' : body_y,
                        'face_xs': face_x,
                        'face_ys': face_y,
                        'handL_xs': handL_x,
                        'handL_ys': handL_y,
                        'handR_xs': handR_x,
                        'handR_ys': handR_y}
    except:

        parsed_frame = {'body_xs': [],
                        'body_ys' : [],
                        'face_xs': [],
                        'face_ys': [],
                        'handL_xs': [],
                        'handL_ys': [],
                        'handR_xs': [],
                        'handR_ys': []}
    return parsed_frame

'''
Arugments: rel_path_videos (string)

Gets all video names of the files at the location [rel_path_videos]

Returns: video_names (list)
'''

def get_all_video_names(rel_path_videos):
    video_names = []
    filenames_videos = get_all_filepaths(rel_path_videos)
    for name in filenames_videos:
        video_name = re.split('_b',name)[0]
        video_names.append(video_name)
    return video_names


'''
Arugments: start (int), stop(int), step(int)

Creates a list of ranges, based on start position, end position and intermediate steps.

Returns: l (list)
'''

def frange(start, stop, step):
    l = []
    i = start
    while i < stop:
        l.append(i)
        i += step
    return l

'''
Arugments: fn_video(int), key (string), conversion_dictionary(dictionary)

Retrieves the label of a frame, based on annotated data in [conversion_dictionary].

Returns: label (int)
'''

def get_label(fn_video,key, conversion_dictionary):
    #Label is either 0 (no PO) or 1 (PO)
    for tup in conversion_dictionary[key]:
        start_frame = tup[0]
        end_frame = tup[1]
        range_list = frange(start_frame,end_frame+1,1)
        if(fn_video in range_list):
            return 1
    return 0

'''
Arugments: OP_features (dictionary), VGG19_features (dictionary), OP_features(dictionary), offset_dictionary_keypoints(dictionary), offset_dictionary_video(dictionary), conversion_dictionary(dictionary),n (int),oversample (string)

Creates train- and test data of the VGG19 model and Open Pose features.


Returns: X (dictionary),y(dictionary),X_train(dictionary), X_test(dictionary), y_train(dictionary), y_test(dictionary), fns(dictionary), fns_OP(dictionary)
'''
def split_data(annotated_dictionary, VGG19_features,OP_features, offset_dictionary_keypoints, offset_dictionary_video, conversion_dictionary,n,oversample):
    X = dict()
    y = dict()
    fns = dict()
    fns_OP = dict()
    X_train = dict()
    y_train = dict()
    X_test = dict()
    y_test = dict()

    video_names = list(annotated_dictionary.keys())
   
    for i in range(0,n):
        X[video_names[i]] = []
        y[video_names[i]] = []
        fns[video_names[i]] = []
        fns_OP[video_names[i]] = []
        X_train[video_names[i]] = []
        y_train[video_names[i]] = []
        X_test[video_names[i]] = []
        y_test[video_names[i]] = []
        
        start_frame_video =  offset_dictionary_video[video_names[i]][0]
        stop_frame_video = offset_dictionary_video[video_names[i]][1]
        start_frame_keypoints = offset_dictionary_keypoints[video_names[i]][0]
        stop_frame_keyponts = offset_dictionary_keypoints[video_names[i]][1]
        
        feature_names, lengths_features =  get_feature_names(OP_features,VGG19_features)
        for fn_video in range(start_frame_video, stop_frame_video):
            Xs = []
            
            fn_keypoints = fn_video- start_frame_video +start_frame_keypoints
            
          
            #Add body coordinates OP
            body_xs = OP_features[video_names[i]][fn_keypoints]['body_xs']
            if(body_xs != []):
                Xs = Xs+body_xs
            else:
                Xs = Xs+list(np.zeros(lengths_features[0]))
            
            body_ys = OP_features[video_names[i]][fn_keypoints]['body_ys']
            if(body_ys != []):
                Xs = Xs+body_ys
            else:
                Xs = Xs+list(np.zeros(lengths_features[1]))
                
            #Add face coordinates OP
            face_xs = OP_features[video_names[i]][fn_keypoints]['face_xs']
      
            if(face_xs != []):
                Xs = Xs+face_xs
            else:
                Xs = Xs+list(np.zeros(lengths_features[2]))
                
            face_ys = OP_features[video_names[i]][fn_keypoints]['face_ys']
         
            if(face_ys != []):
                Xs = Xs+face_ys
            else:
                Xs = Xs+list(np.zeros(lengths_features[3]))
            
            #Add left hand coordinates OP
            handL_xs = OP_features[video_names[i]][fn_keypoints]['handL_xs']
            if(handL_xs != []):
                Xs = Xs+handL_xs
            else:
                Xs = Xs+list(np.zeros(lengths_features[4]))
                
                
            handL_ys = OP_features[video_names[i]][fn_keypoints]['handL_ys']
            if(handL_ys != []):
                Xs = Xs+handL_ys
            else:
                Xs = Xs+list(np.zeros(lengths_features[5]))
                
            #Add right hand coordinates OP
            handR_xs = OP_features[video_names[i]][fn_keypoints]['handR_xs']
            if(handR_xs != []):
                Xs = Xs+handR_xs
            else:
                Xs = Xs+list(np.zeros(lengths_features[6]))
            
            handR_ys = OP_features[video_names[i]][fn_keypoints]['handR_ys']
            if(handR_ys != []):
                Xs = Xs+handR_ys
            else:
                Xs = Xs+list(np.zeros(lengths_features[7]))
            
            
            #Add VGG19 features
            Xs = Xs+ list(VGG19_features[video_names[i]][fn_video])
            
          
            X[video_names[i]].append(Xs)
            y_label = get_label(fn_video, video_names[i],conversion_dictionary)
            y[video_names[i]].append(y_label)
            
            fns[video_names[i]] = fns[video_names[i]]+[fn_video]
            if(y_label == 1):
                fns_OP[video_names[i]] =  fns_OP[video_names[i]] + [fn_video]
            
               
          
        X[video_names[i]] = np.array(X[video_names[i]])
        y[video_names[i]] = np.array(y[video_names[i]])
    
        
        if(oversample == 'true'):
            print("Shape X for key {}: {}".format(video_names[i],X[video_names[i]].shape))
            print("Shape y for key {}: {}".format(video_names[i],y[video_names[i]].shape))
            print("Distribution labels before oversampling for key {}: {}".format(video_names[i],Counter(y[video_names[i]])))
            oversample_smote = SMOTE()
            X_new, y_new = oversample_smote.fit_resample(X[video_names[i]], y[video_names[i]])
            print("Distribution labels after oversampling for key {}: {}".format(video_names[i],Counter(y_new)))
        
            X_train[video_names[i]], X_test[video_names[i]], y_train[video_names[i]], y_test[video_names[i]] = train_test_split(X_new, y_new, test_size=0.20, random_state=10)
        else:
            X_train[video_names[i]], X_test[video_names[i]], y_train[video_names[i]], y_test[video_names[i]] = train_test_split(X[video_names[i]] ,y[video_names[i]], test_size=0.20, random_state=10)

    return X,y,X_train, X_test, y_train, y_test, fns, fns_OP


'''
Arugments: x(Numpy array),y (Numpy array),titlePlot (string),xlabelPlot(string), ylabelPlot(string), xticksPlot(list)

Creates barplot.

Returns: Nothing
'''

def create_barplot(x,y,titlePlot,xlabelPlot, ylabelPlot, xticksPlot):
    plt.figure()
    plt.bar(x,y,width=7.0)
    plt.xticks(x,xticksPlot)
    plt.xlabel(xlabelPlot)
  
    plt.ylabel(ylabelPlot)
    plt.title(titlePlot)
    
    plt.show()


'''
Arugments: 

Returns:
'''
    
def perform_classification(classifier,X_train,y_train,X_test,y_test, hyperparams, title):
    if(classifier == 'dt'):
        clf = tree.DecisionTreeClassifier(random_state=0, max_depth=hyperparams['depth'])
        clf = clf.fit(X_train, y_train)
        ypred = clf.predict(X_test)
        disp = plot_confusion_matrix(clf,X_test,y_test)
        disp.ax_.set_title(title)

        
        
    return ypred

'''
Arugments: 

Returns:
'''

    
def combine_data(annotated_dictionary, X_train_smote, y_train_smote, X_test_smote, y_test_smote):
    video_name_first =list(annotated_dictionary.keys())[0]
    video_name_second = list(annotated_dictionary.keys())[1]
    video_name_third = list(annotated_dictionary.keys())[2]
    
    X_train_smote_concat = np.vstack([X_train_smote[video_name_first], X_train_smote[video_name_second],X_train_smote[video_name_third]])
    y_train_smote_concat = np.append(y_train_smote[video_name_first], y_train_smote[video_name_second])
    y_train_smote_concat = np.append(y_train_smote_concat, y_train_smote[video_name_third])

    X_test_smote_concat = np.vstack([X_test_smote[video_name_first], X_test_smote[video_name_second],X_test_smote[video_name_third]])
    y_test_smote_concat = np.append(y_test_smote[video_name_first], y_test_smote[video_name_second])
    y_test_smote_concat = np.append(y_test_smote_concat, y_test_smote[video_name_third])
    
    return X_train_smote_concat, X_test_smote_concat, y_train_smote_concat,y_test_smote_concat

'''
Arugments: 

Returns:
'''


def make_conversion_duration_frame(annotated_dictionary, rel_path_videos, filenames_videos):
    conversion_dur_frame_dictionary = dict()
    for video_name in annotated_dictionary:
        conversion_dur_frame_dictionary[video_name] = []
        path_video = get_path_video(rel_path_videos, video_name)
            
        amount_frames, fps = get_frames_stats(path_video)
        for frame in annotated_dictionary[video_name]:        

            if(fps != 0.0 and video_name in filenames_videos):
                annotation_begin= frame["beginannotation"]
                duration_PO = round(get_frame_number(frame["durationannotation"], fps))
                start_frame = round(get_frame_number(annotation_begin, fps))
                end_frame = int(start_frame+duration_PO)
                conversion_dur_frame_dictionary[video_name].append((start_frame,end_frame))
            elif(fps == 0.0 and video_name in filenames_videos):
                print("File not found: {}!".format(video_name))
    return conversion_dur_frame_dictionary

'''
Arugments: 

Returns:
'''

def extract_OP_features(rel_path_keypoints,video_names,n):
    OP_features_dictionary = dict()
 
        
    for i in range(0,n):
        OP_features = []
        files_keypoints = get_all_filepaths(rel_path_keypoints+video_names[i])
        for j in range(0,len(files_keypoints)):
            path_keypoints = get_path_keypoints(rel_path_keypoints,j,video_names[i])
            data = load_keypoints(path_keypoints)
            OP_features.append(parse_json_frame(data))
             
        OP_features_dictionary[video_names[i]] = np.array(OP_features)
    return OP_features_dictionary

'''
Arugments: 

Returns:
'''

def get_path_video(rel_path_videos, key):
    path_video = "{}{}_b.mpg".format(rel_path_videos,key)
    return path_video
 
'''
Arugments: 

Returns:
'''
def get_path_keypoints(rel_path_keypoints, frame_n, key):
    frame_number = pad_video_name(str(frame_n), 12)
    path_frame = "{}{}/{}_b_{}_keypoints.json".format(rel_path_keypoints,key,key,  frame_number)
    return path_frame

'''
Arugments: 

Returns:
'''
def save_dictionary(filename, data ):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        '''
Arugments: 

Returns:
'''
def load_dictionary(filename):
    with open(filename, 'rb') as f:
        content = pickle.load(f)
        
    return content


'''
Arguments: rel_path_videos (string), video_names (list), frames_videos (list), n (integer)

Creates dictionary with offsets, containing starting and end frame of a video with a person in the frame

dictionary_offsets = {
  <videoname1> : (<start_offset>,<end_offset>),
  <videoname2> : (<start_offset>,<end_offset>),
}
Returns
'''
def get_dictionary_offsets_videos(rel_path_videos,video_names, n):
    dictionary_offsets = dict()
    if(n>len(video_names)):
        print("Out of bounds : n({}) > amount videos({})".format(n,len(video_names)))
    else:
        for i in range(0,n):
            start_offset,end_offset = get_offsets_video(rel_path_videos,video_names[i])
            dictionary_offsets[video_names[i]] = (start_offset, end_offset)
    return dictionary_offsets

'''
Arugments: 

Returns:
'''

def load_keypoints(filename):
    with open(filename, 'r') as f:
        content = json.load(f)
    return content

'''
Arugments: 

Returns:
'''
             
def get_dictionary_offsets_keypoints(rel_path_keypoints, keys,n, offset_dictionary_video):
    offset_dictionary = dict()

   
    for i in range(0,n):
        first_frame_person_b = False
        last_frame_person_b = False
        first_frame_person = 0
        last_frame_person = 0
    
        dir_keypoints = rel_path_keypoints+ keys[i]
        filenames = sorted(get_all_filepaths(dir_keypoints))
        diff_keys_video = offset_dictionary_video[keys[i]][1]- offset_dictionary_video[keys[i]][0]
        for j in range(0,len(filenames)):
            keypoints_dictionary = load_keypoints(dir_keypoints + "/" + filenames[j])
            keys_frame_people  =keypoints_dictionary["people"]
            if(keys_frame_people != [] and first_frame_person_b == False):
                first_frame_person_b = True
                first_frame_person = j
            elif(keys_frame_people == [] and first_frame_person_b == True and last_frame_person_b == False and j > 1000 and (diff_keys_video == j-first_frame_person-1)):
                last_frame_person_b = True
                last_frame_person = j-1
            
                    
        offset_dictionary[keys[i]] = (first_frame_person, last_frame_person)
    return offset_dictionary    

'''
Arugments: 

Returns:
'''

def get_all_filepaths(directory):
    #   '''
    #   A helper function to get all absolute file paths in a directory (recursively)
    #   :param directory:  The directory for which we want to get all file paths
    #   :return         :  A list of all absolute file paths as strings
    #   '''
    #   for dirpath,_,filenames in os.walk(directory):
    #     for f in sorted(filenames):
    #         yield os.path.abspath(os.path.join(dirpath, f))
    return  [f for f in listdir(directory) if isfile(join(directory, f))]
'''
Arugments: 

Returns:
'''

def get_frame(video, frame_number):
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video.read()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb
'''
Arugments: 

Returns:
'''

def pad_video_name(number, n):
    length_number = len(number)
    padded_number = ""
    for i in range(0,n-length_number):
        padded_number += "0"
    padded_number += number
    
    return padded_number
'''
Arugments: 

Returns:
'''

def get_offsets_video(rel_path_videos, key):
    path_video = get_path_video(rel_path_videos, key)
    video = cv2.VideoCapture(path_video)
    amount_frames= video.get(cv2.CAP_PROP_FRAME_COUNT)
    first_frame_person_b = False
    last_frame_person_b = False
    first_frame_person = 0
    last_frame_person = 0
    for i in range(0,round(amount_frames)):      
        rgb = get_frame(video,i)
        frame_category = get_frame_category(rgb)

        if(frame_category == 'person' and first_frame_person_b == False):
            first_frame_person_b = True
            first_frame_person = i
        elif(frame_category == 'noperson' and first_frame_person_b == True and  last_frame_person_b == False and i > 1000):
            last_frame_person_b = True
            last_frame_person = i-1
            
    return first_frame_person,last_frame_person
    
'''
Arugments: 

Returns:
'''

def get_frame_category(frame):
    acc_b_g_r = [0,0,0]
    acc_other = 0
    for i,col in enumerate(('b','g','r')):
        hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        for i_col in range(0,len(hist)):
            if( (round(hist[i_col]) ==1) and i_col>200):
                acc_b_g_r[i]+=1
            elif (hist[i_col] > 0.2 and i_col < 200): 
                acc_other +=1
                
    if(acc_b_g_r[0] <= 2 and acc_b_g_r[0] > 0 \
        and acc_b_g_r[1] <=2 and acc_b_g_r[1] > 0 \
        and acc_b_g_r[2] <=2 and acc_b_g_r[2] >0 \
        and acc_other == 0):
        return 'noperson'
    else:
        return 'person'

    
'''
Arugments: 

Returns:
'''    
def get_frames_stats(filename):
    
    #video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    #ret, frame = video.read()
    stream = cv2.VideoCapture(filename)
    #cv_capture = cv2.CaptureFromFile(filename)
    #amount_frames = cv2.GetCaptureProperty(cv2_capture,cv2.CV_CAP_PROP_FRAME_COUNT)
    amount_frames= stream.get(cv2.CAP_PROP_FRAME_COUNT)

    #fps = cv2.GetCaptureProperty(cvcapture,cv2.CV_CAP_PROP_FPS)
    fps = stream.get(cv2.CAP_PROP_FPS)
    return amount_frames, fps


'''
Arugments: 

Returns:
'''
def get_frame_number(timestamp, fps):
    duration_frame_sec = 1/fps
    duartion_frame_ms = duration_frame_sec * 1000
    return float(timestamp)/duartion_frame_ms

'''
Arguments: filename (string), ext (string)

Reads and saves content of the file with <filename> 

Returns: content (string)
'''
def read_file(filename):
    with open(filename, 'r') as f:    
        content = f.read()    
    return content



'''
Arguments: csv_reader (csvReader)

Makes a dictionary of the annotated videos, only containing information about the PO sign

annotated_dictionary = {
   <TranscriptionName_1> : [{"tiername": <TierName>,
                              "tiertype": <TierType>,
                              "tierpart" : <TierParticipant>,
                              "annotation": <annotation>,
                              "tierannotator" : <TierAnnotator>,
                              "hitlength" : <hitLength>,
                              "beginannotation" : <AnnotationBeginTime>,
                              "durationannotation" : <AnnotationDuration>,
                              "hitposintier" : <HitPostitionInTier>,
                              "leftcontext" : <LeftContext>,
                              "rightocntext" : <RightContext>
                           }],
   <TranscriptionName_2> : [{"tiername": <TierName>,
                              "tiertype": <TierType>,
                              "tierpart" : <TierParticipant>,
                              "annotation": <annotation>,
                              "tierannotator" : <TierAnnotator>,
                              "hitlength" : <hitLength>,
                              "beginannotation" : <AnnotationBeginTime>,
                              "durationannotation" : <AnnotationDuration>,
                              "hitposintier" : <HitPostitionInTier>,
                              "leftcontext" : <LeftContext>,
                              "rightocntext" : <RightContext>
                           }],
    ...

}

Returns: annotated_dictionary(dictionary)
'''


def make_annotated_dictionary(annotated_filename):
    with open(annotated_filename, 'r') as f:
        csv_reader = csv.reader(f, delimiter='\t')
        annotated_dictionary = dict()
        for num,line in enumerate(csv_reader):
            if(line != []):

                if(num == 0):
                    #Store (stripped) header as list
                    columnNames = line
                else:
                    video_dict = dict()
                    videoname = line[0]

                    #Case specific: Remove _NP from title 
                    videoname = re.sub(r'_NP','',videoname)
                    #Remove .eaf from videoname
                    videoname = videoname[:-4]
                    videoname = videoname + "_"+line[3] #Videoname 
                    
                    video_dict["tiername"] = line[1]
                    video_dict["tiertype"] = line[2]
                    video_dict["tierpart"] = line[3]
                    video_dict["annotation"] = line[4]
                    video_dict["tierannotation"] = line[5]
                    video_dict["hitlength"] = line[6]
                    video_dict["beginannotation"] = line[7]
                    video_dict["durationannotation"] = line[8]
                    video_dict["hitposintier"] = line[9]
                    video_dict["leftcontext"] = line[10]
                    video_dict["rightcontext"] = line[11]

                    if(videoname in annotated_dictionary):
                        annotated_dictionary[videoname].append(video_dict)

                    else:
                        annotated_dictionary[videoname] = [video_dict]

        return annotated_dictionary


         
    
def main():
    print('')

if __name__ == '__main__':
    main()
