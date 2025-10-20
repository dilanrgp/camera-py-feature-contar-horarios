import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial

# def 
# for i in range(len(all_embeddings)): 
#                     #evitar q se compare con el mismo (que sería el último)                       
#                     #get embeddings     
#                     if i+1 == len(all_embeddings): break                   
#                     embed = np.array(all_embeddings[i], dtype=np.float32)
                    
#                     lalala = np.linalg.norm(all_embeddings[i] - image_embeddings, axis=1)
#                     for i, face_distance in enumerate(lalala):
#                         #print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))
#                         if face_distance < 0.6:
#                             print("- With a normal cutoff of 0.6, would the test image match the known image? {}".format(face_distance < 0.6))
#                             total_people += 1

# 'ArcFace':  {'cosine': 0.68, 'euclidean': 4.15, 'euclidean_l2': 1.13},
# https://github.com/serengil/deepface/blob/master/deepface/commons/distance.py

def findCosineskLearn(source_representation, test_representation,threshold=0):
    distance, idx, ID = 0, 0, 0
    distances = []
    match = False
    #embeds =  [x[1] for x in source_representation]
    test_representation = np.array(test_representation)
    for i in range(len(source_representation)):
        distance = spatial.distance.cosine(source_representation[i][1], test_representation)
        distances.append(distance)
    idx = np.argmin(distances)
    ID = source_representation[idx][0]
    if (distances[idx] < threshold):
        match = True
    return match, ID

def findCosineDistance(source_representation, test_representation,threshold=0.68):
    i, distance = 0, 0
    match = False
    if type(source_representation) == list:
        source_representation = np.array(source_representation)
        #image_embeddings = np.array(image_embeddings, dtype=np.float32)
    if type(test_representation) == list:
        test_representation = np.array(test_representation)
    for i in range(len(source_representation)):
        a = np.matmul(np.transpose(source_representation[i]), test_representation)
        b = np.sum(np.multiply(source_representation[i], source_representation[i]))
        c = np.sum(np.multiply(test_representation, test_representation))
        distance = 1 - (a / (np.sqrt(b) * np.sqrt(c)))
        if distance < threshold:
            #print(distance)
            match = True
            break
    return match, i, distance


def findEuclideanDistance(source_representation, test_representation,threshold=1.13):
    i = 0
    #find same people in the current frame vs other frames
    match = False
    if type(source_representation) == list:
        source_representation = np.array(source_representation)
        #image_embeddings = np.array(image_embeddings, dtype=np.float32)
    if type(test_representation) == list:
        test_representation = np.array(test_representation)

    for i in range(len(source_representation)): 
        # compute one face vs all the other ones
        #if match is True: exit
        euclidean_distance = source_representation[i] - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        # print(euclidean_distance)
        if euclidean_distance < threshold:
            #print(euclidean_distance)
            match = True
            break
            # not found this face so, new person
        #if match is True: exit
    return match, i
        #dist = np.sqrt(np.sum(np.square(np.subtract(feats[i,:], feats2[j,:]))))
# dist = np.sqrt(np.sum(np.square(emb_array1[0]-emb_array2[0])))

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))

def compare(source_representation, test_representation,threshold=0.3):
    i = 0
    from numpy.linalg import norm
    #find same people in the current frame vs other frames
    match = False
    if type(source_representation) == list:
        source_representation = np.array(source_representation)
        #image_embeddings = np.array(image_embeddings, dtype=np.float32)
    if type(test_representation) == list:
        test_representation = np.array(test_representation)
    for i in range(len(source_representation)): 
        # compute one face vs all the other ones
        #if match is True: exit
        distance = np.dot(source_representation[i], test_representation) / norm(source_representation[i]) * norm(test_representation)
        # print(euclidean_distance)
        print(distance)
        if distance < -0.05:
                match = True
                break
    return match, i

