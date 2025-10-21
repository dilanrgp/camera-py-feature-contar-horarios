def getQualityFilters(config, data_detector):
    """
    Función que devuelve los filtros de calidad que se aplicarán a la imagen
    Empieza con un try porque algunas veces se obtienen desde el webservice que
    y otras desde el config_detector.ini
    """
    try:
        if (config != None and config['face_threshold'] != None):
            face_threshold = config['face_threshold']
        else:
            face_threshold =  data_detector['face_threshold']
        if (config != None and config['embedding_threshold'] != None):
            embedding_threshold = config['embedding_threshold']
        else:
            embedding_threshold =  data_detector['embedding_threshold']
        if (config != None and config['face_size'] != None):
            face_size = config['face_size']
        else:
            face_size =  data_detector['face_size']
        if (config != None and config['cosine_threshold'] != None):
            cosine_threshold = config['cosine_threshold']
        else:
            cosine_threshold =  data_detector['cosine_threshold']
        if (config != None and config['tiempo_estancia_clientes'] != None):
            tiempo_estancia_clientes = config['tiempo_estancia_clientes']
        else:
            tiempo_estancia_clientes =  data_detector['tiempo_estancia_clientes']
    except :
        face_threshold =  data_detector['face_threshold']
        embedding_threshold =  data_detector['embedding_threshold']
        face_size =  data_detector['face_size']
        cosine_threshold =  data_detector['cosine_threshold']
        tiempo_estancia_clientes =  data_detector['tiempo_estancia_clientes']
    return {
        "face_threshold": face_threshold,
        "embedding_threshold": embedding_threshold,
        "face_size": face_size,
        "cosine_threshold": cosine_threshold,
        "tiempo_estancia_clientes": tiempo_estancia_clientes
    }


"""
funciones que no se usan pero pueden ser intesantes
calcularían la distancia entre los dos ojos y si están alineados.
los modelos devuelven unos landmarks que son los puntos de interés de la cara
no se usa porque los landmarks que devuelve el modelo no son muy precisos ya que 
los devuelve después de haber alineado la cara
"""

# Otros posibles params de calidad que se podrían aplicar este código iría
# en camera_control.py y las funciones quedan debajo
                            
# dist_betweenEyes = face.eyesDistance(f.kps)
# face_alignment_right,face_alignment_left = face.checkStraightLine(f.kps)
# if -20 < face_alignment_right < 20 or -20 < face_alignment_left < 20:
#     break
# if dist_betweenEyes < 10:
#     break

def eyesDistance(self, kps):
    # check face landmarks alignment
    """

    https://docs.aws.amazon.com/rekognition/latest/dg/recommendations-facial-input-images.html

    insightface kps: 0 --> right-eye
    1 --> left-eye 2--> nose
    3 --> mouth-right 4 --> mouth-left
    """
    kps = kps.astype(np.intc)
    # extract the left and right eye (x, y)-coordinates
    lStart, lEnd = kps[1]
    rStart, rEnd = kps[0]
    # https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
    # compute the center of mass for each eye
    leftEyeCenter = kps[1].mean(axis=0).astype("int")
    rightEyeCenter = kps[0].mean(axis=0).astype("int")
    # compute the angle between the eye centroids
    dY = kps[0][1] - kps[1][1]
    dX = kps[0][0] - kps[1][0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180
    # the ratio of the distance between eyes in the *current*
    dist = np.sqrt((dX**2) + (dY**2))
    # print(dist)
    return dist

def checkStraightLine(self, kps):
    # https://stackoverflow.com/questions/3813681/checking-to-see-if-3-points-are-on-the-same-line
    # right eye
    Ax, Ay = kps[0][0], kps[0][1]
    # nose
    Bx, By = kps[2][0], kps[2][1]
    # right jawl
    Cx, Cy = kps[3][0], kps[3][1]
    alignment = (Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By)) / 2

    # left eye
    Ax, Ay = kps[1][0], kps[1][1]
    # nose
    Bx, By = kps[2][0], kps[2][1]
    # right jawl
    Cx, Cy = kps[4][0], kps[4][1]
    alignment2 = (Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By)) / 2
    # the bigger the less aligned
    return alignment, alignment2