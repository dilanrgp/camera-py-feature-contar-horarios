import json

class TagModel:
    tags_json = """
        [{
            "idtag": 484,
            "idcategory": 133,
            "name": "Mujer de 0-9"
        },
        {
            "idtag": 485,
            "idcategory": 133,
            "name": "Mujer de 10-19"
        },
        {
            "idtag": 486,
            "idcategory": 133,
            "name": "Mujer de 20-29"
        },
        {
            "idtag": 487,
            "idcategory": 133,
            "name": "Mujer de 30-39"
        },
        {
            "idtag": 488,
            "idcategory": 133,
            "name": "Mujer de 40-49"
        },
        {
            "idtag": 489,
            "idcategory": 133,
            "name": "Mujer de 50-59"
        },
        {
            "idtag": 490,
            "idcategory": 133,
            "name": "Mujer de 60-69"
        },
        {
            "idtag": 491,
            "idcategory": 133,
            "name": "Mujer de 70-79"
        },
        {
            "idtag": 492,
            "idcategory": 133,
            "name": "Mujer mayor de 79"
        },
        {
            "idtag": 494,
            "idcategory": 133,
            "name": "Hombre de 0-9"
        },
        {
            "idtag": 495,
            "idcategory": 133,
            "name": "Hombre de 10-19"
        },
        {
            "idtag": 497,
            "idcategory": 133,
            "name": "Hombre de 20-29"
        },
        {
            "idtag": 498,
            "idcategory": 133,
            "name": "Hombre de 30-39"
        },
        {
            "idtag": 499,
            "idcategory": 133,
            "name": "Hombre de 40-49"
        },
        {
            "idtag": 500,
            "idcategory": 133,
            "name": "Hombre de 50-59"
        },
        {
            "idtag": 501,
            "idcategory": 133,
            "name": "Hombre de 60-69"
        },
        {
            "idtag": 502,
            "idcategory": 133,
            "name": "Hombre de 70-79"
        },
        {
            "idtag": 503,
            "idcategory": 133,
            "name": "Hombre mayor de 79"
        }]
    """
    def __init__(self):
        self.tags = json.loads(self.tags_json)
           
    def ageStr(self, face):
        age_type = ''
        if face.age < 10:
            age_type = 'de 0-9'
        elif face.age >= 10 and face.age < 20 :
            age_type = 'de 10-19'
        elif face.age >= 20 and face.age < 30 :
            age_type = 'de 20-29'
        elif face.age >= 30 and face.age < 40 :
            age_type = 'de 30-39'
        elif face.age >= 40 and face.age < 50 :
            age_type = 'de 40-49'
        elif face.age >= 50 and face.age < 60 :
            age_type = 'de 50-59'
        elif face.age >= 60 and face.age < 70 :
            age_type = 'de 60-69'
        elif face.age >= 70 and face.age < 79 :
                age_type = 'de 70-79'
        else:
            age_type = 'mayor de 79'

        return age_type
    
    def filter(self, name):
        for tag in self.tags:
            if tag['name'] == name:
                return tag
            
        


