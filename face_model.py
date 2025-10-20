import time

class FaceModel:
    def __init__(self, fid, box, embedding_norm, gender, age):
        self.id = fid
        self.age = age
        self.gender = gender
        self.first_time = time.time()
        self.last_time = self.first_time
        self.best_embedding = embedding_norm
        self.man = 0
        self.woman = 0
        self.updateGender(gender)
        (startX, startY, endX, endY) = box
        self.pos = [{ "x": (startX + endX) / 2,"y": (startY + endY) / 2}]

    def count(self):
        return self.man + self.woman

    def updateGender(self, gender):
        if gender == 'M':
            self.man += 1
        else:
            self.woman += 1

        #self.gender = 'M' if self.man >= self.woman else 'F'

    def add(self, fid, box, embedding_norm, gender,age):
        if str(self.id) != fid:
            return False

        self.updateGender(gender)
        self.last_time = time.time()
        (startX, startY, endX, endY) = box
        self.pos.append({ "x": (startX + endX) / 2,"y": (startY + endY) / 2})

        if embedding_norm >= self.best_embedding:
            self.best_embedding = embedding_norm
            self.gender = gender
            self.age = age

        return True


    def genderStr(self):
        return 'Hombre' if self.gender == 'M' else 'Mujer'

    def ageStr(self):
        age_type = ''

        if self.age < 45:
            age_type = 'menor de 45'
        else:
            age_type = 'mayor de 45'
        #if self.age < 10:
        #    age_type = 'menos de 10'
        #elif self.age >= 10 and self.age < 20 :
        #    age_type = 'entre 10 y 19'
        #elif self.age >= 20 and self.age < 30 :
        #    age_type = 'entre 20 y 29'
        #elif self.age >= 30 and self.age < 40 :
        #    age_type = 'entre 30 y 39'
        #elif self.age >= 40 and self.age < 50 :
        #    age_type = 'entre 40 y 49'
        #elif self.age >= 50 and self.age < 60 :
        #    age_type = 'entre 50 y 59'
        #elif self.age >= 60 and self.age < 70 :
        #    age_type = 'entre 60 y 69'
        #elif self.age >= 70 and self.age < 80 :
        #    age_type = 'entre 70 y 79'
        #else:
        #    age_type = 'mayor de 79'

        return age_type

    def update_position(self, face_box):
        (startX, startY, endX, endY) = face_box
        self.face_pos.append({"x": (startX + endX) / 2, "y": (startY + endY) / 2})
        self.last_time = time.time()