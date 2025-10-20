from face_model import FaceModel
from tag_model import TagModel

class FaceStore:
    def __init__(self):
        self.sio = None
        self.config = None
        self.lastTagsSent = []
        self.data = {}

    def setThreadInfo(self, sio, config):
        self.sio = sio
        self.config = config

    def add(self, fid, box, embedding_norm, gender,age):
        face = None
        shouldSendTags = False
        if str(fid) in self.data.keys():
            face = self.data[str(fid)]
            face.add(fid, box, embedding_norm, gender,age)
            shouldSendTags = (face.count() >= 30)
        else:
            face = FaceModel(fid, box,embedding_norm ,gender ,age)

        self.data[str(fid)] = face
        if shouldSendTags == True:
            self.sendTags()

        return face

    def get(self, fid):
        if str(fid) in self.data:
            return self.data[str(fid)]
        return None

    def remove(self, fid):
        if str(fid) in self.data:
            print("Se ha ido alguien")
            del self.data[str(fid)]
            if not self.data:
                print("No tiene caras")
                self.sendEmptyData()
            else:
                self.sendTags()

    def gender(self):
        man = 0
        woman = 0
        for key in self.data:
            face = self.data[key]
            if face.gender == 'M':
                man = man + 1
            else:
                woman = woman + 1

        if man == 0 and woman == 0:
            return ''
        elif man >= woman:
            return 'M'
        else:
            return 'F'
    
    def sendEmptyData(self):
        tag = TagModel()
        tags = tag.tags
        if self.sio is not None:
            print('Envío DEL tags: ', [])
            send_data = {
                'from': self.sio.sid,
                'to': 'site-'+ str(self.config['idsite']),
                'event': 'viewer.deltags',
                'data': tags
            }
            self.sio.emit('send:message', send_data)
    
    def sendTags(self):
        tags = self.getTagsToSend()
        if len(tags) > 0:
            if self.compareTags(tags, self.lastTagsSent) and self.sio is not None:
                print("Envío ADD tags: ", tags)
                send_data = {
                    'from': self.sio.sid,
                    'to': 'site-'+ str(self.config['idsite']),
                    'event': 'viewer.addtags',
                    'data': tags
                }
                self.lastTagsSent = tags
                self.sio.emit('send:message', send_data)
    
    def compareTags(self, tags, lastTags):
        diferent = []
        if len(tags) != len(lastTags):
            return True
        else:
            for i in tags:
                if i not in lastTags:
                    diferent.append(i)
        return len(diferent) > 0                
    
    def getTagsData(self): 
        tag = TagModel()
        tags_data = {}
        
        for key in self.data:
            face = self.data[key]
            tag_string = ''
            if face.gender == "M":
                tag_string += 'Hombre '
            else:
                tag_string += 'Mujer '
            tag_string += tag.ageStr(face)
            if tag_string not in tags_data:
                tags_data[tag_string] = 1
            else:
                tags_data[tag_string] += 1
                
        return tags_data        

    def getTagsToSend(self):
        tags_data = {}
        tags_data = self.getTagsData()
        array_tags = self.getArrayTagsData(tags_data)
        return array_tags
   
    def getArrayTagsData(self,tags_data):
        tag = TagModel()
        total_elems = len(self.data)
        dif = 0.85
        array_tags = []
        final_array = []
        for key in tags_data:
            if self.percentage(tags_data[key], total_elems) > 33:
                filtered_tag = tag.filter(key)
                array_tags.append(filtered_tag)
        if(len(array_tags) == 0):
            array_percentages = []
            for key in tags_data:
                array_percentages.append({"tag": key, "percentage": self.percentage(tags_data[key], total_elems)})
            percentages_sorted = sorted(array_percentages, key=lambda x: x['percentage'], reverse=True)
            for value in percentages_sorted:
                if value['percentage'] >= (percentages_sorted[0]['percentage'] * dif):
                    final_array.append(tag.filter(value['tag']))
        else:
            final_array = array_tags
        return final_array
    
    def percentage(self, val, total):
        return (val*100)/total
            
            


