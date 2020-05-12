class Frame:
    def __init__(self, time, size, pict_type):
        self.time = time
        self.size = size
        self.pict_type = pict_type
    
    @staticmethod
    def get_fields():
        return ['time', 'size', 'pict_type']
