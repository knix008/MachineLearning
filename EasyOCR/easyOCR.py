import math
import time
import datetime
import easyocr

reader = easyocr.Reader(['ko','en'], gpu=True) # this needs to run only once to load the model into memory
start = time.time()
result = reader.readtext('Test02.png')
end = time.time()

for res in result:
  coord=res[0]
  text=res[1]
  conf=res[2]
  print(text)

sec = (end - start)
result = datetime.timedelta(seconds=sec)
print("The Ellapsed Time ", result)