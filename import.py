import requests
import csv

MAX_IMG = 5

with open ('cat_2.csv') as images:
    images = csv.reader(images)
    img_count = 1
    for image in images:
        try:
            r = requests.get(image[0],stream = True)
            headers = r.headers
            content_type = ""
            if(headers.get('Content-Type')):
                content_type = headers["Content-Type"]
            elif(headers.get("content-type")):
                content_type = headers["content-type"]
            else:
                continue
            print("Content type " + content_type)
            if r.status_code == 200 and content_type == 'image/jpeg':
                with open('train/cat/image_'+str(img_count+94)+'.jpg', 'wb') as f:
                    for chunk in r:
                        f.write(chunk)
                    f.close()
                    print("Downloaded imb number " + str(img_count))
                if(img_count > MAX_IMG):
                    break
                img_count += 1
            else:
                continue
        except requests.ConnectionError as exception:
            print("URL does not exist on Internet")
