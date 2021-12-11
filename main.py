import lib
import csv
import os

image_path = './images'
list_images_path = os.listdir(image_path)

header = ['Image', 'Brand']
data = []


def get_text(img_path):
    reader = lib.Detector(['en'])
    return reader.read(img_path)


for i in range(len(list_images_path)):

    im_1_path = image_path + '/' + list_images_path[i]
    a = get_text(im_1_path)
    data.append([list_images_path[i]])
    detected = False
    for j in range(len(a)):
        if j < len(a)-1:
            if detected == False and ('phuc long' in a[j][1].lower() or ('phuc long' in (a[j][1] + ' ' + a[j+1][1]).lower())):
                data[i].append('PHUC LONG')
                detected = True
            if detected == False and ('starbucks' in a[j][1].lower() or ('starbucks' in (a[j][1] + ' ' + a[j+1][1]).lower())):
                data[i].append('Starbulks')
                detected = True
            if detected == False and ('highlands' in a[j][1].lower() or ('highlands' in (a[j][1] + ' ' + a[j+1][1]).lower())):
                data[i].append('Highlands')
                detected = True
            if detected == False and ('circle k' in a[j][1].lower() or ('circle k' in (a[j][1] + ' ' + a[j+1][1]).lower())):
                data[i].append('Circle K')
                detected = True
    if detected == False:
        data[i].append('others')

with open('./output.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(data)

print(list_images_path)
