This is code to download image from csv url column and keep it in appropriate class folders.

filename="myntra_train_dataset.csv"
import pandas
train=pandas.read_csv(filename)
#Number of classes
class_num=len(train.Sub_category.unique())
#Unique class list
classes=list()
for i in range(0,class_num):
    print(train.Sub_category.unique()[i])
    classes.append(train.Sub_category.unique()[i])
#Create Folders as per classes.
for i in range(0,class_num):
    if not os.path.exists(classes[i]):
        os.makedirs(classes[i])

#Code to download files
import urllib.request
nulllist=list()
for filenumber in range(24589,train.shape[0]):
    if (pandas.isnull(train.Link_to_the_image[filenumber])!=True):
        filename=str(filenumber)+'.jpg'
        file_path=os.getcwd()+'\\'+train.Sub_category[filenumber]+'\\'+filename
        url=train.Link_to_the_image[filenumber]
        print(filenumber)
        try:
            urllib.request.urlretrieve(url, file_path)
        except urllib.error.HTTPError as err:
            print(err.code)
        except urllib.error.ContentTooShortError as err:
            import time
            time.sleep(10)
            urllib.request.urlretrieve(url, file_path)
    else:
        nulllist.append(filenumber)

print(nulllist)

