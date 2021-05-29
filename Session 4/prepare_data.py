import tarfile
import os
import envoy
import urllib.request
import shutil

url = "https://drive.google.com/file/d/1nhBeJoVyAs1q9Ld53b_kotx6DuS2sfYl/view?usp=sharing"

path_save = "data\\save\\20news-bydate.tar.gz"

if not os.path.exists(path_save):
    urllib.request.urlretrieve(url,path_save)


# file = "data\\20news-bydate.tar.gz"
# path_save = "data\\data_20news"

if (path_save.endswith("tar.gz")):
    envoy.run("tar -xzf %s -C %s" % (path_save, "data"))

shutil.unpack_archive(path_save, 'data')