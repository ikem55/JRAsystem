import datetime as dt
import os
import urllib.request
import pandas as pd
from bs4 import BeautifulSoup
import glob
import shutil

import modules.util as mu
import my_config as mc


class JrdbDownload(object):
    """ JRDBデータをダウンロードするのに利用

      :param str base_uri: JRDBのURL
      :type str base_uri: str
      :param str jrdb_id: JRDBのID
      :type str jrdb_id: str

    """

    def __init__(self):
        self.base_uri = mc.BASE_URL
        self.jrdb_id = mc.JRDB_ID
        self.jrdb_pw = mc.JRDB_PW
        self.download_path = mc.return_jrdb_path() + "jrdb_data/download/"
        self.archive_path = mc.return_jrdb_path() + "jrdb_data/archive/"
        self.target_folder = 'member/datazip/Paci/2018/'
        self.filename = 'PACI181021.zip'
        self.url = self.base_uri + self.target_folder + self.filename
        mu.setup_basic_auth(self.base_uri, self.jrdb_id, self.jrdb_pw)

    def procedure_download_sokuho(self):
        """ 速報データのダウンロードをまとめた手順 """
        print("============== DOWNLOAD JRDB SOKUHO ====================")
        self.download_path = mc.return_jrdb_path() + "jrdb_data/sokuho/"
        self.archive_path = mc.return_jrdb_path() + "jrdb_data/sokuho/archive/"
        # typelist = ["TYB", "SED", "HJC"]
        typelist = ["SED", "HJC"]
        target_date = dt.date.today().strftime('%Y%m%d')
        print(target_date)
        for type in typelist:
            print("----------------" + type + "---------------")
            filename = type + target_date[2:8] + ".zip"
            url = target_date[0:4] + '/' + filename
            print(url)
            self.download_jrdb_file(type.title(), url, filename)
        for p in glob.glob(self.archive_path + "/*.zip"):
            if os.path.isfile(p):
                os.remove(p)

    def procedure_download(self):
        """ 通常データのダウンロードをまとめた手順  """
        print("============== DOWNLOAD JRDB ====================")
        typelist = ["PACI", "SED", "SKB", "HJC", "TYB"]
        for type in typelist:
            print("----------------" + type + "---------------")
            filelist = mu.get_downloaded_list(type, self.archive_path)
            target_df = self.get_jrdb_page(type)
            for index, row in target_df.iterrows():
                if row['filename'] not in filelist:
                    self.download_jrdb_file(
                        type.title(), row['url'], row['filename'])

    def download_jrdb_file(self, type, url, filename):
        """ 指定したJRDBファイルのダウンcdロードと解凍を実施

        :param str type: JRDBファイルの種類
        :type str type: str
        :param str url: 該当ファイルのURLのフルパス
        :type str url: str
        :param str filename: ファイル名
        :type str filename: str
        """
        target_file_url = self.base_uri + 'member/datazip/' + type + '/' + url
        print('Downloading ... {0} as {1}'.format(target_file_url, filename))
        urllib.request.urlretrieve(
            target_file_url, self.download_path + filename)
        mu.unzip_file(filename, self.download_path, self.archive_path)

    def get_jrdb_page(self, type):
        """ 指定したファイルタイプに対してページにアクセスして対象ファイルのリストを取得する

        :param str type: ファイルの種類
        :return: データフレーム
        """
        if type == "PACI":
            html = urllib.request.urlopen(
                'http://www.jrdb.com/member/datazip/Paci/index.html').read()
        elif type == "SED":
            html = urllib.request.urlopen(
                'http://www.jrdb.com/member/datazip/Sed/index.html').read()
        elif type == "SKB":
            html = urllib.request.urlopen(
                'http://www.jrdb.com/member/datazip/Skb/index.html').read()
        elif type == "HJC":
            html = urllib.request.urlopen(
                'http://www.jrdb.com/member/datazip/Hjc/index.html').read()
        elif type == "TYB":
            html = urllib.request.urlopen(
                'http://www.jrdb.com/member/datazip/Tyb/index.html').read()

        else:
            html = ""
        soup = BeautifulSoup(html, 'lxml')
        tables = soup.findAll('table')
        target_df = pd.DataFrame(index=[], columns=['filename', 'url'])
        for table in tables:
            for li in table.findAll('li'):
                if type == "PACI":
#                    if li.text[0:6] == "PACI16" or li.text[0:6] == "PACI17" or li.text[0:6] == "PACI18" or li.text[0:6] == "PACI19" or li.text[0:5] == "PACI2":
                    if li.text[0:5] == "PACI2":
                        sr = pd.Series(
                            [li.text.strip('\n'), li.a.attrs['href']], index=target_df.columns)
                        target_df = target_df.append(sr, ignore_index=True)
                elif type == "SED":
                    if li.text[0:4] == "SED2":
                        sr = pd.Series(
                            [li.text.strip('\n'), li.a.attrs['href']], index=target_df.columns)
                        target_df = target_df.append(sr, ignore_index=True)
                elif type == "SKB":
                    if li.text[0:4] == "SKB2":
                        sr = pd.Series(
                            [li.text.strip('\n'), li.a.attrs['href']], index=target_df.columns)
                        target_df = target_df.append(sr, ignore_index=True)
                elif type == "HJC":
                    if li.text[0:4] == "HJC2":
                        sr = pd.Series(
                            [li.text.strip('\n'), li.a.attrs['href']], index=target_df.columns)
                        target_df = target_df.append(sr, ignore_index=True)
                elif type == "TYB":
                    if li.text[0:4] == "TYB2":
                        sr = pd.Series(
                            [li.text.strip('\n'), li.a.attrs['href']], index=target_df.columns)
                        target_df = target_df.append(sr, ignore_index=True)
                else:
                    print("nothing")
        return target_df

    def move_file(self):
        """ downloadしたファイルを各フォルダに格納する """
        os.chdir(self.download_path)
        filelist = glob.glob("*.*")
        for file in filelist:
            if file[0:3] == "BAC":
                if os.path.exists("../BAC/" + file): os.remove("../BAC/" + file)
                shutil.move(file, "../BAC/")
            if file[0:3] == "CHA":
                if os.path.exists("../CHA/" + file): os.remove("../CHA/" + file)
                shutil.move(file, "../CHA/")

            if file[0:3] == "CYB":
                if os.path.exists("../CYB/" + file): os.remove("../CYB/" + file)
                shutil.move(file, "../CYB/")
            if file[0:3] == "HJC":
                if os.path.exists("../HJC/" + file): os.remove("../HJC/" + file)
                shutil.move(file, "../HJC/")
            if file[0:3] == "JOA":
                if os.path.exists("../JOA/" + file): os.remove("../JOA/" + file)
                shutil.move(file, "../JOA/")
            if file[0:3] == "KAB":
                if os.path.exists("../KAB/" + file): os.remove("../KAB/" + file)
                shutil.move(file, "../KAB/")
            if file[0:3] == "KKA":
                if os.path.exists("../KKA/" + file): os.remove("../KKA/" + file)
                shutil.move(file, "../KKA/")
            if file[0:3] == "KYI":
                if os.path.exists("../KYI/" + file): os.remove("../KYI/" + file)
                shutil.move(file, "../KYI/")
            if file[0:2] == "OT":
                if os.path.exists("../OT/" + file): os.remove("../OT/" + file)
                shutil.move(file, "../OT/")
            if file[0:2] == "OU":
                if os.path.exists("../OU/" + file): os.remove("../OU/" + file)
                shutil.move(file, "../OU/")
            if file[0:2] == "OW":
                if os.path.exists("../OW/" + file): os.remove("../OW/" + file)
                shutil.move(file, "../OW/")
            if file[0:2] == "OZ":
                if os.path.exists("../OZ/" + file): os.remove("../OZ/" + file)
                shutil.move(file, "../OZ/")
            if file[0:3] == "SED":
                if os.path.exists("../SED/" + file): os.remove("../SED/" + file)
                shutil.move(file, "../SED/")
            if file[0:3] == "SKB":
                if os.path.exists("../SKB/" + file): os.remove("../SKB/" + file)
                shutil.move(file, "../SKB/")
            if file[0:3] == "SRB":
                if os.path.exists("../SRB/" + file): os.remove("../SRB/" + file)
                shutil.move(file, "../SRB/")
            if file[0:3] == "TYB":
                if os.path.exists("../TYB/" + file): os.remove("../TYB/" + file)
                shutil.move(file, "../TYB/")
            if file[0:3] == "UKC":
                if os.path.exists("../UKC/" + file): os.remove("../UKC/" + file)
                shutil.move(file, "../UKC/")
            if file[0:3] == "ZED":
                if os.path.exists("../ZED/" + file): os.remove("../ZED/" + file)
                shutil.move(file, "../ZED/")
            if file[0:3] == "ZKB":
                if os.path.exists("../ZKB/" + file): os.remove("../ZKB/" + file)
                shutil.move(file, "../ZKB/")

    def delete_temp_text_file(self):
        """ 翌日データ処理用。中途半端なファイル（textファイル）を削除する """
        target_folder = ["BAC", "CHA", "CYB", "HJC", "JOA", "KAB", "KKA", "KYI", "OT", "OU", "OW", "OZ", "SED", "SKB", "UKC", "ZED", "ZKB"]
        for target in target_folder:
            target_file = mc.return_jrdb_path() + target + "/*.txt"
            print(target_file)
            for file in glob.glob(target_file):
                os.remove(file)

