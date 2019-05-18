from bs4 import BeautifulSoup as bs
from selenium import webdriver
import requests
import re
from itertools import chain
from collections import defaultdict
import traceback
from time import sleep
import os


class WebConnector(object):

    def __init__(self, url):
        self.url = url
        self.page = requests.get(self.url).text


class Parser(WebConnector):

    def __init__(self):
        self.URL = "http://cvil.eecs.yorku.ca/projects/public_html/illuminant/illuminant.html"
        super(Parser, self).__init__(self.URL)
        self.cameras = ["Canon1D", "Canon600", "Fujifilm",
                        "Nikon", "Olympus", "Panasonic", "Samsung", "Sony"]
        self.png = ["PNG_"+i for i in self.cameras]
        self.chk = ["CHK_"+i for i in self.cameras]
        self.links = defaultdict()

    def get_links(self):

        soup = bs(self.page, "lxml")
        for camera in chain(self.png, self.chk):
            for link in soup.find_all("a", id=re.compile(camera)):
                self.links[link.attrs["id"]] = link.attrs["href"]

        return self.links.values()


class Page(object):

    def __init__(self, url, path):
        self.url = url
        self.path = path

    def __enter__(self):
        options = webdriver.ChromeOptions()
        options.add_experimental_option("prefs", {
            "download.default_directory": self.path,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        })
        self.driver = webdriver.Chrome(chrome_options=options)
        self.driver.implicitly_wait(30)
        self.driver.get(self.url)

        return self

    def __exit__(self, exeption_type, exception_value, trace):
        if exeption_type is not None:
            traceback.print_exception(
                exeption_type, exception_value, trace)
        self.driver.close()

    def download(self):

        self.driver.find_element_by_css_selector(
            ".btn.btn-primary.btn-xs.btn-compat-download").click()
        sleep(15)

        while re.search("crdownload", "|".join(os.listdir(path))):
            sleep(60)


if __name__ == "__main__":
    path = r"absolute/path/to/you/download/folder"
    parser = Parser()
    for i, link in enumerate(parser.get_links()):
        with Page(link, path) as p:
            print("Downloading file {}/42".format(i+1))
            p.download()
