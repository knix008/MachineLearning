import sys
from PyQT5.QtWidget import *
import requests
from bs4 import BeautifulSoup

class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.le = QLineEdit()
        self.le.setPlaceholderText("Enter your search word")
        self.le.returnPressed.connect(self.crawl_news)

        self.btn = QPushButton("Search")
        self.btn.clicked.connected(self.crawl_news)