from flask import Flask
import main

app=Flask(__name__)

@app.route('/')
def hello():
    return "Hello"

#in this file:
#create route for receiving wav post
#get wav data from post
#call def from main.py to check if watermelon
#return results or html for browser

#in html file:
#add script
#create start/stop for recording the wav file
#include wav as BLOB in post to route created in this file
