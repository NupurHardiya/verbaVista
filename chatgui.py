import nltk
from nltk.stem import WordNetLemmatizer
from tkinter import Tk, Canvas, Frame, Label, ALL, Button, Entry, END, Scrollbar, N, S, E, W, LEFT, RIGHT, PhotoImage
from tkinter.constants import DISABLED, NORMAL
from threading import Thread, Event
from time import sleep
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import tkinter as tk

# Load ChatBot model and data
lemmatizer = WordNetLemmatizer()
model = load_model('D:\MInor_project\chatbot\chatbot_model.h5')
intents = json.loads(open('D:\MInor_project\chatbot\intents.json').read())
words = pickle.load(open('D:\MInor_project\chatbot\words.pkl', 'rb'))
classes = pickle.load(open('D:\MInor_project\chatbot\classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = get_response(ints, intents)
    return res


class ChatGUI:
    def __init__(self):
        self.root = Tk()
        self.root.title("cityCompeer")
        self.root.protocol("WM_DELETE_WINDOW", self.close_handler)
        self.root.resizable(0, 0)

        self.canvas = Canvas(self.root, width=800, height=500, bg="white")
        self.canvas.grid(row=0, column=0)
        self.canvas_scroll_y = Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.canvas_scroll_y.grid(row=0, column=1, sticky=(N, S, W, E))
        self.canvas.configure(yscrollcommand=self.canvas_scroll_y.set)
        
        self.bot_image = PhotoImage(file="D:\\MInor_project\\chatbot\\robot.png")
        self.user_image = PhotoImage(file="D:\\MInor_project\\chatbot\\user.png")


        self.user_input_box = Entry(self.root)
        self.user_input_box.grid(row=1, column=0, padx=5, pady=10, ipady=8, ipadx=290, sticky=W)
        self.user_input_box.bind("<Return>", self.user_input_handler)
        send_image = PhotoImage(file="send.png")  # Update with the actual path
        self.send_button = Button(self.root, image=send_image, command=lambda: self.user_input_handler(None))
        self.send_button.grid(row=1, column=0, sticky=E)

        self.last_bubble = None
        self.user_thread = None
        self.bot_thread = None
        self.thread_event = Event()

        self.root.mainloop()

    def close_handler(self):
        if self.user_thread and self.user_thread.is_alive():
            self.bot_thread._tstate_lock.release_lock()
            self.user_thread._stop()
        if self.bot_thread and self.bot_thread.is_alive():
            self.bot_thread._tstate_lock.release_lock()
            self.bot_thread._stop()
        self.root.destroy()

    def show_bubble(self, message="", bot=True):
        if self.last_bubble:
            self.canvas.move(ALL, 0, -(self.last_bubble.winfo_height() + 10))
        bg_color = "light blue" if bot else "light grey"
        color = "black"  # if bot else "white"
        frame = Frame(self.canvas, bg=bg_color)
        self.last_bubble = frame

        widget = self.canvas.create_window(50 if bot else 700, 440, window=frame, anchor='nw' if bot else 'ne')

        chat_label = Label(frame, text=message, wraplength=600, justify=LEFT if bot else RIGHT,
                           font=("Helvetica", 12), bg=bg_color, fg=color)
        chat_label.pack(anchor="w" if bot else 'e', side=LEFT if bot else RIGHT, pady=10, padx=10)

        self.root.update_idletasks()

        self.canvas.create_polygon(self.draw_triangle(widget, bot), fill=bg_color, outline=bg_color)
        self.add_icon(widget, bot)

    def add_icon(self, widget, bot=True):
        x1, y1, x2, y2 = self.canvas.bbox(widget)
        if bot:
            self.canvas.create_image(x1 - 72, y2, image=self.bot_image, anchor=W)
        else:
            self.canvas.create_image(x2 + 72, y2, image=self.user_image, anchor=E)

    def draw_triangle(self, widget, bot=True):
        x1, y1, x2, y2 = self.canvas.bbox(widget)
        if bot:
            return x1, y2 - 10, x1 - 10, y2, x1, y2
        return 700, y2 - 10, 700, y2, 710, y2

    def add_user_message(self, message):
        self.user_input_box.config(state=DISABLED)
        self.send_button.config(state=DISABLED)
        self.show_bubble(message, bot=False)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.thread_event.set()

    def add_bot_message(self, message):
        self.show_bubble(message, bot=True)
        self.user_input_box.config(state=NORMAL)
        self.send_button.config(state=NORMAL)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def process_message(self, message):
        bot_message = chatbot_response(message)
        while not self.thread_event.is_set():
            sleep(0.1)
        self.add_bot_message(bot_message)

    def user_input_handler(self, event):
        message = self.user_input_box.get()

        if not message:
            return

        self.thread_event.clear()
        self.user_thread = Thread(target=self.add_user_message, args=(message,))
        self.bot_thread = Thread(target=self.process_message, args=(message,))
        self.user_thread.start()
        self.bot_thread.start()

        self.user_input_box.delete(0, END)

if __name__ == "__main__":
    chat_gui = ChatGUI()
