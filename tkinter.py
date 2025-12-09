#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tkinter as tk
from tkinter import messagebox


# In[4]:


def close():
    pass


# In[ ]:


root = tk.Tk()
root.title("Virus Warning")
root.attributes("-fullscreen", True)
root.attributes("-topmost", True)

root.protocol("WM_DELETE_WINDOW", close)

label = tk.Label(root, text="Virus Detected! You cannot close this window",  font=("Calibri", 14), fg="red")
label.pack(expand=True)

root.mainloop()


# In[ ]:




