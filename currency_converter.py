#!/usr/bin/env python
# coding: utf-8

# In[6]:


pip install requests


# In[18]:


import requests

def convert_currency(amount, from_currency, to_currency):
    url = f"https://api.frankfurter.app/latest?amount={amount}&from={from_currency}&to={to_currency}"
    response = requests.get(url)

    if response.status_code != 200:
        return None

    data = response.json()
    return data["rates"][to_currency]


print("Currency Converter")

amount = float(input("Enter amount: "))
from_currency = input("From currency (e.g., USD, INR, EUR): ").upper()
to_currency = input("To currency (e.g., USD, INR, EUR): ").upper()

converted = convert_currency(amount, from_currency, to_currency)

if converted is not None:
    print(f"\n{amount} {from_currency} = {converted:.2f} {to_currency}")
else:
    print("\n Conversion failed. Invalid currency code or API error.")


# In[ ]:




