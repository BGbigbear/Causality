import google.generativeai as genai
import os

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

genai.configure(api_key="AIzaSyDVNcTeCipNSIKtMQ-fKea8Vy-gi_l6xeY")

# model = genai.GenerativeModel("gemma-2-27b-it")
# response = model.generate_content("Write a story about a magic backpack.")
# print(response.text)
print([m.name for m in genai.list_models()])
