# dental-implants
This Python code implements a chatbot that answers questions related to dental implants. The chatbot is based on LLama2 language model, which is a powerful and versatile natural language processing model.

Before running the code, you need to ensure that Python has been installed. My python version is Python 3.8.16.

The link for downloading Python
https://www.python.org/downloads/

Python 3 Installation & Setup Guide
https://realpython.com/installing-python/
## Notes
1.There are some python files, folder and requirements.txt.

a) Folder:

vector: store the database

b) Python files:

prepare.ipynb: preparation of the database.

train.py: model storage.

main.py: front interface and main function.

c) requirements.txt file: list all Python libraries.

2.The requirements.txt file should list all Python libraries that your notebooks depend on, and they will be installed using:

```pip install -r requirements.txt```

3.There are some steps you should edit by yourself.

a) Due to the large size of the model, you can download it using this link:https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q4_0.bin

Also, if you download this model, you can edit the path in train.py. My model path is "F:/test/llama-2-7b-chat.ggmlv3.q4_0.bin", you should use your path instead.

<img src="1698165072554.png" width="500px" height="140px">

b) Edit the path of the vector folder in main.py, you should use your own path.

<img src="1698165502554.png" width="350px" height="30px">

4.Use the command in the terminal

```streamlit run main.py```

After that, you can now view your streamlit app in your browser.
