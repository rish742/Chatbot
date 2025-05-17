import tkinter as tk
from chatbot import get_response_from_bot

# Initialize main window
root = tk.Tk()
root.title("Chatbot")
root.geometry("400x500")
root.resizable(width=False, height=False)

# Function to handle sending messages
def send_message():
    user_message = entry_box.get("1.0", tk.END).strip()
    if user_message == "":
        return

    chat_log.config(state=tk.NORMAL)
    chat_log.insert(tk.END, "You: " + user_message + '\n\n')
    chat_log.config(foreground="#442265", font=("Verdana", 12))

    response = get_response_from_bot(user_message)
    chat_log.insert(tk.END, "Bot: " + response + '\n\n')

    entry_box.delete("1.0", tk.END)
    chat_log.config(state=tk.DISABLED)
    chat_log.yview(tk.END)

# Chat log area
chat_log = tk.Text(root, bd=0, bg="white", height="8", width="50", font="Arial")
chat_log.config(state=tk.DISABLED)

# Scroll bar
scroll_bar = tk.Scrollbar(root, command=chat_log.yview)
chat_log['yscrollcommand'] = scroll_bar.set

# Entry box where user types messages
entry_box = tk.Text(root, bd=0, bg="white", width="29", height="5", font="Arial")

# Send button
send_button = tk.Button(
    root, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
    bd=0, bg="#32de97", activebackground="#3c9d9b", fg='black',
    command=send_message
)

# Place components in the window
scroll_bar.place(x=376, y=6, height=386)
chat_log.place(x=6, y=6, height=386, width=370)
entry_box.place(x=128, y=401, height=90, width=265)
send_button.place(x=6, y=401, height=90)

# Start the GUI
root.mainloop()
