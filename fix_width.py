with open("app/app.py", "r") as f:
    content = f.read()

content = content.replace("use_container_width=True", 'width="stretch"')
content = content.replace("use_container_width=False", 'width="content"')

with open("app/app.py", "w") as f:
    f.write(content)
