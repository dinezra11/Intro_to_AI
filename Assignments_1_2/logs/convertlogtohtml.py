# In the console run first "python run.py | tee rt_a_star_log.txt" to get the colored logs as a text file.
# After that, run this script to convert the txt file into html.
# From html we can convert to pdf to prepare it for submission.

from ansi2html import Ansi2HTMLConverter
with open("rt_a_star_log.txt", "r", encoding="utf-16") as f:
    ansi = f.read()
conv = Ansi2HTMLConverter()
html = conv.convert(ansi)
with open("rt_a_star_log.html", "w", encoding="utf-8") as f:
    f.write(html)